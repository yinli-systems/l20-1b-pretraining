#!/usr/bin/env python3
"""Pair-atomic cross-fit development of a fixed-token real-image readout.

The language model, its existing LoRA adapter, SigLIP2, and the Stage-D bridge
core remain immutable.  Each fold initializes a fresh rank-64 query readout and
trains it on the other four folds.  This is a method-development diagnostic,
not a final model-training or benchmark protocol.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent
CANDIDATES = ("sitting", "standing")
VIEWS = ("full", "marked", "crop")
CONDITIONS = ("true_image", "paired_swap", "blank")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for pair in pairs:
        for side, donor_side in (("image_a", "image_b"), ("image_b", "image_a")):
            image = pair[side]
            donor = pair[donor_side]
            if image["answer"] == donor["answer"]:
                raise ValueError(f"pair lacks opposite labels: {pair['pair_id']}")
            targets.append(
                {
                    "target_id": f"{pair['pair_id']}:{side}",
                    "donor_target_id": f"{pair['pair_id']}:{donor_side}",
                    "pair_id": pair["pair_id"],
                    "fold": int(pair["crossfit"]["fold"]),
                    "side": side,
                    "class_name": pair["class_name"],
                    "expected": image["answer"],
                    "image": image,
                    "donor": donor,
                }
            )
    return sorted(targets, key=lambda row: row["target_id"])


def folds_from_pairs(pairs: list[dict[str, Any]], fold_count: int) -> dict[int, list[dict[str, Any]]]:
    grouped = {fold: [] for fold in range(fold_count)}
    seen: set[str] = set()
    for pair in sorted(pairs, key=lambda row: row["pair_id"]):
        pair_id = str(pair["pair_id"])
        if pair_id in seen:
            raise ValueError(f"duplicate pair id: {pair_id}")
        seen.add(pair_id)
        fold = int(pair["crossfit"]["fold"])
        if fold not in grouped:
            raise ValueError(f"invalid fold {fold}: {pair_id}")
        grouped[fold].append(pair)
    if any(not rows for rows in grouped.values()):
        raise ValueError("every cross-fit fold must be non-empty")
    return grouped


def region_attention_objective(attention, regions: list[list[int]]):
    """Negative log attention mass within each audited region."""
    import torch

    if attention.ndim != 3 or attention.shape[1] != len(CANDIDATES):
        raise ValueError("attention must be [targets, candidates, visual_tokens]")
    if len(regions) != attention.shape[0] or any(not region for region in regions):
        raise ValueError("one non-empty region is required per target")
    mean_attention = attention.float().mean(dim=1)
    masses = torch.stack(
        [mean_attention[index, region].sum() for index, region in enumerate(regions)]
    )
    loss = -masses.clamp_min(1e-8).log().mean()
    top1 = torch.tensor(
        [int(mean_attention[index].argmax()) in region for index, region in enumerate(regions)],
        dtype=torch.float32,
        device=attention.device,
    ).mean()
    candidate_gap = (attention[:, 0] - attention[:, 1]).abs().max()
    return loss, masses.mean(), top1, candidate_gap


def pair_ranking_objective(scores, targets: list[dict[str, Any]], margin: float):
    """Require standing evidence to rank above sitting evidence within each pair."""
    import torch
    import torch.nn.functional as F

    if scores.shape != (len(targets), len(CANDIDATES)):
        raise ValueError("score shape does not match targets")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[target["pair_id"]].append(index)
    separations = []
    for pair_id, indices in sorted(grouped.items()):
        if len(indices) != 2:
            raise ValueError(f"incomplete pair in batch: {pair_id}")
        by_answer = {targets[index]["expected"]: index for index in indices}
        if set(by_answer) != set(CANDIDATES):
            raise ValueError(f"pair does not contain both labels: {pair_id}")
        posture_logodds = scores[:, 1] - scores[:, 0]
        separations.append(
            posture_logodds[by_answer["standing"]] - posture_logodds[by_answer["sitting"]]
        )
    separation = torch.stack(separations)
    return F.relu(float(margin) - separation).mean(), separation.mean(), (separation > 0).float().mean()


def validate_protocol(path: Path) -> dict[str, Any]:
    from train_stage_a_full_token import sha256_file

    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_openimages_posture_readout_crossfit_development_v1":
        raise RuntimeError("posture readout protocol is not authorized")
    if protocol.get("formal_test_present") is not False:
        raise RuntimeError("formal test must remain absent")
    if protocol.get("model_parameter_updates_authorized") is not True:
        raise RuntimeError("readout development was not authorized")
    source = protocol["source_code"]
    checks = (
        (Path(__file__), "trainer_sha256"),
        (ROOT / "test_train_openimages_posture_readout_crossfit_v1.py", "test_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
        (ROOT / "counterfactual_losses.py", "losses_sha256"),
        (ROOT / "evaluate_naturalbench.py", "candidate_scoring_sha256"),
        (ROOT / "evaluate_openimages_posture_zero_shot_v1.py", "posture_evaluator_sha256"),
    )
    for file_path, key in checks:
        if sha256_file(file_path) != source[key]:
            raise RuntimeError(f"source hash mismatch: {file_path.name}")
    for key in ("manifest", "probe_receipt", "zero_shot_receipt"):
        item = protocol["data"][key]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"data evidence hash mismatch: {key}")
    parents = protocol["parents"]
    if sha256_file(Path(parents["bridge"])) != parents["bridge_sha256"]:
        raise RuntimeError("parent bridge hash mismatch")
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(Path(parents["language_adapter"]) / name) != expected:
            raise RuntimeError(f"parent language adapter mismatch: {name}")
    arms = protocol["arms"]
    if set(arms) != {"query_ce", "grounded_query"}:
        raise RuntimeError("exactly two frozen-core readout arms are required")
    if arms["query_ce"]["loss_weights"] != {"answer": 1.0, "address": 0.0, "pair_rank": 0.0}:
        raise RuntimeError("query-CE control weights changed")
    grounded = arms["grounded_query"]["loss_weights"]
    if grounded["answer"] != 1.0 or grounded["address"] <= 0 or grounded["pair_rank"] <= 0:
        raise RuntimeError("grounded arm must include address and pair-ranking supervision")
    return protocol


def verify_images(targets: list[dict[str, Any]]) -> dict[str, Any]:
    from train_stage_a_full_token import sha256_file

    seen: dict[str, str] = {}
    mismatches = []
    for target in targets:
        record = target["image"]
        image_id = record["image_id"]
        if image_id in seen:
            continue
        path = Path(record["image_path"])
        actual = sha256_file(path) if path.is_file() else None
        seen[image_id] = actual or "missing"
        if actual != record["image_sha256"]:
            mismatches.append(
                {"image_id": image_id, "path": str(path), "expected": record["image_sha256"], "actual": actual}
            )
    return {"checked_unique_images": len(seen), "mismatches": mismatches, "all_match_manifest": not mismatches}


def cache_vision_features(targets, views, protocol, processor, vision, dtype):
    import torch

    from evaluate_openimages_posture_zero_shot_v1 import render_view
    from modeling import vision_features

    output: dict[str, dict[str, Any]] = {view: {} for view in views}
    batch_size = int(protocol["feature_cache"]["batch_images"])
    padding = float(protocol["feature_cache"]["crop_padding_fraction"])
    layer = int(protocol["architecture"]["vision_feature_layer"])
    with torch.inference_mode():
        for view in views:
            for start in range(0, len(targets), batch_size):
                batch = targets[start : start + batch_size]
                images = [render_view(row["image"], view, padding) for row in batch]
                pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
                    "cuda", dtype=dtype, non_blocking=True
                )
                features = vision_features(vision, pixels, layer).to("cpu", dtype=dtype)
                for row, feature in zip(batch, features):
                    output[view][row["target_id"]] = feature.contiguous()
                print(
                    f"READOUT_FEATURES view={view} targets={min(start + len(batch), len(targets))}/{len(targets)}",
                    flush=True,
                )
    blank_pixels = processor(images=[Image.new("RGB", (224, 224), (127, 127, 127))], return_tensors="pt")[
        "pixel_values"
    ].to("cuda", dtype=dtype)
    with torch.inference_mode():
        blank = vision_features(vision, blank_pixels, layer)[0].to("cpu", dtype=dtype).contiguous()
    return output, blank


def candidate_forward(language, bridge, tokenizer, targets, features, view, max_text_tokens):
    import torch

    from counterfactual_losses import masked_sequence_logprob
    from evaluate_naturalbench import collate_candidates
    from evaluate_openimages_posture_zero_shot_v1 import prompt_for

    prompts, answers = [], []
    for target in targets:
        prompt = prompt_for(view, target["class_name"])
        for candidate in CANDIDATES:
            prompts.append(prompt)
            answers.append(candidate)
    input_ids, text_mask, labels = collate_candidates(
        tokenizer, prompts, answers, int(max_text_tokens)
    )
    input_ids = input_ids.cuda(non_blocking=True)
    text_mask = text_mask.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    feature_batch = torch.stack(features).to("cuda", non_blocking=True).repeat_interleave(2, dim=0)
    with torch.no_grad():
        text = language.get_input_embeddings()(input_ids)
    inputs, attention_mask, injected_labels, readout = bridge.inject_with_readout(
        text.detach(), text_mask, labels, feature_batch
    )
    if injected_labels is None or readout is None:
        raise RuntimeError("query readout did not return labels and attention")
    logits = language(inputs_embeds=inputs, attention_mask=attention_mask).logits[:, :-1]
    shifted = injected_labels[:, 1:]
    token_mask = shifted != -100
    sequence_scores = masked_sequence_logprob(
        logits, shifted.masked_fill(~token_mask, 0), token_mask
    )
    return sequence_scores.reshape(len(targets), 2), readout.reshape(len(targets), 2, -1)


def initialize_bridge(protocol, arm_name, fold, dtype):
    import torch
    from safetensors.torch import load_file

    from modeling import bridge_from_architecture, load_bridge_parent

    seed = int(protocol["optimization"]["seed"]) + int(fold)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    missing = load_bridge_parent(
        bridge,
        load_file(protocol["parents"]["bridge"], device="cpu"),
        allow_new_compressor=True,
    )
    if not missing or not all(name.startswith("query_readout.") for name in missing):
        raise RuntimeError(f"unexpected parent initialization for {arm_name}/fold-{fold}: {missing}")
    bridge.requires_grad_(False)
    if bridge.query_readout is None:
        raise RuntimeError("query readout is absent")
    bridge.query_readout.requires_grad_(True)
    bridge.train()
    return bridge, missing


def train_fold(protocol, arm_name, fold, train_pairs, feature_cache, language, tokenizer, dtype, output_root, epoch_cap):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from evaluate_openimages_posture_zero_shot_v1 import bbox_grid_targets
    from train_stage_a_full_token import cosine_lr, write_json_atomic

    arm = protocol["arms"][arm_name]
    weights = arm["loss_weights"]
    optimization = protocol["optimization"]
    bridge, missing = initialize_bridge(protocol, arm_name, fold, dtype)
    parameters = list(bridge.query_readout.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimization["readout_learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        fused=True,
    )
    epochs = min(int(epoch_cap or optimization["epochs"]), int(optimization["epochs"]))
    batch_pairs = int(optimization["batch_pairs"])
    steps_per_epoch = math.ceil(len(train_pairs) / batch_pairs)
    max_steps = epochs * steps_per_epoch
    warmup = max(1, round(max_steps * float(optimization["warmup_ratio"])))
    seed = int(optimization["seed"]) + fold
    log_path = output_root / arm_name / f"fold-{fold}" / "train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    step = 0
    final_event = None
    for epoch in range(epochs):
        order = list(train_pairs)
        random.Random(seed + epoch * 1009).shuffle(order)
        for start in range(0, len(order), batch_pairs):
            pair_batch = order[start : start + batch_pairs]
            batch_targets = flatten_pairs(pair_batch)
            features = [feature_cache["marked"][row["target_id"]] for row in batch_targets]
            scores, attention = candidate_forward(
                language,
                bridge,
                tokenizer,
                batch_targets,
                features,
                "marked",
                optimization["max_text_tokens"],
            )
            target_indices = torch.tensor(
                [CANDIDATES.index(row["expected"]) for row in batch_targets],
                device="cuda",
                dtype=torch.long,
            )
            answer_loss = F.cross_entropy(scores, target_indices)
            regions = [bbox_grid_targets(row["image"]["bbox"])[1] for row in batch_targets]
            address_loss, region_mass, region_top1, candidate_gap = region_attention_objective(
                attention, regions
            )
            rank_loss, rank_separation, rank_accuracy = pair_ranking_objective(
                scores, batch_targets, float(optimization["pair_margin"])
            )
            loss = (
                float(weights["answer"]) * answer_loss
                + float(weights["address"]) * address_loss
                + float(weights["pair_rank"]) * rank_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite readout loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(optimization["gradient_clip_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite readout gradient")
            step += 1
            scale = cosine_lr(
                step,
                max_steps,
                warmup,
                float(optimization["minimum_learning_rate_ratio"]),
            )
            learning_rate = float(optimization["readout_learning_rate"]) * scale
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.step()
            elapsed = time.monotonic() - started
            final_event = {
                "arm": arm_name,
                "fold": fold,
                "epoch": epoch + 1,
                "optimizer_step": step,
                "total_loss": float(loss.detach()),
                "answer_loss": float(answer_loss.detach()),
                "address_loss": float(address_loss.detach()),
                "pair_rank_loss": float(rank_loss.detach()),
                "candidate_accuracy": float((scores.argmax(dim=-1) == target_indices).float().mean()),
                "region_attention_mass": float(region_mass.detach()),
                "region_attention_top1": float(region_top1.detach()),
                "candidate_attention_max_gap": float(candidate_gap.detach()),
                "pair_rank_separation": float(rank_separation.detach()),
                "pair_rank_accuracy": float(rank_accuracy.detach()),
                "gradient_norm": float(grad_norm),
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(final_event, sort_keys=True) + "\n")
        print(
            f"READOUT_TRAIN arm={arm_name} fold={fold} epoch={epoch + 1}/{epochs} step={step}/{max_steps} loss={final_event['total_loss']:.5f}",
            flush=True,
        )
    checkpoint = log_path.parent / "query_readout.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in bridge.query_readout.state_dict().items()},
        checkpoint,
    )
    receipt = {
        "status": "complete_crossfit_fold_development",
        "arm": arm_name,
        "fold": fold,
        "train_pairs": len(train_pairs),
        "train_pair_ids": sorted(row["pair_id"] for row in train_pairs),
        "epochs": epochs,
        "optimizer_steps": step,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "missing_parent_keys": missing,
        "checkpoint": str(checkpoint),
        "final_event": final_event,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_atomic(log_path.parent / "receipt.json", receipt)
    return bridge, receipt


def score_targets(protocol, bridge, language, tokenizer, targets, feature_cache, blank_feature):
    import torch

    from evaluate_openimages_posture_zero_shot_v1 import bbox_grid_targets

    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        view: {condition: [] for condition in CONDITIONS} for view in VIEWS
    }
    batch_size = int(protocol["evaluation"]["batch_targets"])
    with torch.inference_mode():
        for view in VIEWS:
            for condition in CONDITIONS:
                for start in range(0, len(targets), batch_size):
                    batch = targets[start : start + batch_size]
                    if condition == "blank":
                        features = [blank_feature for _ in batch]
                    elif condition == "paired_swap":
                        features = [feature_cache[view][row["donor_target_id"]] for row in batch]
                    else:
                        features = [feature_cache[view][row["target_id"]] for row in batch]
                    scores, attention = candidate_forward(
                        language,
                        bridge,
                        tokenizer,
                        batch,
                        features,
                        view,
                        protocol["optimization"]["max_text_tokens"],
                    )
                    scores_cpu = scores.float().cpu()
                    attention_cpu = attention.float().cpu()
                    for index, target in enumerate(batch):
                        left, right = (float(value) for value in scores_cpu[index])
                        predicted = None if left == right else CANDIDATES[int(right > left)]
                        expected_index = CANDIDATES.index(target["expected"])
                        correct_score = (left, right)[expected_index]
                        incorrect_score = (right, left)[expected_index]
                        address = None
                        if condition == "true_image" and view in {"full", "marked"}:
                            mean_attention = attention_cpu[index].mean(dim=0)
                            center, region = bbox_grid_targets(target["image"]["bbox"])
                            prediction = int(mean_attention.argmax())
                            address = {
                                "bbox_center_cell": center,
                                "bbox_region_cells": region,
                                "predicted_cell": prediction,
                                "bbox_center_top1": prediction == center,
                                "bbox_region_top1": prediction in region,
                                "bbox_region_attention_mass": float(mean_attention[region].sum()),
                                "candidate_attention_max_gap": float(
                                    (attention_cpu[index, 0] - attention_cpu[index, 1]).abs().max()
                                ),
                            }
                        records[view][condition].append(
                            {
                                "target_id": target["target_id"],
                                "pair_id": target["pair_id"],
                                "fold": target["fold"],
                                "side": target["side"],
                                "class_name": target["class_name"],
                                "expected": target["expected"],
                                "donor_expected": target["donor"]["answer"],
                                "source_image_id": None if condition == "blank" else (
                                    target["donor"]["image_id"] if condition == "paired_swap" else target["image"]["image_id"]
                                ),
                                "candidate_sequence_logprobs": {CANDIDATES[0]: left, CANDIDATES[1]: right},
                                "predicted": predicted,
                                "correct": predicted == target["expected"],
                                "correct_minus_incorrect_logprob": correct_score - incorrect_score,
                                "address": address,
                            }
                        )
    return records


def summarize_records(records, protocol):
    from evaluate_openimages_posture_zero_shot_v1 import (
        compare_conditions,
        strict_bidirectional_flip,
        summarize_condition,
    )

    metrics = {}
    for view in VIEWS:
        true_records = records[view]["true_image"]
        metrics[view] = {
            "conditions": {
                condition: summarize_condition(records[view][condition], protocol)
                for condition in CONDITIONS
            },
            "true_minus_paired_swap_accuracy_pp": compare_conditions(
                true_records, records[view]["paired_swap"], protocol
            ),
            "true_minus_blank_accuracy_pp": compare_conditions(
                true_records, records[view]["blank"], protocol
            ),
            "strict_bidirectional_answer_flip_percent": strict_bidirectional_flip(
                true_records, records[view]["paired_swap"], protocol
            ),
        }
    return metrics


def compare_arms(left_records, right_records, protocol):
    from evaluate_openimages_posture_zero_shot_v1 import interval

    output = {}
    for view in VIEWS:
        left = {row["target_id"]: row for row in left_records[view]["true_image"]}
        right = {row["target_id"]: row for row in right_records[view]["true_image"]}
        if left.keys() != right.keys():
            raise ValueError("arm target sets differ")
        ids = sorted(left)
        output[view] = interval(
            [100.0 * (float(left[key]["correct"]) - float(right[key]["correct"])) for key in ids],
            [left[key]["pair_id"] for key in ids],
            protocol,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke-fold", type=int)
    parser.add_argument("--smoke-arm", choices=("query_ce", "grounded_query"))
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.smoke_fold is None) != (args.smoke_arm is None):
        raise ValueError("smoke fold and arm must be supplied together")
    if args.max_epochs is not None and args.max_epochs < 1:
        raise ValueError("max epochs must be positive")

    protocol = validate_protocol(args.protocol.resolve())
    pairs = load_jsonl(Path(protocol["data"]["manifest"]["path"]))
    folds = folds_from_pairs(pairs, int(protocol["data"]["fold_count"]))
    targets = flatten_pairs(pairs)
    image_audit = verify_images(targets)
    if not image_audit["all_match_manifest"]:
        raise RuntimeError("image integrity audit failed")

    import torch
    from peft import PeftModel
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

    from modeling import freeze
    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(protocol["parents"]["language"], local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(
        protocol["parents"]["vision"], local_files_only=True, use_fast=False
    )
    vision = SiglipVisionModel.from_pretrained(
        protocol["parents"]["vision"], dtype=dtype, local_files_only=True
    ).cuda()
    freeze(vision)
    feature_cache, blank_feature = cache_vision_features(
        targets, VIEWS, protocol, processor, vision, dtype
    )
    del vision, processor
    gc.collect()
    torch.cuda.empty_cache()

    language = AutoModelForCausalLM.from_pretrained(
        protocol["parents"]["language"], dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(
        language,
        protocol["parents"]["language_adapter"],
        is_trainable=False,
        local_files_only=True,
    )
    freeze(language)

    run = {
        "schema_version": "2026-09-14-v1",
        "status": "running",
        "started_at": utc_now(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "device": torch.cuda.get_device_name(),
        "image_integrity_audit": image_audit,
        "pairs": len(pairs),
        "targets": len(targets),
        "formal_test_present": False,
        "model_selection_performed": False,
        "training_scope": "fresh query readout only; language, LoRA parent, vision tower, and bridge core frozen",
        "fold_receipts": [],
    }
    args.output.mkdir(parents=True)
    write_json_atomic(args.output / "run.json", run)
    arms_to_run = [args.smoke_arm] if args.smoke_arm else list(protocol["execution_order"])
    folds_to_run = [args.smoke_fold] if args.smoke_fold is not None else sorted(folds)
    arm_records = {}
    peak_gib = 0.0
    started = time.monotonic()
    for arm_name in arms_to_run:
        stitched = {view: {condition: [] for condition in CONDITIONS} for view in VIEWS}
        for fold in folds_to_run:
            train_pairs = [pair for other, rows in folds.items() if other != fold for pair in rows]
            test_targets = [row for row in targets if row["fold"] == fold]
            train_ids = {pair["pair_id"] for pair in train_pairs}
            test_ids = {row["pair_id"] for row in test_targets}
            if train_ids & test_ids:
                raise RuntimeError("pair leakage across train/test fold")
            torch.cuda.reset_peak_memory_stats()
            bridge, receipt = train_fold(
                protocol,
                arm_name,
                fold,
                train_pairs,
                feature_cache,
                language,
                tokenizer,
                dtype,
                args.output,
                args.max_epochs,
            )
            receipt["test_pairs"] = len(test_ids)
            receipt["test_pair_ids"] = sorted(test_ids)
            fold_records = score_targets(
                protocol, bridge, language, tokenizer, test_targets, feature_cache, blank_feature
            )
            for view in VIEWS:
                for condition in CONDITIONS:
                    stitched[view][condition].extend(fold_records[view][condition])
            peak_gib = max(peak_gib, torch.cuda.max_memory_allocated() / 2**30)
            run["fold_receipts"].append(receipt)
            write_json_atomic(args.output / "run.json", run)
            del bridge
            gc.collect()
            torch.cuda.empty_cache()
            print(f"READOUT_FOLD_COMPLETE arm={arm_name} fold={fold}", flush=True)
        for view in VIEWS:
            for condition in CONDITIONS:
                stitched[view][condition].sort(key=lambda row: row["target_id"])
        arm_records[arm_name] = stitched

    comparisons = {}
    if set(arm_records) == {"query_ce", "grounded_query"}:
        comparisons["grounded_query_minus_query_ce"] = compare_arms(
            arm_records["grounded_query"], arm_records["query_ce"], protocol
        )
        zero_shot = json.loads(Path(protocol["data"]["zero_shot_receipt"]["path"]).read_text())
        parent_records = zero_shot["arms"]["stage_d_parent"]["records"]
        for arm_name in arm_records:
            comparisons[f"{arm_name}_minus_stage_d_parent"] = compare_arms(
                arm_records[arm_name], parent_records, protocol
            )

    run.update(
        {
            "status": "complete_smoke_only" if args.smoke_fold is not None else "complete_crossfit_development_only",
            "completed_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_gib": peak_gib,
            "arms": {
                name: {"metrics": summarize_records(records, protocol), "records": records}
                for name, records in arm_records.items()
            },
            "comparisons": comparisons,
            "claim_boundary": protocol["claim_boundary"],
        }
    )
    write_json_atomic(args.output / "run.json", run)
    compact = {
        "status": run["status"],
        "pairs": run["pairs"],
        "folds_run": folds_to_run,
        "arms": {
            name: {
                view: value["metrics"][view]
                for view in VIEWS
            }
            for name, value in run["arms"].items()
        },
        "comparisons": comparisons,
        "wall_seconds": run["wall_seconds"],
        "peak_allocated_gib": peak_gib,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
