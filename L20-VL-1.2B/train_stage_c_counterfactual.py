#!/usr/bin/env python3
"""Train matched 49-token compression arms with frozen teacher targets."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from counterfactual_losses import candidate_distillation_loss, evidence_delta_loss, invariance_delta_loss, masked_sequence_logprob
from modeling import MultimodalBridge, bridge_from_architecture, freeze, load_bridge_parent, vision_features
from train_stage_a_full_token import cosine_lr, encode_prompt_response, release_records, sha256_file, utc_now, vision_records, write_json_atomic


ROOT = Path(__file__).resolve().parent
VARIANTS = ("base", "edited", "invariant")
ALLOWED_ARMS = {"answer_49", "kd_49", "strong_49", "proposed_49", "shuffled_delta_49"}


class FamilyDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass
class FamilyBatchBuilder:
    tokenizer: Any
    processor: Any
    max_tokens: int
    teacher: dict[str, dict[str, Any]] | None
    deduplicate_images: bool = False

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        images = []
        encoded = []
        target_indices = []
        teacher_scores = []
        teacher_correct = []
        pair_correct = []
        invariant_pair_correct = []
        image_indices = []
        image_index_by_key = {}
        for row in rows:
            cached = None if self.teacher is None else self.teacher[row["scene_family_id"]]
            if cached is not None and (
                cached["task"] != row["task"]
                or cached["candidate_answers"] != row["candidate_answers"]
            ):
                raise RuntimeError("teacher cache/manifest semantic mismatch")
            family_correct = []
            for variant in VARIANTS:
                image_path = row[f"{variant}_image_path"]
                image_key = (image_path, row.get(f"{variant}_image_sha256"))
                image_index = image_index_by_key.get(image_key) if self.deduplicate_images else None
                if image_index is None:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB").copy())
                    image_index = len(images) - 1
                    if self.deduplicate_images:
                        image_index_by_key[image_key] = image_index
                image_indices.append(image_index)
                for answer in row["candidate_answers"]:
                    encoded.append(encode_prompt_response(
                        self.tokenizer, row["question"], answer, self.max_tokens
                    ))
                target_indices.append(row["candidate_answers"].index(row[f"{variant}_answer"]))
                if cached is None:
                    teacher_scores.append([0.0, 0.0])
                    teacher_correct.append(False)
                    family_correct.append(False)
                else:
                    teacher_variant = cached["variants"][variant]
                    teacher_scores.append(teacher_variant["candidate_scores"])
                    teacher_correct.append(teacher_variant["correct"])
                    family_correct.append(teacher_variant["correct"])
            pair_correct.append(family_correct[0] and family_correct[1])
            invariant_pair_correct.append(family_correct[0] and family_correct[2])
        width = max(len(ids) for ids, _ in encoded)
        input_ids = torch.full((len(encoded), width), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(encoded), width), -100, dtype=torch.long)
        attention = torch.zeros((len(encoded), width), dtype=torch.long)
        for index, (ids, targets) in enumerate(encoded):
            length = len(ids)
            input_ids[index, :length] = torch.tensor(ids)
            labels[index, :length] = torch.tensor(targets)
            attention[index, :length] = 1
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": pixels.contiguous(),
            "image_feature_indices": torch.tensor(image_indices, dtype=torch.long).view(len(rows), len(VARIANTS)),
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "target_indices": torch.tensor(target_indices, dtype=torch.long).view(len(rows), len(VARIANTS)),
            "teacher_scores": torch.tensor(teacher_scores, dtype=torch.float32).view(len(rows), len(VARIANTS), 2),
            "teacher_correct": torch.tensor(teacher_correct, dtype=torch.bool).view(len(rows), len(VARIANTS)),
            "teacher_pair_correct": torch.tensor(pair_correct, dtype=torch.bool),
            "teacher_invariant_pair_correct": torch.tensor(invariant_pair_correct, dtype=torch.bool),
            "scene_family_ids": [row["scene_family_id"] for row in rows],
        }


def load_teacher_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        family = row.get("scene_family_id")
        if not family or family in cache:
            raise RuntimeError(f"teacher cache line {line_number}: duplicate/missing family")
        if row.get("split") != "train":
            raise RuntimeError("teacher cache must contain train split only")
        cache[family] = row
    return cache


def save_checkpoint(output: Path, bridge: MultimodalBridge, language, optimizer, state: dict) -> None:
    checkpoint = output / f"step-{state['optimizer_step']:06d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()},
        checkpoint / "bridge.safetensors",
    )
    adapter = checkpoint / "language_adapter"
    language.save_pretrained(adapter, safe_serialization=True)
    partial = checkpoint / "optimizer.pt.partial"
    torch.save(optimizer.state_dict(), partial)
    os.replace(partial, checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    write_json_atomic(output / "latest.json", {
        "checkpoint": str(checkpoint),
        "optimizer_step": state["optimizer_step"],
        "bridge_sha256": sha256_file(checkpoint / "bridge.safetensors"),
        "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "state_sha256": sha256_file(checkpoint / "state.json"),
    })


def main() -> None:
    process_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    status = protocol.get("status")
    if status == "authorized_stage_c_matched_compression_matrix":
        if args.arm is None or args.arm not in protocol.get("arms", {}):
            raise SystemExit("an arm declared in the matched matrix is required")
        arm = args.arm
        arm_config = protocol["arms"][arm]
        weights = arm_config["loss_weights"]
        shuffle_pair_links = bool(arm_config.get("shuffle_pair_links", False))
    elif status == "authorized_stage_c_matched_compression_arm":
        arm = protocol.get("arm")
        if args.arm is not None and args.arm != arm:
            raise SystemExit("CLI arm conflicts with standalone protocol")
        weights = protocol["loss_weights"]
        shuffle_pair_links = bool(protocol.get("shuffle_pair_links", False))
    elif status in {
        "authorized_visual_token_scaling_sweep",
        "authorized_stage_d_lr_microsearch",
        "authorized_stage_d_two_arm_five_seed_replication",
    }:
        if args.arm is None or args.arm not in protocol.get("arms", {}):
            raise SystemExit("an arm declared in the protocol is required")
        arm = args.arm
        arm_config = protocol["arms"][arm]
        weights = arm_config["loss_weights"]
        shuffle_pair_links = False
    elif status == "authorized_binding_answer_only_development_v1":
        arm = protocol.get("arm")
        if args.arm is not None and args.arm != arm:
            raise SystemExit("CLI arm conflicts with binding baseline protocol")
        arm_config = {
            "architecture": protocol["architecture"],
            "loss_weights": protocol["loss_weights"],
        }
        weights = arm_config["loss_weights"]
        shuffle_pair_links = False
    else:
        raise SystemExit("Stage-C protocol is not authorized")
    generic_answer_only = status in {
        "authorized_visual_token_scaling_sweep",
        "authorized_stage_d_lr_microsearch",
        "authorized_stage_d_two_arm_five_seed_replication",
        "authorized_binding_answer_only_development_v1",
    }
    if not generic_answer_only and arm not in ALLOWED_ARMS:
        raise SystemExit(f"unsupported Stage-C arm: {arm}")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("test split must remain sealed")
    actual_weights = (
        weights["ordinary_kd"] > 0,
        weights["invariance_delta"] > 0,
        weights["evidence_delta"] > 0,
    )
    if generic_answer_only:
        if actual_weights != (False, False, False):
            raise SystemExit("generic Stage-D/sweep arms must use answer supervision only")
    else:
        expected_weights = {
            "answer_49": (False, False, False),
            "kd_49": (True, False, False),
            "strong_49": (True, True, False),
            "proposed_49": (True, True, True),
            "shuffled_delta_49": (True, True, True),
        }
        if actual_weights != expected_weights[arm]:
            raise SystemExit(f"loss weights do not match arm {arm}: {actual_weights}")
        if (arm == "shuffled_delta_49") != shuffle_pair_links:
            raise SystemExit("shuffle-pair declaration does not match arm")
    if weights["answer_supervision"] <= 0 or any(value < 0 for value in weights.values()):
        raise SystemExit("loss weights must be nonnegative with answer supervision enabled")
    architecture = (
        arm_config["architecture"]
        if generic_answer_only
        else protocol["architecture"]
    )
    if status == "authorized_visual_token_scaling_sweep":
        if int(architecture.get("visual_tokens", 0)) != 196:
            raise SystemExit("visual-token sweep requires 196 input patches")
        if architecture.get("compressor") == "spatial_query":
            output_tokens = int(architecture.get("output_visual_tokens", 0))
            if output_tokens not in protocol["spatial_token_budgets"]:
                raise SystemExit("spatial-query output token budget is not frozen in protocol")
        elif architecture.get("compressor") != "none" or arm != "native_196":
            raise SystemExit("only native_196 may omit the capacity-matched compressor")
    elif status in {"authorized_stage_d_lr_microsearch", "authorized_stage_d_two_arm_five_seed_replication"}:
        if architecture.get("compressor") not in {"none", "spatial_query"}:
            raise SystemExit("Stage-D D1 supports only native full-token and spatial-query arms")
        output_tokens = 196 if architecture.get("compressor") == "none" else int(
            architecture.get("output_visual_tokens", 0)
        )
        if output_tokens not in {196, 49}:
            raise SystemExit("Stage-D D1 output visual tokens must be 196 or 49")
    elif status == "authorized_binding_answer_only_development_v1":
        if architecture.get("compressor") != "spatial_query":
            raise SystemExit("binding development baseline requires the spatial-query compressor")
        if int(architecture.get("visual_tokens", 0)) != 196 or int(architecture.get("output_visual_tokens", 0)) != 49:
            raise SystemExit("binding development baseline must preserve the 196-to-49 path")
    elif architecture.get("compressor") != "spatial_query" or int(architecture.get("target_ratio", 0)) != 4:
        raise SystemExit("matched Stage-C arms require the same spatial-query 196-to-49 compressor")
    preference_scale = float(protocol.get("preference_scale", 1.0))
    if preference_scale <= 0:
        raise SystemExit("preference_scale must be positive")

    data = protocol["data"]
    manifest = Path(data["manifest"])
    if sha256_file(manifest) != data["manifest_sha256"]:
        raise SystemExit("manifest hash mismatch")
    teacher_cache_path = None
    teacher = None
    if "teacher_cache" in protocol:
        teacher_cache_path = Path(protocol["teacher_cache"]["path"])
        if sha256_file(teacher_cache_path) != protocol["teacher_cache"]["sha256"]:
            raise SystemExit("teacher cache hash mismatch")
        cache_receipt_path = Path(protocol["teacher_cache"]["receipt"])
        if sha256_file(cache_receipt_path) != protocol["teacher_cache"]["receipt_sha256"]:
            raise SystemExit("teacher cache receipt hash mismatch")
        cache_receipt = json.loads(cache_receipt_path.read_text())
        if cache_receipt.get("cache_sha256") != protocol["teacher_cache"]["sha256"]:
            raise SystemExit("teacher cache receipt does not bind the declared cache")
        teacher = load_teacher_cache(teacher_cache_path)
    all_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if status == "authorized_binding_answer_only_development_v1":
        evidence = data["evidence"]
        for label in ("generation_receipt", "static_audit", "human_audit"):
            declared = evidence[label]
            if sha256_file(Path(declared["path"])) != declared["sha256"]:
                raise SystemExit(f"binding data evidence hash mismatch: {label}")
        generation_receipt = json.loads(Path(evidence["generation_receipt"]["path"]).read_text())
        static_audit = json.loads(Path(evidence["static_audit"]["path"]).read_text())
        human_audit = json.loads(Path(evidence["human_audit"]["path"]).read_text())
        if generation_receipt.get("manifest_sha256") != data["manifest_sha256"]:
            raise SystemExit("generation receipt does not bind the training manifest")
        if generation_receipt.get("final_test_generated") is not False:
            raise SystemExit("binding development data must not include generated final-test data")
        if static_audit.get("status") != "passed_static_and_render_audit_pending_human_review":
            raise SystemExit("binding static/render audit did not pass")
        if not all(static_audit.get("gates", {}).values()):
            raise SystemExit("binding static/render audit has a failed gate")
        if human_audit.get("status") != "passed_bounded_human_visual_audit_for_development_only":
            raise SystemExit("binding human audit did not admit development use")
        if human_audit.get("formal_confirmation_use") is not False:
            raise SystemExit("binding human audit scope is broader than development")
        if {row["split"] for row in all_rows} != {"train", "mechanism_dev", "selection_dev"}:
            raise SystemExit("binding manifest partition set mismatch")
        if any(row.get("statistical_cluster_id") != row.get("scene_pair_id") for row in all_rows):
            raise SystemExit("binding statistical cluster mismatch")
    training_split = data.get("split", "train")
    rows = [row for row in all_rows if row["split"] == training_split]
    if len(rows) != data["scene_families"]:
        raise RuntimeError("manifest train-family count mismatch")
    if teacher is not None and set(teacher) != {row["scene_family_id"] for row in rows}:
        raise RuntimeError("manifest/teacher family set mismatch")
    optimization = dict(protocol["optimization"])
    seed = optimization["seed"] if args.seed is None else args.seed
    allowed_seeds = protocol.get("allowed_seeds", [optimization["seed"]])
    if seed not in allowed_seeds:
        raise SystemExit(f"seed {seed} is not frozen in the protocol")
    multiplier = float(args.learning_rate_multiplier)
    if multiplier <= 0:
        raise SystemExit("learning-rate multiplier must be positive")
    if status == "authorized_stage_d_lr_microsearch":
        if multiplier not in protocol["allowed_learning_rate_multipliers"]:
            raise SystemExit("learning-rate multiplier is not frozen for the Stage-D screen")
    elif status == "authorized_stage_d_two_arm_five_seed_replication":
        if multiplier != float(arm_config["selected_learning_rate_multiplier"]):
            raise SystemExit("formal Stage-D multiplier must equal the development-selected value")
    elif multiplier != 1.0:
        raise SystemExit("learning-rate multiplier override is not authorized by this protocol")
    for name in ("adapter_learning_rate", "bridge_learning_rate", "compressor_learning_rate"):
        if name in optimization:
            optimization[name] = float(optimization[name]) * multiplier
    rng = random.Random(seed)
    if data.get("batch_group_by") == "scene_pair_id":
        grouped = {}
        for row in rows:
            grouped.setdefault(row["scene_pair_id"], []).append(row)
        expected_questions = int(data["questions_per_scene_pair"])
        if any(len(family) != expected_questions for family in grouped.values()):
            raise RuntimeError("scene-pair question count mismatch")
        group_ids = sorted(grouped)
        rng.shuffle(group_ids)
        rows = [
            row
            for group_id in group_ids
            for row in sorted(grouped[group_id], key=lambda item: item["question_index"])
        ]
    else:
        rows.sort(key=lambda row: row["scene_family_id"])
        rng.shuffle(rows)
    max_steps = min(args.max_optimizer_steps or optimization["max_optimizer_steps"], optimization["max_optimizer_steps"])
    max_wall = min(args.max_wall_seconds or optimization["max_wall_seconds"], optimization["max_wall_seconds"])
    if max_steps < 1 or max_wall < 1:
        raise SystemExit("invalid bounded run")

    parents = protocol["parents"]
    base = Path(parents["language"])
    vision_path = Path(parents["vision"])
    parent_bridge = Path(parents["bridge"])
    parent_adapter = Path(parents["language_adapter"])
    if sha256_file(parent_bridge) != parents["bridge_sha256"]:
        raise SystemExit("parent bridge hash mismatch")
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(parent_adapter / name) != expected:
            raise SystemExit(f"parent adapter hash mismatch: {name}")
    parent_before = {"language": release_records(base), "vision": vision_records(vision_path)}

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(
        vision_path, local_files_only=True, use_fast=False
    )
    loader = DataLoader(
        FamilyDataset(rows),
        batch_size=optimization["micro_batch_families"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=FamilyBatchBuilder(
            tokenizer,
            processor,
            optimization["text_tokens_max"],
            teacher,
            deduplicate_images=bool(data.get("deduplicate_images_within_batch", False)),
        ),
        drop_last=True,
    )
    language = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language.config.use_cache = False
    from peft import PeftModel

    language = PeftModel.from_pretrained(
        language, parent_adapter, is_trainable=True, local_files_only=True
    )
    language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    language.train()
    vision = SiglipVisionModel.from_pretrained(
        vision_path, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    bridge = bridge_from_architecture(architecture).to(device="cuda", dtype=torch.bfloat16)
    missing_parent_keys = load_bridge_parent(
        bridge, load_file(parent_bridge, device="cpu"), allow_new_compressor=True
    )
    bridge.train()
    lora_parameters = [parameter for name, parameter in language.named_parameters() if parameter.requires_grad and "lora_" in name]
    unexpected_language = [name for name, parameter in language.named_parameters() if parameter.requires_grad and "lora_" not in name]
    if not lora_parameters or unexpected_language:
        raise RuntimeError(f"unexpected trainable language parameters: {unexpected_language}")
    compressor_parameters = [] if bridge.compressor is None else [
        parameter for parameter in bridge.compressor.parameters() if parameter.requires_grad
    ]
    compressor_ids = {id(parameter) for parameter in compressor_parameters}
    bridge_parameters = [parameter for parameter in bridge.parameters() if parameter.requires_grad]
    non_compressor = [parameter for parameter in bridge_parameters if id(parameter) not in compressor_ids]
    optimizer_groups = [
        ("adapter", lora_parameters, optimization["adapter_learning_rate"]),
        ("bridge", non_compressor, optimization["bridge_learning_rate"]),
    ]
    if compressor_parameters:
        optimizer_groups.append(
            ("compressor", compressor_parameters, optimization["compressor_learning_rate"])
        )
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "lr": learning_rate} for _, parameters, learning_rate in optimizer_groups],
        weight_decay=optimization["weight_decay"], fused=True,
    )
    accumulation = optimization["gradient_accumulation_steps"]
    expected_steps = len(rows) // (optimization["micro_batch_families"] * accumulation)
    if expected_steps != optimization["max_optimizer_steps"]:
        raise RuntimeError(f"one-epoch optimizer-step mismatch: {expected_steps}")
    warmup = max(1, round(max_steps * optimization["warmup_ratio"]))
    temperature = float(protocol["distillation_temperature"])
    beta = float(protocol["huber_beta"])

    args.output.mkdir(parents=True)
    log_path = args.output / "train.jsonl"
    run = {
        "schema_version": "2026-09-13-v1",
        "status": "running",
        "started_at": utc_now(),
        "arm": arm,
        "shuffle_pair_links": shuffle_pair_links,
        "device": torch.cuda.get_device_name(),
        "protocol_sha256": sha256_file(args.protocol),
        "runner_sha256": sha256_file(Path(__file__)),
        "teacher_cache_sha256": None if teacher_cache_path is None else sha256_file(teacher_cache_path),
        "manifest_sha256": sha256_file(manifest),
        "parent_bridge_sha256": sha256_file(parent_bridge),
        "loss_weights": weights,
        "seed": seed,
        "learning_rate_multiplier": multiplier,
        "effective_learning_rates": {
            name: optimization[name]
            for name in ("adapter_learning_rate", "bridge_learning_rate", "compressor_learning_rate")
            if name in optimization
        },
        "input_visual_tokens": bridge.spec.input_tokens,
        "output_visual_tokens": bridge.spec.output_tokens,
        "achieved_compression_ratio": bridge.spec.achieved_ratio,
        "missing_bridge_parent_keys": missing_parent_keys,
        "families": len(rows),
        "image_examples": len(rows) * len(VARIANTS),
        "unique_scene_pairs": len({row.get("scene_pair_id", row["scene_family_id"]) for row in rows}),
        "deduplicate_images_within_batch": bool(data.get("deduplicate_images_within_batch", False)),
        "teacher_pair_coverage_percent": None if teacher is None else 100.0 * sum(row["teacher_pair_correct"] for row in teacher.values()) / len(teacher),
        "teacher_invariant_pair_coverage_percent": None if teacher is None else 100.0 * sum(row["teacher_invariant_pair_correct"] for row in teacher.values()) / len(teacher),
        "parent_before": parent_before,
        "max_optimizer_steps": max_steps,
        "max_wall_seconds": max_wall,
        "optimizer_step": 0,
        "micro_step": 0,
        "prediction_tokens": 0,
    }
    write_json_atomic(args.output / "run.json", run)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    totals = {name: 0.0 for name in ("total", "supervised", "kd", "invariance", "delta", "accuracy")}
    optimizer_step = 0
    prediction_tokens = 0
    total_tokens = 0
    stop_reason = "data_exhausted"
    try:
        for micro_step, batch in enumerate(loader, 1):
            if time.monotonic() - started >= max_wall:
                stop_reason = "wall_time_cap"
                break
            pixels = batch["pixel_values"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention = batch["attention_mask"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            target_indices = batch["target_indices"].to("cuda", non_blocking=True)
            teacher_scores = batch["teacher_scores"].to("cuda", non_blocking=True)
            with torch.no_grad():
                features = vision_features(vision, pixels, int(architecture.get("vision_feature_layer", -1)))
                text = language.get_input_embeddings()(input_ids)
            feature_indices = batch["image_feature_indices"].to("cuda", non_blocking=True)
            features = features[feature_indices.reshape(-1)].repeat_interleave(2, dim=0)
            inputs, mask, targets = bridge.inject(text.detach(), attention, labels, features)
            logits = language(inputs_embeds=inputs, attention_mask=mask).logits[:, :-1]
            shifted = targets[:, 1:]
            token_mask = shifted != -100
            scores = masked_sequence_logprob(
                logits, shifted.masked_fill(~token_mask, 0), token_mask
            ).view(target_indices.shape[0], len(VARIANTS), 2)
            supervised = F.cross_entropy(scores.reshape(-1, 2), target_indices.reshape(-1))
            kd = candidate_distillation_loss(scores, teacher_scores, temperature)
            student_preference = scores[..., 0] - scores[..., 1]
            teacher_preference = teacher_scores[..., 0] - teacher_scores[..., 1]
            student_preference = student_preference / preference_scale
            teacher_preference = teacher_preference / preference_scale
            invariant = invariance_delta_loss(
                student_preference[:, 0], student_preference[:, 2],
                teacher_preference[:, 0], teacher_preference[:, 2],
                batch["teacher_invariant_pair_correct"].to("cuda"), beta,
            )
            if shuffle_pair_links:
                permutation = torch.roll(torch.arange(scores.shape[0], device="cuda"), shifts=1)
                delta = evidence_delta_loss(
                    student_preference[:, 0], student_preference[permutation, 1],
                    teacher_preference[:, 0], teacher_preference[permutation, 1],
                    batch["teacher_correct"][:, 0].to("cuda") & batch["teacher_correct"][:, 1].to("cuda")[permutation], beta,
                )
            else:
                delta = evidence_delta_loss(
                    student_preference[:, 0], student_preference[:, 1],
                    teacher_preference[:, 0], teacher_preference[:, 1],
                    batch["teacher_pair_correct"].to("cuda"), beta,
                )
            loss = (
                weights["answer_supervision"] * supervised
                + weights["ordinary_kd"] * kd
                + weights["invariance_delta"] * invariant
                + weights["evidence_delta"] * delta
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at micro step {micro_step}")
            (loss / accumulation).backward()
            accuracy = (scores.argmax(dim=-1) == target_indices).float().mean()
            for name, value in (
                ("total", loss), ("supervised", supervised), ("kd", kd),
                ("invariance", invariant), ("delta", delta), ("accuracy", accuracy),
            ):
                totals[name] += float(value.detach())
            prediction_tokens += int((labels != -100).sum())
            total_tokens += int(attention.sum()) + labels.shape[0] * (bridge.spec.output_tokens + 2)
            run["micro_step"] = micro_step
            if micro_step % accumulation:
                continue
            optimizer_step += 1
            trainable = bridge_parameters + lora_parameters
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, optimization["gradient_clip_norm"])
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at step {optimizer_step}")
            scale = cosine_lr(optimizer_step, max_steps, warmup, optimization["minimum_learning_rate_ratio"])
            learning_rates = {
                name: base_learning_rate * scale
                for name, _, base_learning_rate in optimizer_groups
            }
            for group, (name, _, _) in zip(optimizer.param_groups, optimizer_groups):
                learning_rate = learning_rates[name]
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            event = {
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                **{f"{name}_loss_mean" if name != "accuracy" else "candidate_accuracy_mean": value / accumulation for name, value in totals.items()},
                "gradient_norm": float(grad_norm),
                "adapter_learning_rate": learning_rates["adapter"],
                "bridge_learning_rate": learning_rates["bridge"],
                "compressor_learning_rate": learning_rates.get("compressor", 0.0),
                "prediction_tokens": prediction_tokens,
                "prediction_tokens_per_second": prediction_tokens / elapsed,
                "total_input_and_visual_tokens_per_second": total_tokens / elapsed,
                "elapsed_seconds": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "timestamp": utc_now(),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            totals = {name: 0.0 for name in totals}
            run.update(event)
            write_json_atomic(args.output / "run.json", run)
            if optimizer_step % optimization["checkpoint_every_steps"] == 0:
                save_checkpoint(args.output, bridge, language, optimizer, run)
            if optimizer_step >= max_steps:
                stop_reason = "optimizer_step_cap"
                break
        run["status"] = "complete"
        run["stop_reason"] = stop_reason
        run["completed_at"] = utc_now()
        run["wall_seconds"] = time.monotonic() - started
        run["total_wall_seconds"] = time.monotonic() - process_started
        if optimizer_step and optimizer_step % optimization["checkpoint_every_steps"]:
            save_checkpoint(args.output, bridge, language, optimizer, run)
        parent_after = {"language": release_records(base), "vision": vision_records(vision_path)}
        if parent_after != parent_before:
            raise RuntimeError("immutable parent files changed")
        if sha256_file(parent_bridge) != parents["bridge_sha256"]:
            raise RuntimeError("parent bridge changed")
        for name, expected in parents["language_adapter_files"].items():
            if sha256_file(parent_adapter / name) != expected:
                raise RuntimeError(f"parent adapter changed: {name}")
        run["parent_after"] = parent_after
        write_json_atomic(args.output / "run.json", run)
    except Exception as error:
        run["status"] = "failed"
        run["failed_at"] = utc_now()
        run["error"] = f"{type(error).__name__}: {error}"
        run["wall_seconds"] = time.monotonic() - started
        run["total_wall_seconds"] = time.monotonic() - process_started
        write_json_atomic(args.output / "run.json", run)
        raise


if __name__ == "__main__":
    main()
