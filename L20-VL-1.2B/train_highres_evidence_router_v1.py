#!/usr/bin/env python3
"""Train a small POINT-or-STOP router from frozen responder outcomes."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from evaluate_highres_evidence_responder_floor_v1 import answer_scores
from evidence_acquisition import EvidenceAcquisitionRouter, build_evidence_targets
from modeling import bridge_from_architecture, freeze, vision_features
from train_highres_evidence_crop_bridge_v1 import crop_image, inject_global_and_crop
from train_stage_a_full_token import cosine_lr, encode_prompt_response, release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_highres_evidence_router_train_only_v1"


def load_train_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == "train"]
    selected.sort(key=lambda row: (row["family_id"], row["variant"]))
    if any(row["split"] != "train" for row in selected):
        raise RuntimeError("non-train row reached router trainer")
    return selected


def balanced_epoch_batches(rows, batch_rows: int, seed: int, epoch: int):
    if batch_rows % 2:
        raise ValueError("balanced batch size must be even")
    local = [index for index, row in enumerate(rows) if row["expected_action"] == "POINT"]
    stop = [index for index, row in enumerate(rows) if row["expected_action"] == "STOP"]
    if len(local) != len(stop) or len(local) % (batch_rows // 2):
        raise RuntimeError("POINT/STOP rows cannot form exact balanced batches")
    rng = random.Random(seed + epoch)
    rng.shuffle(local)
    rng.shuffle(stop)
    half = batch_rows // 2
    batches = []
    for start in range(0, len(local), half):
        batch = local[start:start + half] + stop[start:start + half]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def collate_true_answers(tokenizer, rows, max_tokens: int):
    encoded = [encode_prompt_response(tokenizer, row["question"], row["answer"], max_tokens) for row in rows]
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for index, (ids, targets) in enumerate(encoded):
        input_ids[index, :len(ids)] = torch.tensor(ids)
        attention[index, :len(ids)] = 1
        labels[index, :len(ids)] = torch.tensor(targets)
    return input_ids, attention, labels


def cache_vision(rows, protocol, processor, vision, dtype):
    root = Path(protocol["data"]["root"])
    batch_size = int(protocol["teacher"]["feature_batch_images"])
    global_cache, crop_cache = {}, {}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            globals_, local_rows, crops = [], [], []
            for row in batch:
                with Image.open(root / row["image_path"]) as source:
                    image = source.convert("RGB")
                globals_.append(image)
                if row["expected_action"] == "POINT":
                    local_rows.append(row)
                    crops.append(crop_image(image, row["target_crop_box_xyxy_normalized"]))
            pixels = processor(images=globals_ + crops, return_tensors="pt")["pixel_values"].to(
                "cuda", dtype=dtype
            )
            features = vision_features(
                vision, pixels, int(protocol["architecture"]["vision_feature_layer"])
            ).to("cpu", dtype=dtype)
            for row, feature in zip(batch, features[:len(batch)]):
                global_cache[(row["family_id"], row["variant"])] = feature.contiguous()
            for row, feature in zip(local_rows, features[len(batch):]):
                crop_cache[(row["family_id"], row["variant"])] = feature.contiguous()
    return global_cache, crop_cache


def responder_gains(local_rows, protocol, tokenizer, language, parent_bridge, crop_bridge, caches):
    global_cache, crop_cache = caches
    gains, global_nlls, crop_nlls = {}, {}, {}
    batch_size = int(protocol["teacher"]["responder_batch_rows"])
    with torch.inference_mode():
        for start in range(0, len(local_rows), batch_size):
            batch = local_rows[start:start + batch_size]
            input_ids, attention, labels = collate_true_answers(
                tokenizer, batch, int(protocol["teacher"]["max_text_tokens"])
            )
            input_ids = input_ids.cuda(non_blocking=True)
            attention = attention.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            text = language.get_input_embeddings()(input_ids)
            global_features = torch.stack([
                global_cache[(row["family_id"], row["variant"])] for row in batch
            ]).to("cuda", non_blocking=True)
            crop_features = torch.stack([
                crop_cache[(row["family_id"], row["variant"])] for row in batch
            ]).to("cuda", non_blocking=True)
            global_inputs, global_mask, global_targets = parent_bridge.inject(
                text, attention, labels, global_features
            )
            crop_inputs, crop_mask, crop_targets = inject_global_and_crop(
                parent_bridge, crop_bridge, text, attention, labels, global_features, crop_features
            )
            global_score = answer_scores(
                language(inputs_embeds=global_inputs, attention_mask=global_mask).logits,
                global_targets,
                tokenizer.eos_token_id,
            )
            crop_score = answer_scores(
                language(inputs_embeds=crop_inputs, attention_mask=crop_mask).logits,
                crop_targets,
                tokenizer.eos_token_id,
            )
            for row, before, after in zip(batch, global_score.tolist(), crop_score.tolist()):
                key = (row["family_id"], row["variant"])
                global_nlls[key] = -float(before)
                crop_nlls[key] = -float(after)
                gains[key] = float(after - before)
    return gains, global_nlls, crop_nlls


def balanced_router_loss(pointer_logits, predicted_gain, pointer_targets, gain_targets, point_mask, gain_weight: float):
    if not point_mask.any() or point_mask.all():
        raise RuntimeError("router batch must contain both POINT and STOP")
    point_ce = F.cross_entropy(pointer_logits[point_mask].float(), pointer_targets[point_mask])
    stop_ce = F.cross_entropy(pointer_logits[~point_mask].float(), pointer_targets[~point_mask])
    pointer_ce = 0.5 * (point_ce + stop_ce)
    gain_huber = F.smooth_l1_loss(predicted_gain.float(), gain_targets.float())
    return pointer_ce + float(gain_weight) * gain_huber, {
        "point_pointer_ce": point_ce.detach(),
        "stop_pointer_ce": stop_ce.detach(),
        "gain_huber": gain_huber.detach(),
    }


def save_checkpoint(output: Path, router, optimizer, state: dict[str, Any]):
    checkpoint = output / f"step-{state['optimizer_step']:06d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in router.state_dict().items()},
        checkpoint / "router.safetensors",
    )
    partial = checkpoint / "optimizer.pt.partial"
    torch.save(optimizer.state_dict(), partial)
    os.replace(partial, checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    write_json_atomic(output / "latest.json", {
        "checkpoint": str(checkpoint),
        "optimizer_step": state["optimizer_step"],
        "router_sha256": sha256_file(checkpoint / "router.safetensors"),
        "optimizer_sha256": sha256_file(checkpoint / "optimizer.pt"),
        "state_sha256": sha256_file(checkpoint / "state.json"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise SystemExit("router training is not authorized")
    if protocol.get("allowed_data_splits") != ["train"]:
        raise SystemExit("router trainer is train-only")
    if protocol.get("trainable_components") != ["evidence_acquisition_router"]:
        raise SystemExit("only the router may be trainable")
    for path, key in (
        (Path(__file__), "trainer_sha256"),
        (ROOT / "test_train_highres_evidence_router_v1.py", "test_sha256"),
        (ROOT / "evidence_acquisition.py", "router_module_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
        (ROOT / "train_highres_evidence_crop_bridge_v1.py", "crop_trainer_sha256"),
        (ROOT / "evaluate_highres_evidence_responder_floor_v1.py", "floor_evaluator_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"prerequisite hash mismatch: {name}")

    parents = protocol["parents"]
    language_root = Path(parents["language"])
    if sha256_file(language_root / "release-manifest.json") != parents["language_release_manifest_sha256"]:
        raise SystemExit("language release manifest mismatch")
    language_before = release_records(language_root)
    vision_root = Path(parents["vision"])
    for name, expected in parents["vision_artifacts"].items():
        if sha256_file(vision_root / name) != expected:
            raise SystemExit(f"vision artifact mismatch: {name}")
    parent_bridge_path = Path(parents["bridge"])
    crop_bridge_path = Path(parents["crop_bridge"])
    if sha256_file(parent_bridge_path) != parents["bridge_sha256"]:
        raise SystemExit("parent bridge mismatch")
    if sha256_file(crop_bridge_path) != parents["crop_bridge_sha256"]:
        raise SystemExit("crop bridge mismatch")
    adapter_root = Path(parents["language_adapter"])
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(adapter_root / name) != expected:
            raise SystemExit(f"language adapter mismatch: {name}")

    rows = load_train_rows(Path(protocol["data"]["manifest"]))
    if len(rows) != int(protocol["data"]["train_rows"]):
        raise RuntimeError("unexpected train row count")
    counts = {action: sum(row["expected_action"] == action for row in rows) for action in ("POINT", "STOP")}
    if counts != protocol["data"]["action_counts"]:
        raise RuntimeError(f"unexpected action counts: {counts}")
    optimization = protocol["optimization"]
    batches_per_epoch = len(rows) // int(optimization["batch_rows"])
    if batches_per_epoch * int(optimization["epochs"]) != int(optimization["max_optimizer_steps"]):
        raise RuntimeError("router optimizer-step arithmetic mismatch")
    max_steps = min(args.max_optimizer_steps or int(optimization["max_optimizer_steps"]), int(optimization["max_optimizer_steps"]))
    max_wall = min(args.max_wall_seconds or int(optimization["max_wall_seconds"]), int(optimization["max_wall_seconds"]))

    seed = int(optimization["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(language_root, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_root, local_files_only=True, use_fast=False)
    vision = SiglipVisionModel.from_pretrained(vision_root, dtype=dtype, local_files_only=True).cuda()
    freeze(vision)
    caches = cache_vision(rows, protocol, processor, vision, dtype)
    del vision
    torch.cuda.empty_cache()

    from peft import PeftModel

    language = AutoModelForCausalLM.from_pretrained(
        language_root, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(language, adapter_root, is_trainable=False, local_files_only=True)
    freeze(language)
    parent_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    parent_bridge.load_state_dict(load_file(parent_bridge_path, device="cpu"), strict=True)
    freeze(parent_bridge)
    crop_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    crop_bridge.load_state_dict(load_file(crop_bridge_path, device="cpu"), strict=True)
    freeze(crop_bridge)
    local_rows = [row for row in rows if row["expected_action"] == "POINT"]
    gains, global_nlls, crop_nlls = responder_gains(
        local_rows, protocol, tokenizer, language, parent_bridge, crop_bridge, caches
    )
    input_ids, attention, labels = collate_true_answers(
        tokenizer, rows, int(protocol["teacher"]["max_text_tokens"])
    )
    with torch.inference_mode():
        text_embeddings = language.get_input_embeddings()(input_ids.cuda()).to("cpu", dtype=dtype)
    global_features = torch.stack([
        caches[0][(row["family_id"], row["variant"])] for row in rows
    ]).contiguous()
    del language, parent_bridge, crop_bridge, caches
    torch.cuda.empty_cache()

    raw_gain = torch.tensor([
        gains.get((row["family_id"], row["variant"]), 0.0) for row in rows
    ], dtype=torch.float32)
    best_patch = torch.tensor([
        int(row["target_patch_index_14x14"] or 0) for row in rows
    ], dtype=torch.long)
    targets = build_evidence_targets(
        torch.zeros_like(raw_gain), -raw_gain, best_patch,
        int(protocol["router"]["patch_count"]), float(protocol["teacher"]["minimum_observed_gain"]),
    )
    expected_point = torch.tensor([row["expected_action"] == "POINT" for row in rows])
    positive_fraction = float(targets.zoom_is_beneficial[expected_point].float().mean())
    if positive_fraction < float(protocol["teacher"]["minimum_local_positive_gain_fraction"]):
        raise RuntimeError(f"insufficient positive responder-gain coverage: {positive_fraction}")
    # STOP is structural for global-answer rows: they require no additional view.
    pointer_targets = torch.where(
        expected_point, best_patch, torch.full_like(best_patch, int(protocol["router"]["patch_count"]))
    )
    gain_targets = raw_gain.clamp(
        min=float(protocol["teacher"]["gain_clip_min"]),
        max=float(protocol["teacher"]["gain_clip_max"]),
    )

    args.output.mkdir(parents=True)
    supervision_path = args.output / "teacher_targets.jsonl"
    with supervision_path.open("x") as handle:
        for row, pointer, gain in zip(rows, pointer_targets.tolist(), raw_gain.tolist()):
            key = (row["family_id"], row["variant"])
            handle.write(json.dumps({
                "family_id": row["family_id"],
                "variant": row["variant"],
                "expected_action": row["expected_action"],
                "pointer_target": pointer,
                "raw_observed_correct_answer_nll_gain": gain,
                "global_correct_answer_nll": global_nlls.get(key),
                "crop_correct_answer_nll": crop_nlls.get(key),
            }, sort_keys=True) + "\n")

    router = EvidenceAcquisitionRouter(
        vision_dim=int(protocol["router"]["vision_dim"]),
        language_dim=int(protocol["router"]["language_dim"]),
        rank=int(protocol["router"]["rank"]),
    ).to("cuda", dtype=dtype)
    router.train()
    trainable = [parameter for parameter in router.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]), fused=True,
    )
    warmup = max(1, round(max_steps * float(optimization["warmup_ratio"])))
    run = {
        "schema_version": "2026-09-14-v1",
        "status": "running",
        "started_at": utc_now(),
        "device": torch.cuda.get_device_name(),
        "protocol_sha256": sha256_file(args.protocol),
        "runner_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": sha256_file(Path(protocol["data"]["manifest"])),
        "teacher_targets": {"path": str(supervision_path), "sha256": sha256_file(supervision_path)},
        "parent_before": {"language": language_before, "vision_artifacts": parents["vision_artifacts"]},
        "trainable_components": ["evidence_acquisition_router"],
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "action_counts": counts,
        "local_positive_gain_fraction": positive_fraction,
        "local_gain_summary": {
            "minimum": min(gains.values()),
            "mean": sum(gains.values()) / len(gains),
            "maximum": max(gains.values()),
        },
        "max_optimizer_steps": max_steps,
        "optimizer_step": 0,
        "development_rows_accessed": 0,
        "sealed_test_rows_accessed": 0,
    }
    write_json_atomic(args.output / "run.json", run)
    started = time.monotonic()
    optimizer_step = 0
    examples_seen = 0
    try:
        for epoch in range(int(optimization["epochs"])):
            for indices in balanced_epoch_batches(rows, int(optimization["batch_rows"]), seed, epoch):
                if time.monotonic() - started >= max_wall:
                    raise TimeoutError("router wall-time cap reached before optimizer-step cap")
                index = torch.tensor(indices, dtype=torch.long)
                vision_batch = global_features[index].to("cuda", non_blocking=True)
                text_batch = text_embeddings[index].to("cuda", non_blocking=True)
                attention_batch = attention[index].to("cuda", non_blocking=True)
                labels_batch = labels[index].to("cuda", non_blocking=True)
                pointer_batch = pointer_targets[index].to("cuda", non_blocking=True)
                gain_batch = gain_targets[index].to("cuda", non_blocking=True)
                point_mask = expected_point[index].to("cuda", non_blocking=True)
                pointer_logits, predicted_gain = router(
                    vision_batch, text_batch, attention_batch, labels_batch
                )
                loss, parts = balanced_router_loss(
                    pointer_logits, predicted_gain, pointer_batch, gain_batch, point_mask,
                    float(optimization["gain_loss_weight"]),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_step += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(optimization["gradient_clip_norm"]))
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"non-finite gradient at step {optimizer_step}")
                scale = cosine_lr(
                    optimizer_step, max_steps, warmup, float(optimization["minimum_learning_rate_ratio"])
                )
                learning_rate = float(optimization["learning_rate"]) * scale
                optimizer.param_groups[0]["lr"] = learning_rate
                optimizer.step()
                examples_seen += len(indices)
                elapsed = time.monotonic() - started
                predicted_action = pointer_logits.argmax(dim=-1)
                event = {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    **{name: float(value) for name, value in parts.items()},
                    "action_accuracy": float((predicted_action.eq(pointer_batch).where(~point_mask, predicted_action.ne(int(protocol["router"]["patch_count"])))).float().mean()),
                    "exact_pointer_accuracy": float((predicted_action[point_mask] == pointer_batch[point_mask]).float().mean()),
                    "stop_accuracy": float((predicted_action[~point_mask] == pointer_batch[~point_mask]).float().mean()),
                    "gain_mae": float((predicted_gain - gain_batch).abs().mean().detach()),
                    "gradient_norm": float(grad_norm),
                    "learning_rate": learning_rate,
                    "examples_seen": examples_seen,
                    "examples_per_second": examples_seen / elapsed,
                    "elapsed_seconds": elapsed,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "timestamp": utc_now(),
                }
                with (args.output / "train.jsonl").open("a") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event, sort_keys=True), flush=True)
                run.update(event)
                write_json_atomic(args.output / "run.json", run)
                if optimizer_step % int(optimization["checkpoint_every_steps"]) == 0:
                    save_checkpoint(args.output, router, optimizer, run)
                if optimizer_step >= max_steps:
                    break
            if optimizer_step >= max_steps:
                break
        run["status"] = "complete"
        run["stop_reason"] = "optimizer_step_cap"
        run["completed_at"] = utc_now()
        run["wall_seconds"] = time.monotonic() - started
        if optimizer_step % int(optimization["checkpoint_every_steps"]):
            save_checkpoint(args.output, router, optimizer, run)
        if release_records(language_root) != language_before:
            raise RuntimeError("language parent changed")
        if sha256_file(parent_bridge_path) != parents["bridge_sha256"]:
            raise RuntimeError("parent bridge changed")
        if sha256_file(crop_bridge_path) != parents["crop_bridge_sha256"]:
            raise RuntimeError("crop bridge changed")
        for name, expected_hash in parents["vision_artifacts"].items():
            if sha256_file(vision_root / name) != expected_hash:
                raise RuntimeError(f"vision parent changed: {name}")
        for name, expected_hash in parents["language_adapter_files"].items():
            if sha256_file(adapter_root / name) != expected_hash:
                raise RuntimeError(f"language adapter parent changed: {name}")
        run["parent_after_verified_unchanged"] = True
        write_json_atomic(args.output / "run.json", run)
    except Exception as error:
        run["status"] = "failed"
        run["failed_at"] = utc_now()
        run["error"] = f"{type(error).__name__}: {error}"
        run["wall_seconds"] = time.monotonic() - started
        write_json_atomic(args.output / "run.json", run)
        raise


if __name__ == "__main__":
    main()
