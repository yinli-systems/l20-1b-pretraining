#!/usr/bin/env python3
"""Adapt the confirmed synthetic evidence router to real red-boxed images with replay."""
from __future__ import annotations

import argparse
import hashlib
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
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from evidence_acquisition import EvidenceAcquisitionRouter
from modeling import freeze, vision_features
from train_highres_evidence_router_v1 import collate_true_answers, save_checkpoint
from train_stage_a_full_token import cosine_lr, release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_openimages_evidence_router_adaptation_train_only_v1"


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def load_split(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == split]
    selected.sort(key=lambda row: (row["family_id"], row["variant"]))
    return selected


def select_replay(rows: list[dict[str, Any]], families: int, seed: int) -> list[dict[str, Any]]:
    if families % 2:
        raise ValueError("replay family count must be even")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["family_id"], []).append(row)
    by_action = {"POINT": [], "STOP": []}
    for family_id, group in grouped.items():
        actions = {row["expected_action"] for row in group}
        if len(actions) != 1:
            raise RuntimeError(f"synthetic replay family mixes actions: {family_id}")
        by_action[actions.pop()].append(family_id)
    half = families // 2
    allowed = set()
    for action, family_ids in by_action.items():
        chosen = sorted(family_ids, key=lambda value: stable_key(seed, f"{action}|{value}"))[:half]
        if len(chosen) != half:
            raise RuntimeError(f"insufficient {action} replay families")
        allowed.update(chosen)
    selected = [row for row in rows if row["family_id"] in allowed]
    selected.sort(key=lambda row: (row["family_id"], row["variant"]))
    return selected


def balanced_batches(rows: list[dict[str, Any]], batch_rows: int, seed: int, epoch: int):
    if batch_rows % 2:
        raise ValueError("balanced batch size must be even")
    point = [index for index, row in enumerate(rows) if row["expected_action"] == "POINT"]
    stop = [index for index, row in enumerate(rows) if row["expected_action"] == "STOP"]
    half = batch_rows // 2
    if len(point) != len(stop) or len(point) % half:
        raise RuntimeError("rows cannot form exact balanced batches")
    rng = random.Random(seed + epoch)
    rng.shuffle(point)
    rng.shuffle(stop)
    batches = []
    for start in range(0, len(point), half):
        batch = point[start:start + half] + stop[start:start + half]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def pointer_targets(rows: list[dict[str, Any]], patch_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    point = torch.tensor([row["expected_action"] == "POINT" for row in rows], dtype=torch.bool)
    targets = torch.tensor([
        int(row["target_patch_index_14x14"]) if row["expected_action"] == "POINT" else patch_count
        for row in rows
    ], dtype=torch.long)
    return targets, point


def prompt_only_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the real prompt while using a fixed short masked response placeholder."""
    return [{**row, "answer": "ok"} for row in rows]


def balanced_pointer_ce(logits: torch.Tensor, targets: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    if not point.any() or point.all():
        raise RuntimeError("batch must contain POINT and STOP")
    return 0.5 * (
        F.cross_entropy(logits[point].float(), targets[point])
        + F.cross_entropy(logits[~point].float(), targets[~point])
    )


def resolve_image(root: Path, row: dict[str, Any]) -> Path:
    path = Path(row["image_path"])
    return path if path.is_absolute() else root / path


def cache_vision_rows(rows, root: Path, protocol, processor, vision, dtype):
    batch_size = int(protocol["features"]["batch_images"])
    cached = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = []
            for row in batch:
                path = resolve_image(root, row)
                if sha256_file(path) != row["image_sha256"]:
                    raise RuntimeError(f"image hash mismatch: {path}")
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda", dtype=dtype)
            features = vision_features(
                vision, pixels, int(protocol["architecture"]["vision_feature_layer"])
            ).to("cpu", dtype=dtype)
            cached.extend(feature.contiguous() for feature in features)
    return torch.stack(cached).contiguous()


def subset_families(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    families = sorted({row["family_id"] for row in rows})[:count]
    allowed = set(families)
    return [row for row in rows if row["family_id"] in allowed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--smoke-families", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise SystemExit("natural router adaptation is not authorized")
    if protocol.get("allowed_data_splits") != {"natural": ["train"], "synthetic_replay": ["train"]}:
        raise SystemExit("trainer must remain train-only")
    if protocol.get("trainable_components") != ["evidence_acquisition_router"]:
        raise SystemExit("only the router may be trainable")
    for path, key in (
        (Path(__file__), "trainer_sha256"),
        (ROOT / "test_train_openimages_evidence_router_adaptation_v1.py", "test_sha256"),
        (ROOT / "evidence_acquisition.py", "router_module_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
        (ROOT / "train_highres_evidence_router_v1.py", "synthetic_trainer_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"prerequisite hash mismatch: {name}")

    natural_manifest = Path(protocol["data"]["natural_manifest"])
    replay_manifest = Path(protocol["data"]["synthetic_replay_manifest"])
    natural_rows = load_split(natural_manifest, "train")
    replay_rows = select_replay(
        load_split(replay_manifest, "train"),
        int(protocol["data"]["synthetic_replay_families"]),
        int(protocol["data"]["synthetic_replay_seed"]),
    )
    if len(natural_rows) != int(protocol["data"]["natural_train_rows"]):
        raise RuntimeError("unexpected natural train rows")
    if len(replay_rows) != int(protocol["data"]["synthetic_replay_rows"]):
        raise RuntimeError("unexpected replay rows")
    smoke = args.smoke_families is not None
    if smoke:
        if args.smoke_families < 2:
            raise ValueError("smoke-families must be at least two")
        natural_rows = subset_families(natural_rows, args.smoke_families)
        replay_rows = select_replay(
            replay_rows, min(args.smoke_families, len({row["family_id"] for row in replay_rows})),
            int(protocol["data"]["synthetic_replay_seed"]) + 1,
        )

    optimization = protocol["optimization"]
    natural_batch = int(optimization["natural_batch_rows"])
    replay_batch = int(optimization["replay_batch_rows"])
    if smoke:
        natural_batch = min(natural_batch, len(natural_rows))
        replay_batch = min(replay_batch, len(replay_rows))
        natural_batch -= natural_batch % 2
        replay_batch -= replay_batch % 2
    batches_per_epoch = len(natural_rows) // natural_batch
    full_steps = batches_per_epoch * int(optimization["epochs"])
    expected_full = int(optimization["max_optimizer_steps"])
    if not smoke and full_steps != expected_full:
        raise RuntimeError("optimizer-step arithmetic mismatch")
    max_steps = min(args.max_optimizer_steps or full_steps, full_steps, expected_full)
    max_wall_seconds = int(optimization["max_wall_seconds"])

    seed = int(optimization["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16
    parents = protocol["parents"]
    language_root = Path(parents["language"])
    if sha256_file(language_root / "release-manifest.json") != parents["language_release_manifest_sha256"]:
        raise SystemExit("language parent mismatch")
    language_before = release_records(language_root)
    vision_root = Path(parents["vision"])
    for name, expected in parents["vision_artifacts"].items():
        if sha256_file(vision_root / name) != expected:
            raise SystemExit(f"vision parent mismatch: {name}")
    router_parent = Path(parents["router"])
    if sha256_file(router_parent) != parents["router_sha256"]:
        raise SystemExit("router parent mismatch")
    adapter_root = Path(parents["language_adapter"])
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(adapter_root / name) != expected:
            raise SystemExit(f"language adapter mismatch: {name}")

    tokenizer = AutoTokenizer.from_pretrained(language_root, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_root, local_files_only=True, use_fast=False)
    vision = SiglipVisionModel.from_pretrained(vision_root, dtype=dtype, local_files_only=True).cuda()
    freeze(vision)
    natural_features = cache_vision_rows(
        natural_rows, Path(protocol["data"]["natural_root"]), protocol, processor, vision, dtype
    )
    replay_features = cache_vision_rows(
        replay_rows, Path(protocol["data"]["synthetic_replay_root"]), protocol, processor, vision, dtype
    )
    del vision
    torch.cuda.empty_cache()

    from peft import PeftModel

    language = AutoModelForCausalLM.from_pretrained(
        language_root, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(language, adapter_root, is_trainable=False, local_files_only=True)
    freeze(language)
    if protocol["features"].get("prompt_tokens_only") is not True:
        raise SystemExit("router adaptation requires prompt-only text features")
    natural_ids, natural_attention, natural_labels = collate_true_answers(
        tokenizer, prompt_only_rows(natural_rows), int(protocol["features"]["max_text_tokens"])
    )
    replay_ids, replay_attention, replay_labels = collate_true_answers(
        tokenizer, prompt_only_rows(replay_rows), int(protocol["features"]["max_text_tokens"])
    )
    with torch.inference_mode():
        natural_text = language.get_input_embeddings()(natural_ids.cuda()).to("cpu", dtype=dtype)
        replay_text = language.get_input_embeddings()(replay_ids.cuda()).to("cpu", dtype=dtype)
    del language
    torch.cuda.empty_cache()

    patch_count = int(protocol["router"]["patch_count"])
    natural_targets, natural_point = pointer_targets(natural_rows, patch_count)
    replay_targets, replay_point = pointer_targets(replay_rows, patch_count)
    parent = EvidenceAcquisitionRouter(
        int(protocol["router"]["vision_dim"]), int(protocol["router"]["language_dim"]),
        int(protocol["router"]["rank"]),
    ).to("cuda", dtype=dtype)
    parent.load_state_dict(load_file(router_parent, device="cpu"), strict=True)
    freeze(parent)
    parent_logits, parent_gain = [], []
    with torch.inference_mode():
        batch = int(protocol["features"]["router_batch_rows"])
        for start in range(0, len(replay_rows), batch):
            stop = min(len(replay_rows), start + batch)
            logits, gain = parent(
                replay_features[start:stop].to("cuda"), replay_text[start:stop].to("cuda"),
                replay_attention[start:stop].to("cuda"), replay_labels[start:stop].to("cuda"),
            )
            parent_logits.append(logits.to("cpu", dtype=torch.float32))
            parent_gain.append(gain.cpu())
    parent_logits = torch.cat(parent_logits)
    parent_gain = torch.cat(parent_gain)

    router = EvidenceAcquisitionRouter(
        int(protocol["router"]["vision_dim"]), int(protocol["router"]["language_dim"]),
        int(protocol["router"]["rank"]),
    ).to("cuda", dtype=dtype)
    router.load_state_dict(load_file(router_parent, device="cpu"), strict=True)
    router.train()
    trainable = [parameter for parameter in router.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]), fused=True,
    )
    warmup = max(1, round(max_steps * float(optimization["warmup_ratio"])))
    replay_batches = balanced_batches(replay_rows, replay_batch, seed + 100000, 0)
    replay_cursor = 0
    args.output.mkdir(parents=True)
    run = {
        "schema_version": "2026-09-14-v1",
        "status": "running",
        "started_at": utc_now(),
        "smoke": smoke,
        "device": torch.cuda.get_device_name(),
        "protocol_sha256": sha256_file(args.protocol),
        "trainer_sha256": sha256_file(Path(__file__)),
        "natural_manifest_sha256": sha256_file(natural_manifest),
        "synthetic_replay_manifest_sha256": sha256_file(replay_manifest),
        "natural_rows": len(natural_rows),
        "replay_rows": len(replay_rows),
        "trainable_components": ["evidence_acquisition_router"],
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "max_optimizer_steps": max_steps,
        "optimizer_step": 0,
        "development_rows_accessed": 0,
        "confirmation_rows_accessed": 0,
    }
    write_json_atomic(args.output / "run.json", run)
    started = time.monotonic()
    optimizer_step = 0
    examples_seen = 0
    try:
        for epoch in range(int(optimization["epochs"])):
            for natural_indices in balanced_batches(natural_rows, natural_batch, seed, epoch):
                if optimizer_step >= max_steps:
                    break
                if time.monotonic() - started >= max_wall_seconds:
                    raise TimeoutError("natural router wall-time cap reached")
                if replay_cursor >= len(replay_batches):
                    replay_batches = balanced_batches(replay_rows, replay_batch, seed + 100000, replay_cursor + epoch + 1)
                    replay_cursor = 0
                replay_indices = replay_batches[replay_cursor]
                replay_cursor += 1
                ni = torch.tensor(natural_indices, dtype=torch.long)
                ri = torch.tensor(replay_indices, dtype=torch.long)
                natural_logits, natural_gain = router(
                    natural_features[ni].to("cuda"), natural_text[ni].to("cuda"),
                    natural_attention[ni].to("cuda"), natural_labels[ni].to("cuda"),
                )
                replay_logits, replay_gain = router(
                    replay_features[ri].to("cuda"), replay_text[ri].to("cuda"),
                    replay_attention[ri].to("cuda"), replay_labels[ri].to("cuda"),
                )
                natural_target = natural_targets[ni].to("cuda")
                natural_mask = natural_point[ni].to("cuda")
                replay_target = replay_targets[ri].to("cuda")
                replay_mask = replay_point[ri].to("cuda")
                natural_ce = balanced_pointer_ce(natural_logits, natural_target, natural_mask)
                replay_ce = balanced_pointer_ce(replay_logits, replay_target, replay_mask)
                teacher_logits = parent_logits[ri].to("cuda")
                distill_kl = F.kl_div(
                    F.log_softmax(replay_logits.float(), dim=-1),
                    F.softmax(teacher_logits, dim=-1), reduction="batchmean",
                )
                gain_huber = F.smooth_l1_loss(replay_gain.float(), parent_gain[ri].to("cuda"))
                loss = (
                    natural_ce
                    + float(optimization["replay_pointer_weight"]) * replay_ce
                    + float(optimization["replay_kl_weight"]) * distill_kl
                    + float(optimization["replay_gain_weight"]) * gain_huber
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(optimization["gradient_clip_norm"]))
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite gradient")
                optimizer_step += 1
                scale = cosine_lr(
                    optimizer_step, max_steps, warmup, float(optimization["minimum_learning_rate_ratio"])
                )
                learning_rate = float(optimization["learning_rate"]) * scale
                optimizer.param_groups[0]["lr"] = learning_rate
                optimizer.step()
                examples_seen += len(natural_indices) + len(replay_indices)
                elapsed = time.monotonic() - started
                natural_prediction = natural_logits.argmax(dim=-1)
                replay_prediction = replay_logits.argmax(dim=-1)
                event = {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "natural_pointer_ce": float(natural_ce.detach()),
                    "replay_pointer_ce": float(replay_ce.detach()),
                    "replay_kl": float(distill_kl.detach()),
                    "replay_gain_huber": float(gain_huber.detach()),
                    "natural_exact_pointer_accuracy": float((natural_prediction == natural_target).float().mean()),
                    "natural_point_exact_accuracy": float((natural_prediction[natural_mask] == natural_target[natural_mask]).float().mean()),
                    "natural_stop_accuracy": float((natural_prediction[~natural_mask] == natural_target[~natural_mask]).float().mean()),
                    "replay_exact_pointer_accuracy": float((replay_prediction == replay_target).float().mean()),
                    "gradient_norm": float(gradient_norm),
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
        run["status"] = "complete"
        run["completed_at"] = utc_now()
        run["stop_reason"] = "optimizer_step_cap"
        run["wall_seconds"] = time.monotonic() - started
        if optimizer_step % int(optimization["checkpoint_every_steps"]):
            save_checkpoint(args.output, router, optimizer, run)
        if release_records(language_root) != language_before:
            raise RuntimeError("language parent changed")
        if sha256_file(router_parent) != parents["router_sha256"]:
            raise RuntimeError("router parent changed")
        for name, expected in parents["vision_artifacts"].items():
            if sha256_file(vision_root / name) != expected:
                raise RuntimeError(f"vision parent changed: {name}")
        for name, expected in parents["language_adapter_files"].items():
            if sha256_file(adapter_root / name) != expected:
                raise RuntimeError(f"language adapter changed: {name}")
        run["parents_verified_unchanged"] = True
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
