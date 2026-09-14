#!/usr/bin/env python3
"""Recover the full-token teacher on audited counterfactual train families."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from modeling import MultimodalBridge, freeze
from train_stage_a_full_token import (
    BatchBuilder,
    ManifestDataset,
    cosine_lr,
    release_records,
    save_checkpoint,
    sha256_file,
    utc_now,
    vision_records,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parent


def recovery_rows(
    manifest: Path,
    split: str,
    epochs: int,
    seed: int,
    task_repeats: dict[str, int] | None = None,
    expected_families: int = 3500,
) -> list[dict]:
    families = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    selected = [row for row in families if row["split"] == split]
    if split != "train" or any(row["split"] != "train" for row in selected):
        raise RuntimeError("Stage-A recovery may use only the train split")
    if len(selected) != expected_families or len({row["scene_family_id"] for row in selected}) != expected_families:
        raise RuntimeError("unexpected counterfactual train-family count")
    repeats = task_repeats or {row["task"]: 1 for row in selected}
    if set(repeats) != {row["task"] for row in selected}:
        raise RuntimeError("task repeat keys do not match manifest tasks")
    if any(not isinstance(value, int) or value < 1 for value in repeats.values()):
        raise RuntimeError("task repeats must be positive integers")
    rows = []
    for epoch in range(epochs):
        for family in selected:
            for repeat in range(repeats[family["task"]]):
                for variant in ("base", "edited", "invariant"):
                    key = f"{epoch}|{repeat}|{family['scene_family_id']}|{variant}"
                    rows.append(
                        {
                            "source": f"counterfactual_{family['task']}",
                            "scene_family_id": family["scene_family_id"],
                            "variant": variant,
                            "epoch": epoch,
                            "repeat": repeat,
                            "image_path": family[f"{variant}_image_path"],
                            "prompt": family["question"],
                            "response": family[f"{variant}_answer"],
                            "candidate_answers": family.get("candidate_answers", ["yes", "no"]),
                            "selection_sha256": hashlib.sha256(key.encode()).hexdigest(),
                        }
                    )
    random.Random(seed).shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "stage_a_recovery_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "authorized_stage_a_recovery_after_visual_floor_failure":
        raise SystemExit("recovery protocol is not authorized")
    if not protocol.get("not_a_compression_experiment") or protocol.get("proposed_delta_loss_used"):
        raise SystemExit("recovery must remain outside the compression experiment")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("test split must remain sealed")
    failure_path = Path(protocol["failure_evidence"]["path"])
    if sha256_file(failure_path) != protocol["failure_evidence"]["sha256"]:
        raise SystemExit("visual-floor failure evidence hash mismatch")
    data = protocol["data"]
    manifest = Path(data["manifest"])
    if sha256_file(manifest) != data["manifest_sha256"]:
        raise SystemExit("counterfactual manifest hash mismatch")
    audit = json.loads(Path(data["human_audit"]).read_text())
    if audit.get("decision") != data["human_audit_decision"]:
        raise SystemExit("counterfactual human audit mismatch")
    optimization = protocol["optimization"]
    max_steps = min(args.max_optimizer_steps or optimization["max_optimizer_steps"], optimization["max_optimizer_steps"])
    max_wall_seconds = min(args.max_wall_seconds or optimization["max_wall_seconds"], optimization["max_wall_seconds"])
    if max_steps < 1 or max_wall_seconds < 1:
        raise SystemExit("invalid bounded run")

    base = Path(protocol["parents"]["language"])
    vision_path = Path(protocol["parents"]["vision"])
    initial_bridge = Path(protocol["parents"]["stage_a_bridge"])
    if sha256_file(initial_bridge) != protocol["parents"]["stage_a_bridge_sha256"]:
        raise SystemExit("Stage-A bridge hash mismatch")
    parent_before = {"language": release_records(base), "vision": vision_records(vision_path)}
    rows = recovery_rows(manifest, data["split"], data["epochs"], optimization["seed"])
    if len(rows) != data["examples_per_epoch"] * data["epochs"]:
        raise RuntimeError("recovery example count mismatch")
    answers = {answer: sum(row["response"] == answer for row in rows) for answer in ("yes", "no")}
    if answers != data["answer_balance"]:
        raise RuntimeError(f"recovery answer imbalance: {answers}")

    seed = optimization["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_path, local_files_only=True)
    loader = DataLoader(
        ManifestDataset(rows),
        batch_size=optimization["micro_batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=BatchBuilder(tokenizer, processor, optimization["text_tokens_max"]),
        drop_last=True,
    )
    language = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).cuda()
    language.config.use_cache = False
    freeze(language)
    language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vision = SiglipVisionModel.from_pretrained(vision_path, dtype=torch.bfloat16, local_files_only=True).cuda()
    freeze(vision)
    bridge = MultimodalBridge(target_ratio=1).to(device="cuda", dtype=torch.bfloat16)
    bridge.load_state_dict(load_file(initial_bridge, device="cpu"), strict=True)
    bridge.train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=optimization["learning_rate"], weight_decay=optimization["weight_decay"], fused=True)
    accumulation = optimization["gradient_accumulation_steps"]
    warmup = max(1, round(max_steps * optimization["warmup_ratio"]))
    args.output.mkdir(parents=True)
    log_path = args.output / "train.jsonl"
    run = {
        "schema_version": "2026-09-13-v1",
        "status": "running",
        "started_at": utc_now(),
        "device": torch.cuda.get_device_name(),
        "protocol_sha256": sha256_file(args.protocol),
        "runner_sha256": sha256_file(Path(__file__)),
        "modeling_sha256": sha256_file(ROOT / "modeling.py"),
        "manifest_sha256": sha256_file(manifest),
        "initial_bridge_sha256": sha256_file(initial_bridge),
        "parent_before": parent_before,
        "examples": len(rows),
        "answer_counts": answers,
        "max_optimizer_steps": max_steps,
        "max_wall_seconds": max_wall_seconds,
        "optimizer_step": 0,
        "micro_step": 0,
        "prediction_tokens": 0,
    }
    write_json_atomic(args.output / "run.json", run)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    accumulated_loss = 0.0
    optimizer_step = 0
    prediction_tokens = 0
    total_tokens = 0
    stop_reason = "data_exhausted"
    try:
        for micro_step, batch in enumerate(loader, 1):
            if time.monotonic() - started >= max_wall_seconds:
                stop_reason = "wall_time_cap"
                break
            pixels = batch["pixel_values"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention = batch["attention_mask"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            with torch.no_grad():
                visual_features = vision(pixel_values=pixels).last_hidden_state
                text_embeddings = language.get_input_embeddings()(input_ids)
            inputs, mask, targets = bridge.inject(text_embeddings.detach(), attention, labels, visual_features)
            loss = language(inputs_embeds=inputs, attention_mask=mask, labels=targets).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at micro step {micro_step}")
            (loss / accumulation).backward()
            accumulated_loss += float(loss.detach())
            prediction_tokens += int((labels != -100).sum())
            total_tokens += int(attention.sum()) + labels.shape[0] * 198
            run["micro_step"] = micro_step
            if micro_step % accumulation:
                continue
            optimizer_step += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), optimization["gradient_clip_norm"])
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at step {optimizer_step}")
            lr = optimization["learning_rate"] * cosine_lr(optimizer_step, max_steps, warmup, optimization["minimum_learning_rate_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            event = {
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "loss_mean": accumulated_loss / accumulation,
                "gradient_norm": float(grad_norm),
                "learning_rate": lr,
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
            accumulated_loss = 0.0
            run.update(event)
            write_json_atomic(args.output / "run.json", run)
            if optimizer_step % optimization["checkpoint_every_steps"] == 0:
                save_checkpoint(args.output, bridge, optimizer, run)
            if optimizer_step >= max_steps:
                stop_reason = "optimizer_step_cap"
                break
        run["status"] = "complete"
        run["stop_reason"] = stop_reason
        run["completed_at"] = utc_now()
        run["wall_seconds"] = time.monotonic() - started
        if optimizer_step and optimizer_step % optimization["checkpoint_every_steps"]:
            save_checkpoint(args.output, bridge, optimizer, run)
        parent_after = {"language": release_records(base), "vision": vision_records(vision_path)}
        if parent_after != parent_before:
            raise RuntimeError("immutable parent changed during recovery")
        run["parent_after"] = parent_after
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
