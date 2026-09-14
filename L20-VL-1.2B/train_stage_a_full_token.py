#!/usr/bin/env python3
"""Train the bounded Stage-A 196-token visual bridge on one L20."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from modeling import MultimodalBridge, freeze


ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def release_records(base: Path) -> dict[str, dict[str, int | str]]:
    manifest = json.loads((base / "release-manifest.json").read_text())
    records = {}
    for name, expected in sorted(manifest["files"].items()):
        path = base / name
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != {"bytes": expected["bytes"], "sha256": expected["sha256"]}:
            raise RuntimeError(f"language parent failed release manifest: {name}")
        records[name] = actual
    return records


def vision_records(vision: Path) -> dict[str, dict[str, int | str]]:
    records = {}
    for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
        path = vision / name
        records[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    expected = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
    if records["model.safetensors"]["sha256"] != expected:
        raise RuntimeError("vision parent weight hash mismatch")
    return records


def read_rows(path: Path, source: str, count: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line
    ]
    rows = [row for row in rows if row["source"] == source and row["split"] == "train"]
    rows.sort(key=lambda row: row["selection_sha256"])
    if len(rows) < count:
        raise RuntimeError(f"{source} manifest has {len(rows)} train rows, need {count}")
    return rows[:count]


def balanced_rows(open_images: list[dict], clevr: list[dict], seed: int) -> list[dict]:
    if len(open_images) != len(clevr):
        raise ValueError("Stage-A sources must be exactly balanced")
    rows = [record for pair in zip(open_images, clevr) for record in pair]
    random.Random(seed).shuffle(rows)
    return rows


def encode_prompt_response(
    tokenizer, prompt: str, response: str, max_tokens: int
) -> tuple[list[int], list[int]]:
    prompt_text = f"User: {prompt.strip()}\nAssistant:"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(
        " " + response.strip(), add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    if len(response_ids) >= max_tokens:
        response_ids = response_ids[: max_tokens - 1] + [tokenizer.eos_token_id]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[: max_tokens - len(response_ids)]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    if not input_ids or not any(label != -100 for label in labels):
        raise RuntimeError("empty prediction target after tokenization")
    return input_ids, labels


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


@dataclass
class BatchBuilder:
    tokenizer: Any
    processor: Any
    max_tokens: int

    def __call__(self, rows: list[dict]) -> dict[str, Any]:
        images = []
        encoded = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB").copy())
            encoded.append(
                encode_prompt_response(
                    self.tokenizer, row["prompt"], row["response"], self.max_tokens
                )
            )
        width = max(len(input_ids) for input_ids, _ in encoded)
        input_ids = torch.full(
            (len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for index, (ids, targets) in enumerate(encoded):
            length = len(ids)
            input_ids[index, :length] = torch.tensor(ids)
            labels[index, :length] = torch.tensor(targets)
            attention[index, :length] = 1
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": pixels.contiguous(),
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "sources": [row["source"] for row in rows],
        }


def cosine_lr(step: int, total: int, warmup: int, minimum_ratio: float) -> float:
    if warmup and step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def save_checkpoint(
    output: Path,
    bridge: MultimodalBridge,
    optimizer: torch.optim.Optimizer,
    state: dict,
) -> None:
    checkpoint = output / f"step-{state['optimizer_step']:06d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in bridge.state_dict().items()
    }
    save_file(tensors, checkpoint / "bridge.safetensors")
    optimizer_path = checkpoint / "optimizer.pt.partial"
    torch.save(optimizer.state_dict(), optimizer_path)
    os.replace(optimizer_path, checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )
    latest = output / "latest.json"
    temporary = latest.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "optimizer_step": state["optimizer_step"],
                "bridge_sha256": sha256_file(checkpoint / "bridge.safetensors"),
                "state_sha256": sha256_file(checkpoint / "state.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, latest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "stage_a_training_protocol.json")
    parser.add_argument("--data-audit", type=Path, default=ROOT / "evidence" / "stage-a-human-audit.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    human_audit = json.loads(args.data_audit.read_text())
    if human_audit.get("decision") != "pass_for_stage_a_research_training":
        raise SystemExit("Stage-A human audit has not admitted training")
    if human_audit.get("formal_release_authorized") is not False:
        raise SystemExit("human audit must not authorize release")
    if not protocol.get("training_authorized_after_human_audit_pass"):
        raise SystemExit("training protocol is not authorized")
    optimization = protocol["optimization"]
    requested_steps = args.max_optimizer_steps or optimization["max_optimizer_steps"]
    max_steps = min(requested_steps, optimization["max_optimizer_steps"])
    requested_wall = args.max_wall_seconds or optimization["max_wall_seconds"]
    max_wall_seconds = min(requested_wall, optimization["max_wall_seconds"])
    if max_steps < 1 or max_wall_seconds < 1:
        raise SystemExit("invalid bounded run")

    base = Path(protocol["parents"]["language"])
    vision_path = Path(protocol["parents"]["vision"])
    data_root = Path("/home/hhai/l20-vl-1.2b/data/stage-a-v1/prepared")
    open_images_manifest = data_root / "open-images-manifest.jsonl"
    clevr_manifest = data_root / "clevr-manifest.jsonl"
    parent_before = {
        "language": release_records(base),
        "vision": vision_records(vision_path),
    }
    manifest_hashes = {
        "open_images": sha256_file(open_images_manifest),
        "clevr": sha256_file(clevr_manifest),
    }
    if manifest_hashes != human_audit["training_manifest_sha256"]:
        raise SystemExit("human audit manifest hashes do not match")

    seed = optimization["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_path, local_files_only=True)
    source_count = protocol["data"]["open_images_localized_narratives_train_examples"]
    rows = balanced_rows(
        read_rows(open_images_manifest, "open_images_localized_narratives", source_count),
        read_rows(clevr_manifest, "clevr_v1_0", protocol["data"]["clevr_train_examples"]),
        seed,
    )
    dataset = ManifestDataset(rows)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=optimization["micro_batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=BatchBuilder(tokenizer, processor, optimization["text_tokens_max"]),
        drop_last=True,
    )

    language = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language.config.use_cache = False
    freeze(language)
    language.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    vision = SiglipVisionModel.from_pretrained(
        vision_path, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    bridge = MultimodalBridge(target_ratio=1).to(device="cuda", dtype=torch.bfloat16)
    bridge.train()
    if any(parameter.requires_grad for parameter in language.parameters()):
        raise RuntimeError("language parent is not frozen")
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise RuntimeError("vision parent is not frozen")
    trainable = sum(parameter.numel() for parameter in bridge.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
        fused=True,
    )
    accumulation = optimization["gradient_accumulation_steps"]
    warmup = max(1, round(max_steps * optimization["warmup_ratio"]))
    args.output.mkdir(parents=True)
    log_path = args.output / "train.jsonl"
    run = {
        "schema_version": "2026-09-13-v1",
        "status": "running",
        "started_at": utc_now(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "protocol_sha256": sha256_file(args.protocol),
        "human_audit_sha256": sha256_file(args.data_audit),
        "runner_sha256": sha256_file(Path(__file__)),
        "modeling_sha256": sha256_file(ROOT / "modeling.py"),
        "parent_before": parent_before,
        "manifest_hashes": manifest_hashes,
        "trainable_parameters": trainable,
        "examples": len(rows),
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
            text_embeddings = text_embeddings.detach()
            inputs, mask, targets = bridge.inject(
                text_embeddings, attention, labels, visual_features
            )
            output = language(inputs_embeds=inputs, attention_mask=mask, labels=targets)
            loss = output.loss
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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                bridge.parameters(), optimization["gradient_clip_norm"]
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at step {optimizer_step}")
            lr_scale = cosine_lr(
                optimizer_step, max_steps, warmup, optimization["minimum_learning_rate_ratio"]
            )
            learning_rate = optimization["learning_rate"] * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            event = {
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "loss_mean": accumulated_loss / accumulation,
                "gradient_norm": float(grad_norm),
                "learning_rate": learning_rate,
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
        parent_after = {
            "language": release_records(base),
            "vision": vision_records(vision_path),
        }
        if parent_after != parent_before:
            raise RuntimeError("immutable parent changed during Stage-A run")
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
