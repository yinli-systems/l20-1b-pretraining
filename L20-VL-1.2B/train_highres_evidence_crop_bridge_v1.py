#!/usr/bin/env python3
"""Train an isolated crop branch while keeping every released parent frozen."""
from __future__ import annotations

import argparse
import copy
import json
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

from modeling import bridge_from_architecture, freeze, vision_features
from train_stage_a_full_token import cosine_lr, encode_prompt_response, release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_highres_evidence_crop_bridge_train_only_v1"
CANDIDATES = [str(value) for value in range(10)]


def load_train_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [
        row for row in rows
        if row["split"] == "train" and row["task"] == "read_local_digit"
    ]
    if any(row["split"] != "train" for row in selected):
        raise RuntimeError("non-train row reached crop-bridge trainer")
    selected.sort(key=lambda row: (row["family_id"], row["variant"]))
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        by_family.setdefault(row["family_id"], []).append(row)
    if any(
        [row["variant"] for row in family] != ["answer_change", "base"]
        for family in by_family.values()
    ):
        raise RuntimeError("each train family must contain answer_change/base variants")
    return selected


def epoch_rows(rows: list[dict[str, Any]], seed: int, epoch: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family_id"], []).append(row)
    families = sorted(by_family)
    random.Random(seed + epoch).shuffle(families)
    return [row for family in families for row in by_family[family]]


def crop_image(image: Image.Image, normalized_box: list[float]) -> Image.Image:
    pixels = tuple(round(float(value) * image.width) for value in normalized_box)
    if not (0 <= pixels[0] < pixels[2] <= image.width and 0 <= pixels[1] < pixels[3] <= image.height):
        raise ValueError(f"invalid crop box: {pixels}")
    return image.crop(pixels)


def cache_vision_features(rows, protocol, processor, vision, dtype):
    root = Path(protocol["data"]["root"])
    cache = {}
    batch_size = int(protocol["optimization"]["feature_batch_images"])
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            globals_, crops = [], []
            for row in batch:
                with Image.open(root / row["image_path"]) as source:
                    image = source.convert("RGB")
                globals_.append(image)
                crops.append(crop_image(image, row["target_crop_box_xyxy_normalized"]))
            pixels = processor(images=globals_ + crops, return_tensors="pt")["pixel_values"].to(
                "cuda", dtype=dtype
            )
            features = vision_features(
                vision, pixels, int(protocol["architecture"]["vision_feature_layer"])
            ).to("cpu", dtype=dtype)
            for index, row in enumerate(batch):
                key = (row["family_id"], row["variant"])
                cache[key] = (
                    features[index].contiguous(),
                    features[len(batch) + index].contiguous(),
                )
    return cache


def collate_candidates(tokenizer, rows, max_tokens: int):
    encoded, identities = [], []
    for row_index, row in enumerate(rows):
        for candidate in CANDIDATES:
            encoded.append(encode_prompt_response(tokenizer, row["question"], candidate, max_tokens))
            identities.append((row_index, candidate))
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    for index, (ids, targets) in enumerate(encoded):
        input_ids[index, :len(ids)] = torch.tensor(ids)
        labels[index, :len(ids)] = torch.tensor(targets)
        attention[index, :len(ids)] = 1
    target_indices = torch.tensor([CANDIDATES.index(row["answer"]) for row in rows], dtype=torch.long)
    return input_ids, attention, labels, identities, target_indices


def inject_global_and_crop(parent_bridge, crop_bridge, text, text_mask, labels, global_features, crop_features):
    batch = text.shape[0]
    with torch.no_grad():
        global_visual = parent_bridge.visual_tokens(global_features)
        global_start = parent_bridge.image_start.expand(batch, -1, -1)
        global_end = parent_bridge.image_end.expand(batch, -1, -1)
    crop_visual = crop_bridge.visual_tokens(crop_features)
    crop_start = crop_bridge.image_start.expand(batch, -1, -1)
    crop_end = crop_bridge.image_end.expand(batch, -1, -1)
    inputs = torch.cat((global_start, global_visual, global_end, crop_start, crop_visual, crop_end, text), dim=1)
    prefix_length = global_visual.shape[1] + crop_visual.shape[1] + 4
    prefix_mask = torch.ones(batch, prefix_length, dtype=text_mask.dtype, device=text_mask.device)
    ignored = torch.full((batch, prefix_length), -100, dtype=labels.dtype, device=labels.device)
    return inputs, torch.cat((prefix_mask, text_mask), dim=1), torch.cat((ignored, labels), dim=1)


def candidate_scores(logits, targets, eos_token_id: int, candidates_per_row: int) -> torch.Tensor:
    shifted = targets[:, 1:]
    token_mask = shifted.ne(-100) & shifted.ne(eos_token_id)
    if not token_mask.any(dim=1).all():
        raise RuntimeError("candidate has no non-EOS answer token")
    safe = shifted.masked_fill(~token_mask, 0)
    logprob = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(
        -1, safe.unsqueeze(-1)
    ).squeeze(-1)
    mean = (logprob * token_mask).sum(dim=-1) / token_mask.sum(dim=-1)
    return mean.view(-1, candidates_per_row)


def save_checkpoint(output: Path, crop_bridge, optimizer, state: dict[str, Any]) -> None:
    checkpoint = output / f"step-{state['optimizer_step']:06d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in crop_bridge.state_dict().items()},
        checkpoint / "crop_bridge.safetensors",
    )
    partial = checkpoint / "optimizer.pt.partial"
    torch.save(optimizer.state_dict(), partial)
    os.replace(partial, checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    write_json_atomic(output / "latest.json", {
        "checkpoint": str(checkpoint),
        "optimizer_step": state["optimizer_step"],
        "crop_bridge_sha256": sha256_file(checkpoint / "crop_bridge.safetensors"),
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
        raise SystemExit("crop-bridge training is not authorized")
    if protocol.get("sealed_test_access_authorized") is not False:
        raise SystemExit("sealed test must remain inaccessible")
    if protocol.get("trainable_components") != ["isolated_crop_bridge"]:
        raise SystemExit("only the isolated crop bridge may be trainable")
    for path, key in (
        (Path(__file__), "trainer_sha256"),
        (ROOT / "test_train_highres_evidence_crop_bridge_v1.py", "test_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
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
    bridge_path = Path(parents["bridge"])
    if sha256_file(bridge_path) != parents["bridge_sha256"]:
        raise SystemExit("parent bridge mismatch")
    adapter_root = Path(parents["language_adapter"])
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(adapter_root / name) != expected:
            raise SystemExit(f"language adapter mismatch: {name}")

    rows = load_train_rows(Path(protocol["data"]["manifest"]))
    expected = protocol["data"]
    if len(rows) != int(expected["train_rows"]) or len({row["family_id"] for row in rows}) != int(expected["train_families"]):
        raise RuntimeError("unexpected train-only corpus size")
    optimization = protocol["optimization"]
    steps_per_epoch = len(rows) // (
        int(optimization["micro_batch_rows"]) * int(optimization["gradient_accumulation_steps"])
    )
    if steps_per_epoch * int(optimization["epochs"]) != int(optimization["max_optimizer_steps"]):
        raise RuntimeError("protocol optimizer-step arithmetic mismatch")
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
    feature_cache = cache_vision_features(rows, protocol, processor, vision, dtype)
    del vision
    torch.cuda.empty_cache()

    from peft import PeftModel

    language = AutoModelForCausalLM.from_pretrained(
        language_root, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(
        language, adapter_root, is_trainable=False, local_files_only=True
    )
    language.config.use_cache = False
    freeze(language)
    parent_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    parent_state = load_file(bridge_path, device="cpu")
    parent_bridge.load_state_dict(parent_state, strict=True)
    freeze(parent_bridge)
    crop_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    crop_bridge.load_state_dict(copy.deepcopy(parent_state), strict=True)
    crop_bridge.train()
    trainable = [parameter for parameter in crop_bridge.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("crop bridge has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        fused=True,
    )
    accumulation = int(optimization["gradient_accumulation_steps"])
    micro_batch = int(optimization["micro_batch_rows"])
    warmup = max(1, round(max_steps * float(optimization["warmup_ratio"])))

    args.output.mkdir(parents=True)
    log_path = args.output / "train.jsonl"
    run = {
        "schema_version": "2026-09-14-v1",
        "status": "running",
        "started_at": utc_now(),
        "device": torch.cuda.get_device_name(),
        "protocol_sha256": sha256_file(args.protocol),
        "runner_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": sha256_file(Path(protocol["data"]["manifest"])),
        "parent_bridge_sha256": sha256_file(bridge_path),
        "parent_before": {"language": language_before, "vision_artifacts": parents["vision_artifacts"]},
        "trainable_components": ["isolated_crop_bridge"],
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "frozen_language_parameters": sum(parameter.numel() for parameter in language.parameters()),
        "frozen_parent_bridge_parameters": sum(parameter.numel() for parameter in parent_bridge.parameters()),
        "cached_train_rows": len(feature_cache),
        "max_optimizer_steps": max_steps,
        "optimizer_step": 0,
        "micro_step": 0,
        "sealed_test_rows_accessed": 0,
        "development_rows_accessed": 0,
    }
    write_json_atomic(args.output / "run.json", run)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    optimizer_step = 0
    micro_step = 0
    loss_sum = accuracy_sum = 0.0
    examples_seen = candidate_sequences = 0
    stop_reason = "epochs_exhausted"
    try:
        for epoch in range(int(optimization["epochs"])):
            ordered = epoch_rows(rows, seed, epoch)
            for start in range(0, len(ordered), micro_batch):
                if time.monotonic() - started >= max_wall:
                    stop_reason = "wall_time_cap"
                    break
                batch = ordered[start:start + micro_batch]
                if len(batch) != micro_batch:
                    raise RuntimeError("partial micro-batch is not authorized")
                micro_step += 1
                input_ids, attention, labels, identities, target_indices = collate_candidates(
                    tokenizer, batch, int(optimization["text_tokens_max"])
                )
                input_ids = input_ids.cuda(non_blocking=True)
                attention = attention.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                target_indices = target_indices.cuda(non_blocking=True)
                with torch.no_grad():
                    text = language.get_input_embeddings()(input_ids)
                row_indices = [index for index, _ in identities]
                global_features = torch.stack([
                    feature_cache[(batch[index]["family_id"], batch[index]["variant"])][0]
                    for index in row_indices
                ]).to("cuda", non_blocking=True)
                crop_features = torch.stack([
                    feature_cache[(batch[index]["family_id"], batch[index]["variant"])][1]
                    for index in row_indices
                ]).to("cuda", non_blocking=True)
                inputs, mask, targets = inject_global_and_crop(
                    parent_bridge, crop_bridge, text, attention, labels, global_features, crop_features
                )
                logits = language(inputs_embeds=inputs, attention_mask=mask).logits
                scores = candidate_scores(logits, targets, tokenizer.eos_token_id, len(CANDIDATES))
                loss = F.cross_entropy(scores, target_indices)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at micro step {micro_step}")
                (loss / accumulation).backward()
                loss_sum += float(loss.detach())
                accuracy_sum += float((scores.argmax(dim=-1) == target_indices).float().mean())
                examples_seen += len(batch)
                candidate_sequences += len(identities)
                if micro_step % accumulation:
                    continue
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
                optimizer.zero_grad(set_to_none=True)
                elapsed = time.monotonic() - started
                event = {
                    "optimizer_step": optimizer_step,
                    "micro_step": micro_step,
                    "epoch": epoch + 1,
                    "loss_mean": loss_sum / accumulation,
                    "candidate_accuracy_mean": accuracy_sum / accumulation,
                    "gradient_norm": float(grad_norm),
                    "learning_rate": learning_rate,
                    "examples_seen": examples_seen,
                    "candidate_sequences": candidate_sequences,
                    "examples_per_second": examples_seen / elapsed,
                    "candidate_sequences_per_second": candidate_sequences / elapsed,
                    "elapsed_seconds": elapsed,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "timestamp": utc_now(),
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event, sort_keys=True), flush=True)
                run.update(event)
                write_json_atomic(args.output / "run.json", run)
                loss_sum = accuracy_sum = 0.0
                if optimizer_step % int(optimization["checkpoint_every_steps"]) == 0:
                    save_checkpoint(args.output, crop_bridge, optimizer, run)
                if optimizer_step >= max_steps:
                    stop_reason = "optimizer_step_cap"
                    break
            if stop_reason in {"wall_time_cap", "optimizer_step_cap"}:
                break
        run["status"] = "complete"
        run["stop_reason"] = stop_reason
        run["completed_at"] = utc_now()
        run["wall_seconds"] = time.monotonic() - started
        run["optimizer_step"] = optimizer_step
        if optimizer_step and optimizer_step % int(optimization["checkpoint_every_steps"]):
            save_checkpoint(args.output, crop_bridge, optimizer, run)
        if release_records(language_root) != language_before:
            raise RuntimeError("language parent changed during crop-bridge training")
        if sha256_file(bridge_path) != parents["bridge_sha256"]:
            raise RuntimeError("parent bridge changed during crop-bridge training")
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
