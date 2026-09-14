#!/usr/bin/env python3
"""Run a no-update, real-image end-to-end systems benchmark.

This is deliberately not a training script.  Every model component executes
under inference_mode, no optimizer is constructed, and immutable parent files
are hash-checked before and after the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from modeling import MultimodalBridge, freeze


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"bytes": stat.st_size, "sha256": sha256(path)}


def immutable_parent_records(base: Path, vision: Path) -> dict[str, dict[str, dict[str, int | str]]]:
    base_manifest = json.loads((base / "release-manifest.json").read_text())
    base_files: dict[str, dict[str, int | str]] = {}
    for name, expected in sorted(base_manifest["files"].items()):
        actual = file_record(base / name)
        if actual != {"bytes": expected["bytes"], "sha256": expected["sha256"]}:
            raise RuntimeError(f"Base release file failed manifest verification: {name}")
        base_files[name] = actual
    vision_files = {
        name: file_record(vision / name)
        for name in ("config.json", "model.safetensors", "preprocessor_config.json")
    }
    return {"base": base_files, "vision": vision_files}


def tensor_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_rows(manifest_path: Path, acquisition_path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    acquisition = json.loads(acquisition_path.read_text())
    if acquisition.get("downloaded_images") != 288 or acquisition.get("failed_images") != 0:
        raise RuntimeError("the frozen 288-image acquisition receipt is not complete")
    by_selection = {row["selection_sha256"]: row for row in acquisition["results"]}
    if len(rows) != 288 or len(by_selection) != 288:
        raise RuntimeError("expected exactly 288 unique manifest and acquisition rows")
    joined = []
    for row in rows:
        acquired = by_selection.get(row["selection_sha256"])
        if acquired is None or acquired["row_id"] != row["row_id"] or acquired["family"] != row["family"]:
            raise RuntimeError(f"manifest/acquisition mismatch for row {row['row_id']}")
        image_path = Path(acquired["image_path"])
        if file_record(image_path) != {"bytes": acquired["bytes"], "sha256": acquired["image_sha256"]}:
            raise RuntimeError(f"image integrity failure for row {row['row_id']}")
        joined.append({**row, "image_path": str(image_path), "image_sha256": acquired["image_sha256"]})
    joined.sort(key=lambda item: (item["family"], item["row_id"]))
    return joined, acquisition


def prepare_inputs(rows: list[dict], processor, tokenizer, text_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    started = time.perf_counter()
    images = []
    modes = defaultdict(int)
    for row in rows:
        with Image.open(row["image_path"]) as image:
            modes[image.mode] += 1
            images.append(image.convert("RGB").copy())
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].contiguous()
    encoded = tokenizer(
        [row["caption"] for row in rows],
        padding="max_length",
        truncation=True,
        max_length=text_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].contiguous()
    attention = encoded["attention_mask"].contiguous()
    labels = input_ids.clone()
    labels[attention == 0] = -100
    elapsed = time.perf_counter() - started
    stats = {
        "seconds": elapsed,
        "images_per_second": len(rows) / elapsed,
        "pixel_tensor_shape": list(pixel_values.shape),
        "pixel_tensor_dtype": str(pixel_values.dtype),
        "source_modes": dict(sorted(modes.items())),
        "non_padding_prediction_tokens": int((labels != -100).sum()),
        "caption_token_length_mean": float(attention.sum(dim=1).float().mean()),
        "caption_token_length_p50": float(attention.sum(dim=1).float().median()),
        "captions_truncated_at_limit": int((attention.sum(dim=1) == text_tokens).sum()),
    }
    return pixel_values, input_ids, attention, labels, stats


def run_arm(
    vision,
    language,
    rows: list[dict],
    pixels: torch.Tensor,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    labels: torch.Tensor,
    ratio: int,
    batch_size: int,
) -> dict:
    print(f"START ratio={ratio} images={len(rows)} batch={batch_size}", flush=True)
    torch.manual_seed(20260913 + ratio)
    torch.cuda.manual_seed_all(20260913 + ratio)
    bridge = MultimodalBridge(target_ratio=ratio).to(device="cuda", dtype=torch.bfloat16)
    freeze(bridge)
    bridge_hash_before = tensor_state_sha256(bridge)

    def forward(start: int, end: int) -> tuple[float, int, float]:
        batch_pixels = pixels[start:end].to("cuda", dtype=torch.bfloat16, non_blocking=False)
        batch_ids = input_ids[start:end].to("cuda", non_blocking=False)
        batch_attention = attention[start:end].to("cuda", non_blocking=False)
        batch_labels = labels[start:end].to("cuda", non_blocking=False)
        visual_features = vision(pixel_values=batch_pixels).last_hidden_state
        text_embeddings = language.get_input_embeddings()(batch_ids)
        inputs, mask, targets = bridge.inject(
            text_embeddings, batch_attention, batch_labels, visual_features
        )
        output = language(inputs_embeds=inputs, attention_mask=mask, labels=targets)
        token_count = int((batch_labels != -100).sum())
        return float(output.loss), token_count, float(visual_features.float().abs().mean())

    with torch.inference_mode():
        forward(0, min(batch_size, len(rows)))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        batch_times = []
        losses = []
        feature_abs_means = []
        prediction_tokens = 0
        family_losses: dict[str, list[float]] = defaultdict(list)
        measured_start = time.perf_counter()
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            torch.cuda.synchronize()
            tick = time.perf_counter()
            loss, token_count, feature_abs_mean = forward(start, end)
            torch.cuda.synchronize()
            duration = time.perf_counter() - tick
            if not math.isfinite(loss) or not math.isfinite(feature_abs_mean):
                raise RuntimeError(f"non-finite output in ratio-{ratio} arm")
            batch_times.append(duration)
            losses.append(loss)
            feature_abs_means.append(feature_abs_mean)
            prediction_tokens += token_count
            for family in {row["family"] for row in rows[start:end]}:
                family_losses[family].append(loss)
        measured_seconds = time.perf_counter() - measured_start

    bridge_hash_after = tensor_state_sha256(bridge)
    if bridge_hash_after != bridge_hash_before:
        raise RuntimeError(f"bridge changed during forward-only ratio-{ratio} arm")
    result = {
        "target_ratio": ratio,
        "input_visual_tokens": bridge.spec.input_tokens,
        "output_visual_tokens": bridge.spec.output_tokens,
        "achieved_ratio": bridge.spec.achieved_ratio,
        "bridge_parameters": sum(parameter.numel() for parameter in bridge.parameters()),
        "bridge_requires_grad_parameters": 0,
        "bridge_state_sha256_before": bridge_hash_before,
        "bridge_state_sha256_after": bridge_hash_after,
        "batch_size": batch_size,
        "measured_batches": len(batch_times),
        "measured_images": len(rows),
        "measured_prediction_tokens": prediction_tokens,
        "measured_seconds": measured_seconds,
        "images_per_second": len(rows) / measured_seconds,
        "prediction_tokens_per_second": prediction_tokens / measured_seconds,
        "batch_seconds_mean": statistics.mean(batch_times),
        "batch_seconds_p50": statistics.median(batch_times),
        "batch_seconds_p95": sorted(batch_times)[max(0, math.ceil(0.95 * len(batch_times)) - 1)],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "caption_loss_mean_untrained_bridge": statistics.mean(losses),
        "caption_loss_p50_untrained_bridge": statistics.median(losses),
        "vision_feature_abs_mean": statistics.mean(feature_abs_means),
        "per_family_batch_loss_mean_untrained_bridge": {
            family: statistics.mean(values) for family, values in sorted(family_losses.items())
        },
    }
    print(
        f"DONE ratio={ratio} images_per_second={result['images_per_second']:.3f} "
        f"prediction_tokens_per_second={result['prediction_tokens_per_second']:.1f} "
        f"peak_gib={result['peak_allocated_gib']:.2f}",
        flush=True,
    )
    del bridge
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or args.text_tokens < 8:
        raise ValueError("invalid batch size or text-token limit")

    total_started = time.perf_counter()
    parent_before = immutable_parent_records(args.base, args.vision)
    rows, acquisition = load_rows(args.manifest, args.acquisition)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(args.vision, local_files_only=True)
    pixels, input_ids, attention, labels, preparation = prepare_inputs(
        rows, processor, tokenizer, args.text_tokens
    )

    torch.set_float32_matmul_precision("high")
    language = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language.config.use_cache = False
    freeze(language)
    vision = SiglipVisionModel.from_pretrained(
        args.vision, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    if any(parameter.requires_grad for parameter in language.parameters()):
        raise RuntimeError("language parent is not frozen")
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise RuntimeError("vision parent is not frozen")

    arms = [
        run_arm(
            vision, language, rows, pixels, input_ids, attention, labels, ratio, args.batch_size
        )
        for ratio in (1, 4, 9, 16)
    ]
    parent_after = immutable_parent_records(args.base, args.vision)
    if parent_after != parent_before:
        raise RuntimeError("an immutable parent artifact changed during the benchmark")

    result = {
        "schema_version": "2026-09-13-v1",
        "scope": "real_288_image_forward_only_system_test_with_frozen_siglip2_frozen_released_base_and_frozen_random_bridge",
        "status": "pass",
        "formal_training": False,
        "optimizer_constructed": False,
        "backward_calls": 0,
        "training_prediction_tokens": 0,
        "real_multimodal_data": True,
        "quality_claim_authorized": False,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "base_parameters": sum(parameter.numel() for parameter in language.parameters()),
        "vision_parameters": sum(parameter.numel() for parameter in vision.parameters()),
        "base_path": str(args.base),
        "vision_path": str(args.vision),
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        "acquisition_receipt": {
            "path": str(args.acquisition),
            "sha256": sha256(args.acquisition),
            "downloaded_images": acquisition["downloaded_images"],
        },
        "runner_sha256": sha256(Path(__file__)),
        "modeling_sha256": sha256(Path(__file__).with_name("modeling.py")),
        "parent_artifacts_before": parent_before,
        "parent_artifacts_after": parent_after,
        "preparation": preparation,
        "arms": arms,
        "wall_seconds_including_hashes_load_and_all_arms": time.perf_counter() - total_started,
        "claim_boundary": "This proves only that the frozen real-image path executes with finite outputs and measured systems cost. Random frozen bridge losses are not a quality metric and no parameter learned from these images.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
