#!/usr/bin/env python3
"""ABBA latency benchmark for the frozen 196-token and selected 49-token models."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from evaluate_stage_a_visual_floor import collate_candidates
from modeling import bridge_from_architecture, freeze, vision_features


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_hashes(path: Path) -> dict[str, str]:
    return {
        name: sha256_file(path / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timed_repeats(callable_, warmups: int, repeats: int) -> dict[str, object]:
    for _ in range(warmups):
        callable_()
    torch.cuda.synchronize()
    wall_ms: list[float] = []
    cuda_ms: list[float] = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        callable_()
        end_event.record()
        torch.cuda.synchronize()
        wall_ms.append(1000 * (time.perf_counter() - wall_start))
        cuda_ms.append(start_event.elapsed_time(end_event))
    return {
        "warmups": warmups,
        "repeats": repeats,
        "wall_ms": wall_ms,
        "cuda_ms": cuda_ms,
        "wall_median_ms": statistics.median(wall_ms),
        "wall_p95_ms": percentile(wall_ms, 0.95),
        "cuda_median_ms": statistics.median(cuda_ms),
        "cuda_p95_ms": percentile(cuda_ms, 0.95),
    }


def run_segment(
    arm: dict,
    base: Path,
    vision,
    pixels: torch.Tensor,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    language = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(
        language, arm["language_adapter"], is_trainable=False, local_files_only=True
    )
    freeze(language)
    bridge = bridge_from_architecture(arm["architecture"]).to(
        device="cuda", dtype=torch.bfloat16
    )
    bridge.load_state_dict(load_file(arm["checkpoint"], device="cpu"), strict=True)
    freeze(bridge)
    layer = int(arm["architecture"].get("vision_feature_layer", -1))
    with torch.inference_mode():
        cached_visual = vision_features(vision, pixels, layer)

    @torch.inference_mode()
    def cached_vision_call() -> None:
        text = language.get_input_embeddings()(input_ids)
        inputs, mask, _ = bridge.inject(text, attention, None, cached_visual)
        language(inputs_embeds=inputs, attention_mask=mask, use_cache=False)

    @torch.inference_mode()
    def end_to_end_gpu_call() -> None:
        visual = vision_features(vision, pixels, layer)
        text = language.get_input_embeddings()(input_ids)
        inputs, mask, _ = bridge.inject(text, attention, None, visual)
        language(inputs_embeds=inputs, attention_mask=mask, use_cache=False)

    torch.cuda.reset_peak_memory_stats()
    cached = timed_repeats(cached_vision_call, warmups, repeats)
    end_to_end = timed_repeats(end_to_end_gpu_call, warmups, repeats)
    result = {
        "arm": arm["name"],
        "input_visual_tokens": 196,
        "output_visual_tokens": bridge.spec.output_tokens,
        "batch_candidate_sequences": int(input_ids.shape[0]),
        "text_sequence_width": int(input_ids.shape[1]),
        "cached_vision_bridge_plus_lm": cached,
        "gpu_image_tensor_to_candidate_logits": end_to_end,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    del language, bridge, cached_visual
    torch.cuda.empty_cache()
    return result


def aggregate(segments: list[dict[str, object]], key: str) -> dict[str, object]:
    values: dict[str, list[float]] = {}
    for segment in segments:
        values.setdefault(str(segment["arm"]), []).extend(segment[key]["wall_ms"])
    result = {}
    for arm, timings in values.items():
        result[arm] = {
            "samples": len(timings),
            "wall_median_ms": statistics.median(timings),
            "wall_p95_ms": percentile(timings, 0.95),
        }
    full = result["full_196"]["wall_median_ms"]
    compressed = result["answer_49"]["wall_median_ms"]
    result["answer_49_vs_full_196"] = {
        "median_latency_reduction_percent": 100 * (full - compressed) / full,
        "median_speedup_x": full / compressed,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-protocol", type=Path, required=True)
    parser.add_argument("--compressed-protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-families", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    full_protocol = json.loads(args.full_protocol.read_text())
    compressed_protocol = json.loads(args.compressed_protocol.read_text())
    base = Path(full_protocol["parents"]["language"])
    vision_path = Path(full_protocol["parents"]["vision"])
    arms = {
        "full_196": {
            "name": "full_196",
            "architecture": full_protocol["architecture"],
            "checkpoint": Path(full_protocol["parents"]["bridge"]),
            "language_adapter": Path(full_protocol["parents"]["language_adapter"]),
        },
        "answer_49": {
            "name": "answer_49",
            "architecture": compressed_protocol["architecture"],
            "checkpoint": Path("runs/stage-c-answer-49-seed20260921-v1/step-000875/bridge.safetensors"),
            "language_adapter": Path("runs/stage-c-answer-49-seed20260921-v1/step-000875/language_adapter"),
        },
    }
    rows = sorted(
        [json.loads(line) for line in args.manifest.read_text().splitlines() if line and json.loads(line)["split"] == "test"],
        key=lambda row: row["scene_family_id"],
    )[: args.batch_families]
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_path, local_files_only=True)
    prompts, answers, images = [], [], []
    for row in rows:
        with Image.open(row["base_image_path"]) as image:
            loaded = image.convert("RGB").copy()
        for answer in row["candidate_answers"]:
            prompts.append(row["question"])
            answers.append(answer)
            images.append(loaded.copy())
    input_ids, attention, _ = collate_candidates(tokenizer, prompts, answers, 96)
    input_ids = input_ids.cuda(non_blocking=True)
    attention = attention.cuda(non_blocking=True)
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
        "cuda", dtype=torch.bfloat16
    )
    vision = SiglipVisionModel.from_pretrained(
        vision_path, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(20260913)
    torch.cuda.manual_seed_all(20260913)

    segments = []
    for name in ("full_196", "answer_49", "answer_49", "full_196"):
        segments.append(
            run_segment(
                arms[name], base, vision, pixels, input_ids, attention,
                args.warmups, args.repeats,
            )
        )
    result = {
        "schema_version": "2026-09-13-v1",
        "benchmark_design": "ABBA, separately prewarmed arms, fixed real test images and candidate prompts",
        "scope": "single_L20_bf16_inference_prefill_latency",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "manifest_sha256": sha256_file(args.manifest),
        "runner_sha256": sha256_file(Path(__file__)),
        "arms": {
            name: {
                "checkpoint_sha256": sha256_file(arm["checkpoint"]),
                "language_adapter_sha256": adapter_hashes(arm["language_adapter"]),
            }
            for name, arm in arms.items()
        },
        "segments": segments,
        "aggregate": {
            "cached_vision_bridge_plus_lm": aggregate(segments, "cached_vision_bridge_plus_lm"),
            "gpu_image_tensor_to_candidate_logits": aggregate(segments, "gpu_image_tensor_to_candidate_logits"),
        },
        "claim_boundary": (
            "The benchmark includes GPU vision encoding, bridge compression, and language-model prefill from preprocessed GPU image tensors. "
            "It excludes disk I/O, CPU image decode/preprocessing, autoregressive decoding, concurrency, and real deployment serving overhead."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
