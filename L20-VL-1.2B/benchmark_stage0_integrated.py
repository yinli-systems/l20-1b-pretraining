#!/usr/bin/env python3
"""Benchmark real frozen vision + bridge training + frozen Base on synthetic inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, SiglipVisionModel

from modeling import MultimodalBridge, freeze, trainable_parameter_count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def run_arm(vision, language, ratio: int, batch: int, text_tokens: int, repeats: int) -> dict:
    bridge = MultimodalBridge(target_ratio=ratio).to(device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    pixels = torch.randn(batch, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.randint(
        0, language.config.vocab_size, (batch, text_tokens), device="cuda"
    )
    attention = torch.ones_like(input_ids)
    labels = input_ids.clone()
    text = language.get_input_embeddings()(input_ids).detach()

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            visual_features = vision(pixel_values=pixels).last_hidden_state
        inputs, mask, targets = bridge.inject(
            text, attention, labels, visual_features.detach()
        )
        loss = language(inputs_embeds=inputs, attention_mask=mask, labels=targets).loss
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []
    losses: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)

    mean = statistics.mean(durations)
    result = {
        "target_ratio": ratio,
        "input_visual_tokens": bridge.spec.input_tokens,
        "output_visual_tokens": bridge.spec.output_tokens,
        "achieved_ratio": bridge.spec.achieved_ratio,
        "trainable_bridge_parameters": trainable_parameter_count(bridge),
        "batch": batch,
        "text_tokens_per_sample": text_tokens,
        "median_step_seconds": statistics.median(durations),
        "mean_step_seconds": mean,
        "text_prediction_tokens_per_second": batch * text_tokens / mean,
        "images_per_second": batch / mean,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "last_loss_random_inputs": losses[-1],
    }
    del optimizer, bridge, pixels, input_ids, attention, labels, text
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    base_weights = args.base / "pytorch_model.bin"
    vision_weights = args.vision / "model.safetensors"
    before = {
        "base": artifact(base_weights),
        "vision": artifact(vision_weights),
    }
    torch.manual_seed(20260913)
    torch.cuda.manual_seed_all(20260913)
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

    result = {
        "schema_version": "2026-09-13-v1",
        "scope": "real_frozen_siglip_forward_plus_bridge_only_optimizer_plus_frozen_released_base_on_synthetic_inputs",
        "benchmark_only": True,
        "formal_training": False,
        "real_multimodal_data": False,
        "training_prediction_tokens": 0,
        "synthetic_optimizer_steps_per_arm": 3 + args.repeats,
        "parent_weights_updated": False,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "base_parameters": sum(parameter.numel() for parameter in language.parameters()),
        "vision_parameters": sum(parameter.numel() for parameter in vision.parameters()),
        "base_weights_before": before["base"],
        "vision_weights_before": before["vision"],
        "runner_sha256": sha256(Path(__file__)),
        "modeling_sha256": sha256(Path(__file__).with_name("modeling.py")),
        "arms": [
            run_arm(vision, language, ratio, args.batch, args.text_tokens, args.repeats)
            for ratio in (1, 4, 9, 16)
        ],
        "claim_boundary": "Synthetic random images/text measure execution cost only; they do not estimate multimodal quality, convergence, or data-pipeline throughput.",
    }
    after = {
        "base": artifact(base_weights),
        "vision": artifact(vision_weights),
    }
    if after != before:
        raise RuntimeError("a frozen parent artifact changed during integrated benchmark")
    result["base_weights_after"] = after["base"]
    result["vision_weights_after"] = after["vision"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
