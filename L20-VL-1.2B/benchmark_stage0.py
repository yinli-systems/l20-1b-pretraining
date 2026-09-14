#!/usr/bin/env python3
"""Measure bridge + frozen Base cost without downloading vision weights or data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

from modeling import MultimodalBridge, freeze, trainable_parameter_count


def synchronize() -> None:
    torch.cuda.synchronize()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_arm(model, ratio: int, batch: int, text_tokens: int, repeats: int) -> dict:
    bridge = MultimodalBridge(target_ratio=ratio).to(device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    vision = torch.randn(batch, 196, 768, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.randint(0, model.config.vocab_size, (batch, text_tokens), device="cuda")
    attention = torch.ones_like(input_ids)
    labels = input_ids.clone()
    text = model.get_input_embeddings()(input_ids).detach()

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        inputs, mask, targets = bridge.inject(text, attention, labels, vision)
        loss = model(inputs_embeds=inputs, attention_mask=mask, labels=targets).loss
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(3):
        step()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    losses = []
    for _ in range(repeats):
        start = time.perf_counter()
        losses.append(step())
        synchronize()
        durations.append(time.perf_counter() - start)
    decoder_tokens = batch * (text_tokens + bridge.spec.output_tokens + 2)
    text_prediction_tokens = batch * text_tokens
    return {
        "target_ratio": ratio,
        "input_visual_tokens": bridge.spec.input_tokens,
        "output_visual_tokens": bridge.spec.output_tokens,
        "achieved_ratio": bridge.spec.achieved_ratio,
        "trainable_bridge_parameters": trainable_parameter_count(bridge),
        "batch": batch,
        "text_tokens_per_sample": text_tokens,
        "median_step_seconds": statistics.median(durations),
        "mean_step_seconds": statistics.mean(durations),
        "min_step_seconds": min(durations),
        "max_step_seconds": max(durations),
        "text_prediction_tokens_per_second": text_prediction_tokens / statistics.mean(durations),
        "decoder_tokens_per_second": decoder_tokens / statistics.mean(durations),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "last_loss_random_inputs": losses[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    weights = args.base / "pytorch_model.bin"
    before = {
        "bytes": weights.stat().st_size,
        "mtime_ns": weights.stat().st_mtime_ns,
        "sha256": sha256(weights),
    }
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    freeze(model)
    result = {
        "scope": "random precomputed vision features plus frozen released Base; excludes vision encoder and data pipeline",
        "benchmark_only": True,
        "formal_training": False,
        "training_prediction_tokens": 0,
        "synthetic_optimizer_steps_per_arm": 3 + args.repeats,
        "parent_weights_updated": False,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "base": str(args.base.resolve()),
        "base_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_weights_before": before,
        "runner_sha256": sha256(Path(__file__)),
        "modeling_sha256": sha256(Path(__file__).with_name("modeling.py")),
        "arms": [
            run_arm(model, ratio, args.batch, args.text_tokens, args.repeats)
            for ratio in (1, 4, 9, 16)
        ],
    }
    after = {
        "bytes": weights.stat().st_size,
        "mtime_ns": weights.stat().st_mtime_ns,
        "sha256": sha256(weights),
    }
    if after != before:
        raise RuntimeError("frozen Base artifact changed during benchmark")
    result["base_weights_after"] = after
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
