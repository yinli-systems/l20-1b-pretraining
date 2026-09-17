#!/usr/bin/env python3
"""Exactness, memory, and latency gate for tiled full-vocabulary CE."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.nn import functional as F

from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--tokens", type=int, default=8192)
    result.add_argument("--hidden", type=int, default=768)
    result.add_argument("--vocab", type=int, default=50280)
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--steps", type=int, default=10)
    return result


def run(
    backend: str,
    hidden_seed: torch.Tensor,
    weight_seed: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    hidden = hidden_seed.detach().clone().requires_grad_(True)
    weight = weight_seed.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if backend == "eager":
            logits = F.linear(hidden, weight)
            loss = F.cross_entropy(logits.float(), targets)
        else:
            loss = LigerFusedLinearCrossEntropyLoss(
                reduction="mean", accum_dtype=torch.float32
            )(weight, hidden, targets)
    loss.backward()
    return loss.item(), hidden.grad.detach(), weight.grad.detach()


def error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (candidate.float() - reference.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "relative_l2": torch.linalg.vector_norm(delta).item()
        / max(torch.linalg.vector_norm(reference.float()).item(), 1e-30),
    }


def benchmark(
    backend: str,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    for _ in range(warmup):
        run(backend, hidden, weight, targets)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(steps):
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        run(backend, hidden, weight, targets)
        ended.record()
        ended.synchronize()
        samples.append(started.elapsed_time(ended))
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": sorted(samples)[int(0.9 * (len(samples) - 1))],
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    arguments = parser().parse_args()
    torch.manual_seed(20260917)
    hidden = torch.randn(arguments.tokens, arguments.hidden, device="cuda")
    weight = torch.randn(arguments.vocab, arguments.hidden, device="cuda") * 0.02
    targets = torch.randint(arguments.vocab, (arguments.tokens,), device="cuda")
    reference = run("eager", hidden, weight, targets)
    candidate = run("liger", hidden, weight, targets)
    eager_timing = benchmark(
        "eager", hidden, weight, targets, arguments.warmup, arguments.steps
    )
    liger_timing = benchmark(
        "liger", hidden, weight, targets, arguments.warmup, arguments.steps
    )
    result = {
        "kind": "liger_full_vocabulary_head_gate",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "tokens": arguments.tokens,
        "hidden": arguments.hidden,
        "vocab": arguments.vocab,
        "loss_absolute_error": abs(candidate[0] - reference[0]),
        "hidden_gradient_error": error(candidate[1], reference[1]),
        "weight_gradient_error": error(candidate[2], reference[2]),
        "eager": eager_timing,
        "liger": liger_timing,
        "speedup": eager_timing["median_ms"] / liger_timing["median_ms"],
        "allocated_memory_reduction_bytes": (
            eager_timing["peak_memory_allocated_bytes"]
            - liger_timing["peak_memory_allocated_bytes"]
        ),
    }
    result["operator_gate_passed"] = (
        result["loss_absolute_error"] <= 1e-4
        and result["hidden_gradient_error"]["max_abs"] <= 1e-4
        and result["weight_gradient_error"]["max_abs"] <= 1e-4
    )
    result["timestamp_unix"] = time.time()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
