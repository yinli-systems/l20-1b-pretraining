#!/usr/bin/env python3
"""Single-GPU paired profile for Top-2 and CVCR hot-path diagnosis."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from cvcr_moe.config import ModelConfig
from cvcr_moe.model import MoELanguageModel


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("configs", nargs="+", type=Path)
    result.add_argument("--microbatch", type=int, default=2)
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--steps", type=int, default=5)
    result.add_argument("--profile", action="store_true")
    return result


def run(path: Path, microbatch: int, warmup: int, steps: int, profile: bool) -> None:
    torch.manual_seed(20260916)
    config = ModelConfig.from_json(path)
    model = MoELanguageModel(config).cuda().train()
    inputs = torch.randint(
        config.vocab_size,
        (microbatch, config.max_position_embeddings),
        device="cuda",
    )
    targets = torch.randint_like(inputs, config.vocab_size)

    def iteration() -> None:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs, targets)[0]
        loss.backward()

    for _ in range(warmup):
        iteration()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        iteration()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "config": str(path),
                "method": config.method,
                "steps": steps,
                "tokens_per_second": microbatch
                * config.max_position_embeddings
                * steps
                / elapsed,
                "step_seconds": elapsed / steps,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            iteration()
            torch.cuda.synchronize()
        print(f"PROFILE {config.method}", flush=True)
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=35
            ),
            flush=True,
        )
    del model, inputs, targets
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    arguments = parser().parse_args()
    for config in arguments.configs:
        run(
            config,
            arguments.microbatch,
            arguments.warmup,
            arguments.steps,
            arguments.profile,
        )


if __name__ == "__main__":
    main()
