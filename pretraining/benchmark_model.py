#!/usr/bin/env python3
"""Measure full forward/backward/optimizer throughput for the target 1.1B model."""

from __future__ import annotations

import argparse
import json
import time

import torch
from litgpt import Config, GPT
from litgpt.utils import chunked_cross_entropy


def synchronize() -> None:
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    config = Config.from_name("tiny-llama-1.1b")
    config.block_size = args.sequence_length
    model = GPT(config)
    model.apply(model._init_weights)
    parameter_count = sum(p.numel() for p in model.parameters())
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )
    if args.compile:
        model = torch.compile(model)

    batch_tokens = args.micro_batch_size * args.sequence_length
    input_ids = torch.randint(
        0,
        config.padded_vocab_size,
        (args.micro_batch_size, args.sequence_length),
        device="cuda",
        dtype=torch.long,
    )
    targets = torch.randint_like(input_ids, high=config.padded_vocab_size)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    for index in range(args.warmup_steps):
        started = time.monotonic()
        loss = step()
        synchronize()
        print(json.dumps({"phase": "warmup", "step": index + 1, "loss": loss, "seconds": time.monotonic() - started}), flush=True)

    torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.monotonic()
    losses: list[float] = []
    for index in range(args.measure_steps):
        losses.append(step())
        synchronize()
        print(json.dumps({"phase": "measure", "step": index + 1, "loss": losses[-1]}), flush=True)
    elapsed = time.monotonic() - started

    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model": "tiny-llama-1.1b",
        "parameters": parameter_count,
        "micro_batch_size": args.micro_batch_size,
        "sequence_length": args.sequence_length,
        "measured_steps": args.measure_steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": batch_tokens * args.measure_steps / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "final_loss": losses[-1],
        "compile": args.compile,
    }
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
