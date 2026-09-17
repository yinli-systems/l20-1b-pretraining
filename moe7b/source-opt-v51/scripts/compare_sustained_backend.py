#!/usr/bin/env python3
"""Aggregate three paired sustained native/scattered training confirmations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def read(path: Path, warmup: int) -> dict:
    training = []
    validation = []
    with (path / "metrics.jsonl").open() as handle:
        for line in handle:
            value = json.loads(line)
            if "tokens_per_second" in value and value["step"] > warmup:
                training.append(value)
            if "validation_cross_entropy" in value:
                validation.append(value)
    status = json.loads((path / "training-status.json").read_text())
    if status["status"] != "COMPLETED" or not training or not validation:
        raise ValueError(f"incomplete sustained run: {path}")
    tokens = sum(value["tokens_per_second"] * value["step_seconds"] for value in training)
    seconds = sum(value["step_seconds"] for value in training)
    return {
        "path": str(path),
        "steps": len(training),
        "weighted_tokens_per_second": tokens / seconds,
        "median_tokens_per_second": statistics.median(
            value["tokens_per_second"] for value in training
        ),
        "median_causal_useful_matmul_mfu": statistics.median(
            value["causal_useful_matmul_mfu"] for value in training
        ),
        "final_training_cross_entropy": training[-1]["cross_entropy"],
        "validation_cross_entropy": validation[-1]["validation_cross_entropy"],
        "peak_memory_allocated_bytes": max(
            value["peak_memory_allocated_bytes"] for value in training
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--profile", choices=("backend", "liger"), default="backend")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.profile == "backend":
        definitions = (
            (20260916, "P1-A-native", "P1-B-scattermoe"),
            (20260917, "P2-A-native", "P2-B-scattermoe"),
            (20260918, "P3-A-native", "P3-B-scattermoe"),
        )
    else:
        definitions = (
            (20260916, "P1-A-eager-mb8", "P1-B-liger-mb32"),
            (20260917, "P2-A-eager-mb8", "P2-B-liger-mb32"),
            (20260918, "P3-A-eager-mb8", "P3-B-liger-mb32"),
        )
    pairs = []
    for seed, baseline_name, candidate_name in definitions:
        baseline = read(arguments.root / baseline_name, arguments.warmup)
        candidate = read(arguments.root / candidate_name, arguments.warmup)
        pairs.append(
            {
                "seed": seed,
                "baseline": baseline,
                "candidate": candidate,
                "throughput_ratio": candidate["weighted_tokens_per_second"]
                / baseline["weighted_tokens_per_second"],
                "validation_cross_entropy_delta": (
                    candidate["validation_cross_entropy"]
                    - baseline["validation_cross_entropy"]
                ),
            }
        )
    throughput_ratios = [pair["throughput_ratio"] for pair in pairs]
    quality_deltas = [pair["validation_cross_entropy_delta"] for pair in pairs]
    quality_mean = statistics.mean(quality_deltas)
    # One-sided 95% Student-t critical value for df=2.
    quality_upper = quality_mean + 2.919986 * statistics.stdev(quality_deltas) / math.sqrt(3)
    result = {
        "protocol": (
            "three paired seeds, same allocation, 381 steps/run, 20-step "
            "throughput exclusion, 8 frozen validation batches"
        ),
        "profile": arguments.profile,
        "pairs": pairs,
        "mean_throughput_ratio": statistics.mean(throughput_ratios),
        "minimum_throughput_ratio": min(throughput_ratios),
        "mean_validation_cross_entropy_delta": quality_mean,
        "one_sided_95_validation_delta_upper": quality_upper,
        "quality_noninferiority_margin": 0.002,
        "promote_backend": (
            min(throughput_ratios) >= 1.05 and quality_upper <= 0.002
        ),
        "claim_boundary": (
            "Promotion here qualifies the proxy backend only; it does not "
            "promote CVCR or authorize 7B training."
        ),
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
