#!/usr/bin/env python3
"""Summarize a same-allocation A/B/B/A performance screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def records(path: Path, warmup: int) -> list[dict]:
    values = []
    with (path / "metrics.jsonl").open() as handle:
        for line in handle:
            value = json.loads(line)
            if "tokens_per_second" in value and value["step"] > warmup:
                values.append(value)
    if not values:
        raise ValueError(f"no post-warmup records: {path}")
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def summarize(paths: list[Path], warmup: int) -> tuple[dict, list[dict]]:
    cells = [records(path, warmup) for path in paths]
    flat = [item for cell in cells for item in cell]
    tokens_per_step = [
        item["tokens_per_second"] * item["step_seconds"] for item in flat
    ]
    result = {
        "runs": [str(path) for path in paths],
        "post_warmup_steps": len(flat),
        "median_tokens_per_second": statistics.median(
            item["tokens_per_second"] for item in flat
        ),
        "weighted_tokens_per_second": sum(tokens_per_step)
        / sum(item["step_seconds"] for item in flat),
        "p95_step_seconds": percentile(
            [item["step_seconds"] for item in flat], 0.95
        ),
        "median_causal_useful_matmul_mfu": statistics.median(
            item["causal_useful_matmul_mfu"] for item in flat
        ),
        "last_cross_entropy_by_run": [cell[-1]["cross_entropy"] for cell in cells],
    }
    return result, cells


def paired_block_interval(
    baseline: list[list[dict]], candidate: list[list[dict]], draws: int = 20_000
) -> tuple[float, float]:
    # Average the two repetitions at each matched training step, then resample
    # contiguous 10-step blocks to retain short-range timing correlation.
    size = min(*(len(cell) for cell in baseline + candidate))
    baseline_steps = [
        statistics.mean(cell[index]["tokens_per_second"] for cell in baseline)
        for index in range(size)
    ]
    candidate_steps = [
        statistics.mean(cell[index]["tokens_per_second"] for cell in candidate)
        for index in range(size)
    ]
    block = min(10, size)
    starts = list(range(0, size - block + 1))
    generator = random.Random(20260917)
    ratios = []
    for _ in range(draws):
        selected = []
        while len(selected) < size:
            start = generator.choice(starts)
            selected.extend(range(start, start + block))
        selected = selected[:size]
        ratios.append(
            statistics.mean(candidate_steps[index] for index in selected)
            / statistics.mean(baseline_steps[index] for index in selected)
        )
    ratios.sort()
    return ratios[int(0.025 * draws)], ratios[int(0.975 * draws)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b2", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    baseline, baseline_cells = summarize([arguments.a1, arguments.a2], arguments.warmup)
    candidate, candidate_cells = summarize([arguments.b1, arguments.b2], arguments.warmup)
    lower, upper = paired_block_interval(baseline_cells, candidate_cells)
    ratio = candidate["weighted_tokens_per_second"] / baseline["weighted_tokens_per_second"]
    p95_ratio = candidate["p95_step_seconds"] / baseline["p95_step_seconds"]
    baseline_ce = baseline["last_cross_entropy_by_run"]
    candidate_ce = candidate["last_cross_entropy_by_run"]
    max_ce_delta = max(
        abs(left - right) for left in baseline_ce for right in candidate_ce
    )
    result = {
        "protocol": "same-allocation ABBA, 20 warm-up plus 100 measured steps/cell",
        "baseline": baseline,
        "candidate": candidate,
        "weighted_throughput_ratio": ratio,
        "paired_block_bootstrap_95_ratio_interval": [lower, upper],
        "candidate_p95_step_time_ratio": p95_ratio,
        "max_absolute_final_cross_entropy_delta": max_ce_delta,
        "promotion_thresholds": {
            "weighted_throughput_ratio_min": 1.05,
            "paired_interval_lower_min": 1.02,
            "p95_step_time_ratio_max": 1.05,
            "max_absolute_final_cross_entropy_delta": 1e-5,
        },
        "promote_to_sustained_confirmation": (
            ratio >= 1.05
            and lower > 1.02
            and p95_ratio <= 1.05
            and max_ce_delta <= 1e-5
        ),
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
