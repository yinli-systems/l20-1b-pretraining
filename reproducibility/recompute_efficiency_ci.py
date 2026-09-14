#!/usr/bin/env python3
"""Exactly replay the published TinyLlama-1T paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_NUMPY = "2.5.2"
TASKS = (
    "hellaswag",
    "piqa",
    "winogrande",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "boolq",
)


def replay(root: Path) -> dict[str, Any]:
    if np.__version__ != REQUIRED_NUMPY:
        raise RuntimeError(
            f"exact replay requires numpy=={REQUIRED_NUMPY}; found {np.__version__}"
        )
    bundle_path = root / "reports/metrics/efficiency-all-results-final-20260912.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    comparison = next(item for item in bundle["checkpoints"] if item["job"] == "tinyllama_1t")
    repetitions = int(bundle["protocol"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(bundle["protocol"]["bootstrap_seed"]))
    task_draws: list[np.ndarray] = []
    task_results: dict[str, Any] = {}

    for task in TASKS:
        record = comparison["tasks"][task]
        sample_count = int(record["n"])
        values = np.asarray([-1.0, 0.0, 1.0])
        counts = np.asarray(
            [
                record["baseline_better_samples"],
                record["tied_samples"],
                record["ours_better_samples"],
            ],
            dtype=np.int64,
        )
        if int(counts.sum()) != sample_count:
            raise ValueError(f"paired counts do not add up for {task}")
        draws = np.empty(repetitions)
        for start in range(0, repetitions, 256):
            size = min(256, repetitions - start)
            sampled = rng.multinomial(sample_count, counts / sample_count, size=size)
            draws[start : start + size] = (sampled @ values) / sample_count
        actual = (np.quantile(draws, [0.025, 0.975]) * 100).tolist()
        expected = record["paired_95_ci_pp"]
        if actual != expected:
            raise ValueError(f"paired CI mismatch for {task}: {actual} != {expected}")
        task_draws.append(draws)
        task_results[task] = {"recomputed": actual, "expected": expected, "exact_match": True}

    seven = (np.quantile(np.mean(task_draws, axis=0), [0.025, 0.975]) * 100).tolist()
    six = (np.quantile(np.mean(task_draws[:-1], axis=0), [0.025, 0.975]) * 100).tolist()
    expected_seven = comparison["seven_task_macro"]["paired_95_ci_pp"]
    expected_six = comparison["six_task_without_boolq"]["paired_95_ci_pp"]
    if seven != expected_seven or six != expected_six:
        raise ValueError("macro paired CI replay mismatch")
    return {
        "status": "verified_exact_statistical_replay",
        "numpy": np.__version__,
        "bootstrap_seed": bundle["protocol"]["bootstrap_seed"],
        "bootstrap_repetitions": repetitions,
        "tasks": task_results,
        "seven_task_macro_paired_95_ci_pp": seven,
        "six_task_without_boolq_paired_95_ci_pp": six,
        "boundary": (
            "This exactly replays bootstrap aggregation from published paired difference "
            "counts. It does not independently regenerate or verify model predictions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = replay(args.root.resolve())
    except (KeyError, StopIteration, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
