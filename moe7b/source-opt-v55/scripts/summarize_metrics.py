"""Summarize steady-state throughput/MFU and optional matched-run overhead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def read_steps(run: Path, warmup: int) -> list[dict]:
    records = []
    with (run / "metrics.jsonl").open() as handle:
        for line in handle:
            value = json.loads(line)
            if "tokens_per_second" in value and value["step"] > warmup:
                records.append(value)
    if not records:
        raise ValueError(f"no steady-state training records in {run}")
    return records


def summarize(run: Path, warmup: int) -> dict:
    records = read_steps(run, warmup)
    result = {
        "run": str(run),
        "steps": len(records),
        "median_tokens_per_second": statistics.median(
            item["tokens_per_second"] for item in records
        ),
        "median_active_mfu": statistics.median(item["active_mfu"] for item in records),
        "median_legacy_reported_mfu": statistics.median(
            item.get("legacy_reported_mfu", item["active_mfu"]) for item in records
        ),
        "last_cross_entropy": records[-1]["cross_entropy"],
        "all_finite": all(
            all(
                isinstance(item[key], (int, float))
                and item[key] == item[key]
                and abs(item[key]) != float("inf")
                for key in ("loss", "cross_entropy", "tokens_per_second", "active_mfu")
            )
            for item in records
        ),
    }
    for key in (
        "hardware_work_mfu_including_probes",
        "rated_full_square_mfu",
        "causal_useful_matmul_mfu",
        "modeled_main_plus_probe_matmul_utilization",
    ):
        values = [item[key] for item in records if key in item]
        if values:
            result[f"median_{key}"] = statistics.median(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = {"candidate": summarize(arguments.run, arguments.warmup)}
    if arguments.baseline:
        baseline = summarize(arguments.baseline, arguments.warmup)
        candidate = result["candidate"]
        result["baseline"] = baseline
        result["candidate_throughput_ratio"] = (
            candidate["median_tokens_per_second"] / baseline["median_tokens_per_second"]
        )
        result["candidate_wall_clock_overhead_fraction"] = (
            baseline["median_tokens_per_second"]
            / candidate["median_tokens_per_second"]
            - 1.0
        )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
