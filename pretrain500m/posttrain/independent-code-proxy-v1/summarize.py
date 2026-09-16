#!/usr/bin/env python3
"""Summarize the frozen MBPP development proxy across Base and F2 seeds."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_ci(candidate: list[float], base: list[float], seed: int,
              resamples: int) -> list[float]:
    if len(candidate) != len(base):
        raise ValueError("paired length mismatch")
    differences = [a - b for a, b in zip(candidate, base)]
    rng = random.Random(seed)
    size = len(differences)
    estimates = [sum(differences[rng.randrange(size)] for _ in range(size)) / size
                 for _ in range(resamples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    reports = [json.loads(path.read_text()) for path in args.result]
    by_id = {report["candidate"]["id"]: report for report in reports}
    expected = {item["id"] for item in plan["candidates"]}
    if set(by_id) != expected or any(
        report["plan_sha256"] != digest(args.plan) for report in reports
    ):
        raise ValueError("result set or plan binding mismatch")
    base = by_id[plan["reference_candidate_id"]]
    base_rows = sorted(base["results"], key=lambda row: row["ordinal"])
    summary_rows = []
    for candidate in plan["candidates"]:
        report = by_id[candidate["id"]]
        rows = sorted(report["results"], key=lambda row: row["ordinal"])
        keys = [(row["ordinal"], row["task_id"], row["row_sha256"]) for row in rows]
        base_keys = [(row["ordinal"], row["task_id"], row["row_sha256"])
                     for row in base_rows]
        if keys != base_keys:
            raise ValueError("candidate rows are not exactly paired")
        metrics = {}
        for name, field in (
            ("execution_pass_at_1", "execution_pass"),
            ("syntax_and_policy_rate", "syntax_and_policy_pass"),
            ("gold_code_token_nll", "gold_code_token_nll"),
        ):
            values = [float(row[field]) for row in rows]
            base_values = [float(row[field]) for row in base_rows]
            metrics[name] = {
                "mean": sum(values) / len(values),
                "delta_vs_base": (sum(values) - sum(base_values)) / len(values),
                "paired_bootstrap_delta_ci95": paired_ci(
                    values, base_values, plan["bootstrap"]["seed"],
                    plan["bootstrap"]["resamples"]),
            }
        summary_rows.append({"id": candidate["id"], "seed": candidate.get("seed"),
                             **metrics})
    seed_rows = [row for row in summary_rows
                 if row["id"] != plan["reference_candidate_id"]]
    names = ("execution_pass_at_1", "syntax_and_policy_rate", "gold_code_token_nll")
    result = {
        "schema": "p529m-independent-code-proxy-mbpp-summary-v1",
        "status": "INDEPENDENT_CODE_PROXY_DEVELOPMENT_BASELINE_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": digest(args.plan),
        "candidates": summary_rows,
        "two_seed_mean": {
            name: sum(row[name]["mean"] for row in seed_rows) / len(seed_rows)
            for name in names
        },
        "two_seed_mean_delta_vs_base": {
            name: sum(row[name]["delta_vs_base"] for row in seed_rows) / len(seed_rows)
            for name in names
        },
        "formal_promotion": False,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
