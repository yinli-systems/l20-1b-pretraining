#!/usr/bin/env python3
"""Summarize the frozen development-proxy baseline and paired differences."""

from __future__ import annotations

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


def paired_ci(candidate: list[bool], base: list[bool], *, seed: int, resamples: int) -> list[float]:
    if len(candidate) != len(base):
        raise ValueError("paired result length mismatch")
    rng = random.Random(seed)
    differences = [int(a) - int(b) for a, b in zip(candidate, base)]
    estimates = []
    for _ in range(resamples):
        estimates.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
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
    if set(by_id) != expected or any(report["plan_sha256"] != digest(args.plan) for report in reports):
        raise ValueError("result set or plan binding mismatch")
    base = by_id[plan["reference_candidate_id"]]
    ordered_base = sorted(base["results"], key=lambda item: item["ordinal"])
    summary_candidates = []
    for candidate in plan["candidates"]:
        report = by_id[candidate["id"]]
        rows = sorted(report["results"], key=lambda item: item["ordinal"])
        if [(x["ordinal"], x["row_sha256"]) for x in rows] != [(x["ordinal"], x["row_sha256"]) for x in ordered_base]:
            raise ValueError("candidate rows are not exactly paired")
        metric_rows = {}
        for metric in ["acc_norm", "acc"]:
            candidate_correct = [x[f"correct_{metric}"] for x in rows]
            base_correct = [x[f"correct_{metric}"] for x in ordered_base]
            delta = sum(candidate_correct) / len(rows) - sum(base_correct) / len(rows)
            metric_rows[metric] = {
                "score": report["metrics"][metric]["score"],
                "delta_vs_base": delta,
                "paired_bootstrap_delta_ci95": paired_ci(candidate_correct, base_correct,
                    seed=plan["bootstrap"]["seed"], resamples=plan["bootstrap"]["resamples"]),
            }
        summary_candidates.append({"id": candidate["id"], "seed": candidate.get("seed"), "metrics": metric_rows})
    seed_rows = [item for item in summary_candidates if item["id"] != plan["reference_candidate_id"]]
    result = {
        "schema": "p529m-independent-proxy-belebele-baseline-summary-v1",
        "status": "INDEPENDENT_PROXY_FIRST_TRANCHE_BASELINE_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": digest(args.plan),
        "reference_candidate_id": plan["reference_candidate_id"],
        "candidates": summary_candidates,
        "two_seed_mean": {metric: sum(row["metrics"][metric]["score"] for row in seed_rows) / len(seed_rows)
                          for metric in ["acc_norm", "acc"]},
        "two_seed_mean_delta_vs_base": {metric: sum(row["metrics"][metric]["delta_vs_base"] for row in seed_rows) / len(seed_rows)
                                        for metric in ["acc_norm", "acc"]},
        "formal_promotion": False,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
