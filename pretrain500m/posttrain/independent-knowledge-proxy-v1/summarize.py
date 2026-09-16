#!/usr/bin/env python3
"""Summarize paired SciQ accuracy and two-seed F2 means."""

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


def paired_ci(candidate: list[float], base: list[float], *, seed: int, resamples: int) -> list[float]:
    if len(candidate) != len(base):
        raise ValueError("paired result length mismatch")
    rng = random.Random(seed)
    differences = [a - b for a, b in zip(candidate, base)]
    size = len(differences)
    estimates = [sum(differences[rng.randrange(size)] for _ in range(size)) / size for _ in range(resamples)]
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
    if any(report["role"] != "development" for report in reports):
        raise ValueError("confirmation scoring is forbidden in this summary")
    base = by_id[plan["reference_candidate_id"]]
    ordered_base = sorted(base["results"], key=lambda item: item["ordinal"])
    summary_candidates = []
    per_candidate_rows = {}
    for candidate in plan["candidates"]:
        report = by_id[candidate["id"]]
        rows = sorted(report["results"], key=lambda item: item["ordinal"])
        if [(x["ordinal"], x["row_sha256"], x["gold"]) for x in rows] != [(x["ordinal"], x["row_sha256"], x["gold"]) for x in ordered_base]:
            raise ValueError("candidate rows are not exactly paired")
        metric_rows = {}
        per_candidate_rows[candidate["id"]] = {}
        for metric in ["acc_norm", "acc"]:
            candidate_correct = [float(x[f"correct_{metric}"]) for x in rows]
            base_correct = [float(x[f"correct_{metric}"]) for x in ordered_base]
            per_candidate_rows[candidate["id"]][metric] = candidate_correct
            delta = sum(candidate_correct) / len(rows) - sum(base_correct) / len(rows)
            metric_rows[metric] = {
                "score": report["metrics"][metric]["score"],
                "correct": report["metrics"][metric]["correct"],
                "delta_vs_base": delta,
                "paired_bootstrap_delta_ci95": paired_ci(candidate_correct, base_correct,
                    seed=plan["bootstrap"]["seed"], resamples=plan["bootstrap"]["resamples"]),
            }
        summary_candidates.append({"id": candidate["id"], "seed": candidate.get("seed"), "metrics": metric_rows})
    f2_ids = [item["id"] for item in plan["candidates"] if item["id"] != plan["reference_candidate_id"]]
    two_seed = {}
    for metric in ["acc_norm", "acc"]:
        base_rows = per_candidate_rows[plan["reference_candidate_id"]][metric]
        mean_rows = [sum(per_candidate_rows[candidate_id][metric][index] for candidate_id in f2_ids) / len(f2_ids)
                     for index in range(len(base_rows))]
        two_seed[metric] = {
            "score": sum(mean_rows) / len(mean_rows),
            "delta_vs_base": sum(a - b for a, b in zip(mean_rows, base_rows)) / len(base_rows),
            "paired_bootstrap_delta_ci95": paired_ci(mean_rows, base_rows,
                seed=plan["bootstrap"]["seed"], resamples=plan["bootstrap"]["resamples"]),
        }
    result = {
        "schema": "p529m-independent-knowledge-proxy-sciq-summary-v1",
        "status": "INDEPENDENT_KNOWLEDGE_PROXY_DEVELOPMENT_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": digest(args.plan),
        "reference_candidate_id": plan["reference_candidate_id"],
        "candidates": summary_candidates,
        "two_seed_mean": two_seed,
        "formal_promotion": False,
        "confirmation_scored": False,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
