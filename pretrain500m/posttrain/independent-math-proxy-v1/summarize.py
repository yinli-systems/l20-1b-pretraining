#!/usr/bin/env python3
"""Summarize the frozen MGSM development proxy across Base and F2 seeds."""

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


def paired_ci(candidate: list[float], base: list[float], seed: int, resamples: int) -> list[float]:
    if len(candidate) != len(base):
        raise ValueError("paired length mismatch")
    differences = [a - b for a, b in zip(candidate, base)]
    rng = random.Random(seed)
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
    base = by_id[plan["reference_candidate_id"]]
    base_rows = sorted(base["results"], key=lambda row: row["ordinal"])
    summary_rows = []
    for candidate in plan["candidates"]:
        report = by_id[candidate["id"]]
        rows = sorted(report["results"], key=lambda row: row["ordinal"])
        if [(x["ordinal"], x["row_sha256"]) for x in rows] != [(x["ordinal"], x["row_sha256"]) for x in base_rows]:
            raise ValueError("candidate rows are not exactly paired")
        candidate_acc = [float(row["exact_numeric_match"]) for row in rows]
        base_acc = [float(row["exact_numeric_match"]) for row in base_rows]
        candidate_nll = [row["gold_answer_token_nll"] for row in rows]
        base_nll = [row["gold_answer_token_nll"] for row in base_rows]
        summary_rows.append({
            "id": candidate["id"], "seed": candidate.get("seed"),
            "exact_numeric_match": {
                "score": sum(candidate_acc) / len(candidate_acc),
                "delta_vs_base": (sum(candidate_acc) - sum(base_acc)) / len(candidate_acc),
                "paired_bootstrap_delta_ci95": paired_ci(candidate_acc, base_acc, plan["bootstrap"]["seed"], plan["bootstrap"]["resamples"]),
            },
            "gold_answer_token_nll": {
                "mean": sum(candidate_nll) / len(candidate_nll),
                "delta_vs_base": (sum(candidate_nll) - sum(base_nll)) / len(candidate_nll),
                "paired_bootstrap_delta_ci95": paired_ci(candidate_nll, base_nll, plan["bootstrap"]["seed"], plan["bootstrap"]["resamples"]),
            },
        })
    seed_rows = [row for row in summary_rows if row["id"] != plan["reference_candidate_id"]]
    result = {
        "schema": "p529m-independent-math-proxy-mgsm-summary-v1",
        "status": "INDEPENDENT_MATH_PROXY_DEVELOPMENT_BASELINE_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": digest(args.plan),
        "candidates": summary_rows,
        "two_seed_mean": {
            "exact_numeric_match": sum(row["exact_numeric_match"]["score"] for row in seed_rows) / len(seed_rows),
            "gold_answer_token_nll": sum(row["gold_answer_token_nll"]["mean"] for row in seed_rows) / len(seed_rows),
        },
        "two_seed_mean_delta_vs_base": {
            "exact_numeric_match": sum(row["exact_numeric_match"]["delta_vs_base"] for row in seed_rows) / len(seed_rows),
            "gold_answer_token_nll": sum(row["gold_answer_token_nll"]["delta_vs_base"] for row in seed_rows) / len(seed_rows),
        },
        "formal_promotion": False,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()

