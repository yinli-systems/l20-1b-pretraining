#!/usr/bin/env python3
"""Independently recompute SciQ result bindings, metrics, and bootstraps."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import random


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap(values: list[float], *, seed: int, resamples: int) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    estimates = [sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(resamples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def paired(candidate: list[float], base: list[float], *, seed: int, resamples: int) -> list[float]:
    return bootstrap([a - b for a, b in zip(candidate, base)], seed=seed, resamples=resamples)


def close(actual: object, expected: object, tolerance: float = 1e-15) -> None:
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("list length mismatch")
        for left, right in zip(actual, expected):
            close(left, right, tolerance)
        return
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"numeric mismatch: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.source_dir / "plan.json"
    split_path = args.source_dir / "splits.json"
    data_path = args.source_dir / "sciq-test.jsonl"
    plan = json.loads(plan_path.read_text())
    split = json.loads(split_path.read_text())
    data = [json.loads(line) for line in data_path.read_text().splitlines()]
    summary_path = args.result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    seed = plan["bootstrap"]["seed"]
    resamples = plan["bootstrap"]["resamples"]
    if digest(plan_path) != summary["plan_sha256"] or digest(split_path) != plan["split_manifest_sha256"]:
        raise ValueError("plan or split digest mismatch")
    if digest(data_path) != plan["data"]["sha256"]:
        raise ValueError("data digest mismatch")
    if plan["prompt_template"] != "Question: {question}\\nAnswer:" or "support" in plan["prompt_template"].lower():
        raise ValueError("prompt is not closed-book")
    development = {row["ordinal"]: row for row in split["rows"] if row.get("role") == "development"}
    confirmation = {row["ordinal"] for row in split["rows"] if row.get("role") == "confirmation"}
    if len(development) != 489 or len(confirmation) != 490 or set(development) & confirmation:
        raise ValueError("role partition mismatch")

    reports = {}
    for candidate in plan["candidates"]:
        candidate_id = candidate["id"]
        report = json.loads((args.result_dir / f"{candidate_id}.json").read_text())
        if report["candidate"] != candidate or report["role"] != "development":
            raise ValueError("candidate or role mismatch")
        if report["plan_sha256"] != digest(plan_path) or report["data_sha256"] != digest(data_path) or report["split_manifest_sha256"] != digest(split_path):
            raise ValueError("report input binding mismatch")
        if report["result_rows_sha256"] != content_digest(report["results"]):
            raise ValueError("result content digest mismatch")
        if len(report["results"]) != 489 or {row["ordinal"] for row in report["results"]} != set(development):
            raise ValueError("development result set mismatch")
        for result in report["results"]:
            assignment = development[result["ordinal"]]
            source = data[result["ordinal"]]
            if result["row_sha256"] != assignment["row_sha256"]:
                raise ValueError("result row digest mismatch")
            options = [source["correct_answer"], source["distractor1"], source["distractor2"], source["distractor3"]]
            by_hash = {hashlib.sha256(text.encode()).hexdigest(): text for text in options}
            ordered = [by_hash[value] for value in assignment["choice_text_sha256"]]
            if result["gold"] != assignment["gold_position"] or result["gold"] != ordered.index(source["correct_answer"]):
                raise ValueError("gold or choice binding mismatch")
            norm_prediction = max(range(4), key=lambda index: (result["choice_loglikelihood_per_token"][index], -index))
            raw_prediction = max(range(4), key=lambda index: (result["choice_loglikelihood"][index], -index))
            if result["prediction_acc_norm"] != norm_prediction or result["prediction_acc"] != raw_prediction:
                raise ValueError("prediction recomputation mismatch")
            if result["correct_acc_norm"] != (norm_prediction == result["gold"]) or result["correct_acc"] != (raw_prediction == result["gold"]):
                raise ValueError("correctness recomputation mismatch")
        for metric in ["acc_norm", "acc"]:
            values = [float(row[f"correct_{metric}"]) for row in report["results"]]
            metrics = report["metrics"][metric]
            if metrics["correct"] != sum(values) or metrics["samples"] != 489:
                raise ValueError("metric count mismatch")
            close(metrics["score"], sum(values) / 489)
            close(metrics["bootstrap_ci95"], bootstrap(values, seed=seed, resamples=resamples))
        reports[candidate_id] = report

    base_id = plan["reference_candidate_id"]
    summary_by_id = {row["id"]: row for row in summary["candidates"]}
    for candidate in plan["candidates"]:
        candidate_id = candidate["id"]
        for metric in ["acc_norm", "acc"]:
            values = [float(row[f"correct_{metric}"]) for row in reports[candidate_id]["results"]]
            base = [float(row[f"correct_{metric}"]) for row in reports[base_id]["results"]]
            compact = summary_by_id[candidate_id]["metrics"][metric]
            if compact["correct"] != sum(values):
                raise ValueError("summary count mismatch")
            close(compact["score"], sum(values) / 489)
            close(compact["delta_vs_base"], sum(a - b for a, b in zip(values, base)) / 489)
            close(compact["paired_bootstrap_delta_ci95"], paired(values, base, seed=seed, resamples=resamples))
    f2_ids = [candidate["id"] for candidate in plan["candidates"] if candidate["id"] != base_id]
    for metric in ["acc_norm", "acc"]:
        base = [float(row[f"correct_{metric}"]) for row in reports[base_id]["results"]]
        f2 = [[float(row[f"correct_{metric}"]) for row in reports[candidate_id]["results"]] for candidate_id in f2_ids]
        mean = [sum(values) / len(values) for values in zip(*f2)]
        compact = summary["two_seed_mean"][metric]
        close(compact["score"], sum(mean) / 489)
        close(compact["delta_vs_base"], sum(a - b for a, b in zip(mean, base)) / 489)
        close(compact["paired_bootstrap_delta_ci95"], paired(mean, base, seed=seed, resamples=resamples))

    verification = {
        "schema": "p529m-independent-knowledge-proxy-sciq-independent-verification-v1",
        "status": "VERIFY_OK",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": args.job_id,
        "development_rows": 489,
        "confirmation_rows_scored": 0,
        "predictions_recomputed": True,
        "metrics_recomputed": True,
        "bootstrap_resamples_recomputed": resamples,
        "result_pairing_verified": True,
        "choice_and_gold_binding_verified": True,
        "plan_sha256": digest(plan_path),
        "summary_sha256": digest(summary_path),
        "result_rows_sha256": {candidate_id: report["result_rows_sha256"] for candidate_id, report in reports.items()},
        "claim_boundary": plan["claim_boundary"],
    }
    args.output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    print(f"verification_sha256={digest(args.output)}")


if __name__ == "__main__":
    main()
