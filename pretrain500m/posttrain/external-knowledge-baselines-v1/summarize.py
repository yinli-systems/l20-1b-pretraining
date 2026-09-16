#!/usr/bin/env python3
"""Verify and summarize matched external closed-book SciQ baselines."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_interval(base: list[bool], candidate: list[bool], seed: int, resamples: int) -> list[float]:
    if len(base) != len(candidate):
        raise ValueError("paired result length mismatch")
    differences = [int(right) - int(left) for left, right in zip(base, candidate)]
    rng = random.Random(seed)
    size = len(differences)
    values = [sum(differences[rng.randrange(size)] for _ in range(size)) / size
              for _ in range(resamples)]
    return [percentile(values, 0.025), percentile(values, 0.975)]


def validate_rows(report: dict) -> None:
    rows = report["results"]
    if len(rows) != 489 or report["role"] != "development":
        raise ValueError("unexpected evaluation role or row count")
    for metric in ["acc_norm", "acc"]:
        prediction_key = f"prediction_{metric}"
        correct_key = f"correct_{metric}"
        recomputed = []
        for row in rows:
            values = (row["choice_loglikelihood_per_token"] if metric == "acc_norm"
                      else row["choice_loglikelihood"])
            prediction = max(range(4), key=lambda index: (values[index], -index))
            if prediction != row[prediction_key] or (prediction == row["gold"]) != row[correct_key]:
                raise ValueError("prediction binding mismatch")
            recomputed.append(row[correct_key])
        metrics = report["metrics"][metric]
        if sum(recomputed) != metrics["correct"] or sum(recomputed) / len(recomputed) != metrics["score"]:
            raise ValueError("metric arithmetic mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    base = json.loads(args.base.read_text())
    reports = [json.loads(path.read_text()) for path in args.result]
    for report in [base] + reports:
        validate_rows(report)
    expected_ids = [item["id"] for item in plan["candidates"]]
    found_ids = [item["candidate"]["id"] for item in reports]
    if found_ids != expected_ids or len(set(found_ids)) != len(found_ids):
        raise ValueError("candidate order or identity mismatch")
    base_bindings = [(row["ordinal"], row["row_sha256"], row["gold"]) for row in base["results"]]
    candidates = []
    for report, path in zip(reports, args.result):
        bindings = [(row["ordinal"], row["row_sha256"], row["gold"]) for row in report["results"]]
        if bindings != base_bindings:
            raise ValueError("row/choice/gold pairing differs from Base")
        metrics = {}
        for metric in ["acc_norm", "acc"]:
            base_correct = [row[f"correct_{metric}"] for row in base["results"]]
            candidate_correct = [row[f"correct_{metric}"] for row in report["results"]]
            score = report["metrics"][metric]["score"]
            metrics[metric] = {
                "score": score,
                "correct": report["metrics"][metric]["correct"],
                "delta_vs_our_base": score - base["metrics"][metric]["score"],
                "paired_bootstrap_delta_ci95": paired_interval(
                    base_correct, candidate_correct, plan["bootstrap"]["seed"],
                    plan["bootstrap"]["resamples"]),
            }
        candidates.append({
            "id": report["candidate"]["id"],
            "repo_id": report["candidate"]["repo_id"],
            "revision": report["candidate"]["revision"],
            "parameters": report["parameters"],
            "model_class": report["model_class"],
            "metrics": metrics,
            "result_sha256": digest(path),
        })
    ranking = sorted(
        [{"id": "our_base", "acc_norm": base["metrics"]["acc_norm"]["score"]}] +
        [{"id": item["id"], "acc_norm": item["metrics"]["acc_norm"]["score"]}
         for item in candidates], key=lambda item: (-item["acc_norm"], item["id"]))
    output = {
        "schema": "p529m-matched-external-closed-book-sciq-summary-v1",
        "status": "MATCHED_EXTERNAL_CLOSED_BOOK_SCIQ_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": {
            "samples": 489,
            "dataset": "allenai/sciq test",
            "prompt": "Question: {question}\\nAnswer:",
            "support_in_prompt": False,
            "completion": " {choice_text}",
            "primary_metric": "mean answer-token conditional log likelihood accuracy (acc_norm)",
            "secondary_metric": "total answer-token conditional log likelihood accuracy (acc)",
            "choice_order": "identical frozen order for every model",
            "dtype": plan["execution"]["dtype"],
        },
        "our_base": {
            "acc_norm": base["metrics"]["acc_norm"]["score"],
            "acc": base["metrics"]["acc"]["score"],
            "result_sha256": digest(args.base),
        },
        "candidates": candidates,
        "ranking_by_acc_norm": ranking,
        "verification": {
            "all_489_row_choice_gold_bindings_match_our_base": True,
            "predictions_and_metrics_recomputed": True,
            "paired_bootstrap_resamples": plan["bootstrap"]["resamples"],
        },
        "claim_boundary": plan["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
