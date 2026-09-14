#!/usr/bin/env python3
"""Evaluate frozen counterfactual predictions with paired clustered intervals.

Expected JSONL schema (one row per scene family):
{
  "scene_family_id": "cf-count-yes-0001",
  "task": "count",
  "split": "test",
  "methods": {
    "full_196": {"base_correct": true, "edited_correct": true,
                 "invariant_correct": true},
    "strong_49": {"base_correct": true, "edited_correct": false,
                  "invariant_correct": true},
    "proposed_49": {"base_correct": true, "edited_correct": true,
                    "invariant_correct": true}
  }
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from counterfactual_losses import paired_cluster_bootstrap


REQUIRED_VARIANTS = ("base_correct", "edited_correct", "invariant_correct")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _validate_rows(
    rows: Iterable[dict[str, Any]], methods: tuple[str, str, str]
) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        family = str(row.get("scene_family_id", ""))
        if not family or family in seen:
            raise ValueError(f"line {line_number}: missing or duplicate scene_family_id")
        seen.add(family)
        if not row.get("task"):
            raise ValueError(f"line {line_number}: task is required")
        available = row.get("methods")
        if not isinstance(available, dict):
            raise ValueError(f"line {line_number}: methods must be an object")
        for method in methods:
            prediction = available.get(method)
            if not isinstance(prediction, dict):
                raise ValueError(f"line {line_number}: method {method!r} is missing")
            for variant in REQUIRED_VARIANTS:
                if type(prediction.get(variant)) is not bool:
                    raise ValueError(
                        f"line {line_number}: {method}.{variant} must be boolean"
                    )
        checked.append(row)
    if not checked:
        raise ValueError("prediction file is empty")
    return checked


def _joint(method: dict[str, bool], edited_key: str = "edited_correct") -> float:
    return float(method["base_correct"] and method[edited_key])


def _interval(values: list[float], families: list[str], resamples: int) -> dict[str, Any]:
    interval = paired_cluster_bootstrap(values, families, resamples=resamples)
    return {
        "estimate_pp": 100.0 * interval.estimate,
        "lower_95_ci_pp": 100.0 * interval.lower,
        "upper_95_ci_pp": 100.0 * interval.upper,
        "clusters": interval.clusters,
        "samples": interval.samples,
        "resamples": interval.resamples,
    }


def evaluate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    full: str,
    resamples: int = 10_000,
) -> dict[str, Any]:
    checked = _validate_rows(rows, (candidate, baseline, full))

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        families = [str(row["scene_family_id"]) for row in subset]
        candidate_joint = [_joint(row["methods"][candidate]) for row in subset]
        baseline_joint = [_joint(row["methods"][baseline]) for row in subset]
        joint_differences = [a - b for a, b in zip(candidate_joint, baseline_joint)]
        candidate_invariant = [
            _joint(row["methods"][candidate], "invariant_correct") for row in subset
        ]
        baseline_invariant = [
            _joint(row["methods"][baseline], "invariant_correct") for row in subset
        ]
        eligible_indices = [
            index
            for index, row in enumerate(subset)
            if _joint(row["methods"][full]) == 1.0
        ]
        failure_reduction = [
            (1.0 - baseline_joint[index]) - (1.0 - candidate_joint[index])
            for index in eligible_indices
        ]
        eligible_families = [families[index] for index in eligible_indices]
        failure_result: dict[str, Any] = {
            "eligible_full_token_pairs": len(eligible_indices),
            "baseline_failure_percent": 100.0
            * _mean([1.0 - baseline_joint[index] for index in eligible_indices]),
            "candidate_failure_percent": 100.0
            * _mean([1.0 - candidate_joint[index] for index in eligible_indices]),
        }
        if failure_reduction:
            failure_result["baseline_minus_candidate_failure"] = _interval(
                failure_reduction, eligible_families, resamples
            )
        else:
            failure_result["baseline_minus_candidate_failure"] = None
        return {
            "scene_families": len(subset),
            "paired_joint_accuracy_percent": {
                candidate: 100.0 * _mean(candidate_joint),
                baseline: 100.0 * _mean(baseline_joint),
                "candidate_minus_baseline": _interval(
                    joint_differences, families, resamples
                ),
            },
            "base_plus_invariant_joint_accuracy_percent": {
                candidate: 100.0 * _mean(candidate_invariant),
                baseline: 100.0 * _mean(baseline_invariant),
            },
            "compression_induced_failure": failure_result,
        }

    by_task = {
        task: summarize([row for row in checked if row["task"] == task])
        for task in sorted({str(row["task"]) for row in checked})
    }
    overall = summarize(checked)
    lower = overall["paired_joint_accuracy_percent"]["candidate_minus_baseline"][
        "lower_95_ci_pp"
    ]
    return {
        "schema_version": "2026-09-13-v1",
        "candidate": candidate,
        "baseline": baseline,
        "full_token_reference": full,
        "primary_metric": "base_and_answer_changing_edit_joint_accuracy",
        "primary_confirmatory_rule": {
            "lower_95_ci_pp_gt": 0.0,
            "practical_effect_estimate_pp_gte": 2.0,
            "passes_statistical_rule": lower > 0.0,
            "passes_both_rules": lower > 0.0
            and overall["paired_joint_accuracy_percent"]["candidate_minus_baseline"][
                "estimate_pp"
            ]
            >= 2.0,
        },
        "overall": overall,
        "by_task": by_task,
        "claim_boundary": "This receipt evaluates supplied predictions only; it does not establish data validity, real-image transfer, or multi-seed replication.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--candidate", default="proposed_49")
    parser.add_argument("--baseline", default="strong_49")
    parser.add_argument("--full", default="full_196")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    result = evaluate_rows(
        rows,
        candidate=args.candidate,
        baseline=args.baseline,
        full=args.full,
        resamples=args.resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["primary_confirmatory_rule"], sort_keys=True))


if __name__ == "__main__":
    main()
