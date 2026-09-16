#!/usr/bin/env python3
"""Complete the two-seed F2 seven-task summary after a missing-pair rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TASKS = (
    "hellaswag",
    "piqa",
    "winogrande",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "boolq",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def validate(document: dict, protocol_sha256: str) -> None:
    if document.get("status") != "PASS_FROZEN_EVALUATION_AGGREGATE":
        raise ValueError("aggregate status is not passing")
    if document.get("protocol_sha256") != protocol_sha256:
        raise ValueError("aggregate protocol hash mismatch")
    if set(document.get("tasks", {})) != set(TASKS):
        raise ValueError("aggregate task set mismatch")


def scores(document: dict) -> dict[str, float]:
    return {task: float(document["tasks"][task]["score"]) for task in TASKS}


def reproduction_comparison(current: dict, reference: dict) -> dict:
    task_equal = {}
    for task in TASKS:
        left, right = current["tasks"][task], reference["tasks"][task]
        task_equal[task] = {
            "score_equal": left["score"] == right["score"],
            "samples_equal": left["samples"] == right["samples"],
            "primary_metric_equal": left["primary_metric"] == right["primary_metric"],
        }
    aggregate_equal = {
        key: current["aggregate"][key] == reference["aggregate"][key]
        for key in ("score", "bootstrap_ci_95", "bootstrap_replicates", "bootstrap_seed", "percentile_method", "task_weighting")
    }
    passed = all(aggregate_equal.values()) and all(
        all(values.values()) for values in task_equal.values()
    )
    return {"passed": passed, "aggregate_equal": aggregate_equal, "task_equal": task_equal}


def atomic_json(path: Path, value: dict) -> None:
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
    parser.add_argument("--new-parent", type=Path, required=True)
    parser.add_argument("--new-continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load(args.plan)
    if plan.get("schema") != "p529m-seven-task-f2-continuation-completion-plan-v1":
        raise ValueError("unexpected plan schema")
    if plan.get("status") != "FROZEN_BEFORE_MISSING_PAIR_EVALUATION":
        raise ValueError("plan is not frozen")
    protocol_sha = plan["protocol"]["sha256"]
    references = {}
    for name, item in plan["references"].items():
        path = args.plan.parent / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"reference hash mismatch: {name}")
        references[name] = load(path)
        validate(references[name], protocol_sha)
    parent = load(args.new_parent)
    continuation = load(args.new_continuation)
    validate(parent, protocol_sha)
    validate(continuation, protocol_sha)
    reproduction = reproduction_comparison(parent, references["parent_seed20260915"])

    output = {
        "schema": "p529m-seven-task-f2-continuation-completion-result-v1",
        "status": "PARENT_REPRODUCTION_FAILED_NO_CROSS_JOB_COMBINATION",
        "plan_sha256": digest(args.plan),
        "parent_reproduction": reproduction,
        "formal_promotion": False,
        "claim_boundary": plan["claim_boundary"],
        "input_sha256": {
            "new_parent": digest(args.new_parent),
            "new_continuation": digest(args.new_continuation),
            **{f"reference_{name}": item["sha256"] for name, item in plan["references"].items()},
        },
    }
    if reproduction["passed"]:
        rows = {
            20260914: {
                "parent": references["parent_seed20260914"],
                "continuation": references["continuation_seed20260914"],
            },
            20260915: {"parent": parent, "continuation": continuation},
        }
        pairs = []
        for seed, pair in rows.items():
            parent_score = float(pair["parent"]["aggregate"]["score"])
            continuation_score = float(pair["continuation"]["aggregate"]["score"])
            parent_tasks = scores(pair["parent"])
            continuation_tasks = scores(pair["continuation"])
            deltas = {task: continuation_tasks[task] - parent_tasks[task] for task in TASKS}
            pairs.append({
                "seed": seed,
                "parent_aggregate_score": parent_score,
                "continuation_aggregate_score": continuation_score,
                "aggregate_delta_new_vs_parent": continuation_score - parent_score,
                "task_delta_new_vs_parent": deltas,
                "strict_aggregate_gain": continuation_score > parent_score,
                "no_task_decline_over_two_percentage_points": min(deltas.values()) >= -0.02,
            })
        mean_parent = sum(row["parent_aggregate_score"] for row in pairs) / 2
        mean_continuation = sum(row["continuation_aggregate_score"] for row in pairs) / 2
        parent_task_means = {
            task: sum(float(rows[seed]["parent"]["tasks"][task]["score"]) for seed in rows) / 2
            for task in TASKS
        }
        continuation_task_means = {
            task: sum(float(rows[seed]["continuation"]["tasks"][task]["score"]) for seed in rows) / 2
            for task in TASKS
        }
        output.update({
            "status": "COMPLETE_ADAPTIVE_TWO_SEED_F2_CONTINUATION_SEVEN_TASK_COMPARISON",
            "pairs": pairs,
            "two_seed": {
                "parent_mean_aggregate": mean_parent,
                "continuation_mean_aggregate": mean_continuation,
                "aggregate_delta_new_vs_parent": mean_continuation - mean_parent,
                "parent_mean_tasks": parent_task_means,
                "continuation_mean_tasks": continuation_task_means,
                "task_delta_new_vs_parent": {
                    task: continuation_task_means[task] - parent_task_means[task]
                    for task in TASKS
                },
            },
            "progression_gate": {
                "both_seeds_strict_aggregate_gain": all(row["strict_aggregate_gain"] for row in pairs),
                "both_seeds_no_task_decline_over_two_percentage_points": all(row["no_task_decline_over_two_percentage_points"] for row in pairs),
                "two_seed_mean_strictly_over_0_50": mean_continuation > 0.50,
            },
        })
    atomic_json(args.output, output)
    print(json.dumps(output, sort_keys=True))
    if not reproduction["passed"]:
        raise SystemExit("parent reproduction gate failed")


if __name__ == "__main__":
    main()
