#!/usr/bin/env python3
"""Validate and summarize one F2 fresh-N1 seven-task retention evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_aggregate(document: dict, protocol_sha256: str) -> None:
    if document.get("status") != "PASS_FROZEN_EVALUATION_AGGREGATE":
        raise ValueError("aggregate did not pass the frozen evaluator")
    if document.get("protocol_sha256") != protocol_sha256:
        raise ValueError("aggregate protocol hash differs from the frozen plan")
    if set(document.get("tasks", {})) != set(TASKS):
        raise ValueError("aggregate task set differs from the frozen seven tasks")
    for task in TASKS:
        score = float(document["tasks"][task]["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score for {task}: {score}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = read_json(args.plan)
    if plan.get("schema") != "p529m-f2-fresh-n1-seven-task-eval-plan-v1":
        raise ValueError("unexpected plan schema")
    if plan.get("status") != "FROZEN_TWO_COMPLETED_F2_FRESH_N1_ARMS_BEFORE_EVALUATION":
        raise ValueError("plan was not frozen before evaluation")
    candidates = {item["id"]: item for item in plan.get("candidates", [])}
    if len(candidates) != 2 or args.candidate_id not in candidates:
        raise ValueError("candidate set or requested candidate differs from plan")
    candidate = candidates[args.candidate_id]

    if digest(args.base) != plan["base"]["aggregate_sha256"]:
        raise ValueError("matched two-GPU Base aggregate hash mismatch")
    base = read_json(args.base)
    aggregate = read_json(args.aggregate)
    validate_aggregate(base, plan["protocol_sha256"])
    validate_aggregate(aggregate, plan["protocol_sha256"])

    receipt = read_json(args.export_receipt)
    if receipt.get("status") != "PASS_HF_EXPORT_BITWISE_STATE_AND_BOUNDED_BF16_LOGIT_DRIFT":
        raise ValueError("candidate export parity did not pass")
    if receipt.get("checkpoint_sha256") != candidate["checkpoint_sha256"]:
        raise ValueError("candidate export checkpoint hash mismatch")
    if int(receipt.get("checkpoint_step", -1)) != int(candidate["expected_step"]):
        raise ValueError("candidate export checkpoint step mismatch")

    base_score = float(base["aggregate"]["score"])
    candidate_score = float(aggregate["aggregate"]["score"])
    base_tasks = {task: float(base["tasks"][task]["score"]) for task in TASKS}
    candidate_tasks = {task: float(aggregate["tasks"][task]["score"]) for task in TASKS}
    result = {
        "schema": "p529m-f2-fresh-n1-seven-task-eval-result-v1",
        "status": "PASS_ADAPTIVE_F2_FRESH_N1_SEVEN_TASK_SCREEN",
        "candidate": candidate,
        "protocol_id": plan["protocol_id"],
        "protocol_sha256": plan["protocol_sha256"],
        "parent_aggregate_score": base_score,
        "candidate_aggregate_score": candidate_score,
        "aggregate_delta_vs_parent": candidate_score - base_score,
        "parent_tasks": base_tasks,
        "candidate_tasks": candidate_tasks,
        "task_delta_vs_parent": {
            task: candidate_tasks[task] - base_tasks[task] for task in TASKS
        },
        "input_sha256": {
            "plan": digest(args.plan),
            "parent_aggregate": digest(args.base),
            "candidate_aggregate": digest(args.aggregate),
            "export_receipt": digest(args.export_receipt),
        },
        "formal_promotion": False,
        "claim_boundary": plan["claim_boundary"],
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
