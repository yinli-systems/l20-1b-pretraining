#!/usr/bin/env python3
"""Bind and summarize matched two-GPU seven-task evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
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
        raise ValueError("aggregate did not pass its frozen evaluator")
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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference-base", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = read_json(args.plan)
    if plan.get("schema") != "p529m-seven-task-confirmation-2gpu-plan-v1":
        raise ValueError("unexpected plan schema")
    if plan.get("status") != "FROZEN_BEFORE_MATCHED_TWO_GPU_EVALUATION":
        raise ValueError("plan was not frozen before evaluation")
    if plan.get("execution", {}).get("world_size") != 2:
        raise ValueError("plan is not the frozen two-GPU execution")
    if len(plan.get("candidates", [])) != 4:
        raise ValueError("the frozen plan must contain exactly four candidates")
    ids = [item["id"] for item in plan["candidates"]]
    if len(set(ids)) != 4:
        raise ValueError("candidate IDs must be unique")

    base = read_json(args.base)
    validate_aggregate(base, plan["protocol_sha256"])
    base_receipt_path = args.results_root / "base" / "export-receipt.json"
    base_receipt = read_json(base_receipt_path)
    if digest(base_receipt_path) != plan["base"]["hf_export_receipt_sha256"]:
        raise ValueError("matched Base export receipt hash mismatch")
    if base_receipt.get("checkpoint_sha256") != plan["base"]["checkpoint_sha256"]:
        raise ValueError("matched Base checkpoint hash mismatch")
    if int(base_receipt.get("checkpoint_step", -1)) != int(plan["base"]["expected_step"]):
        raise ValueError("matched Base checkpoint step mismatch")
    base_score = float(base["aggregate"]["score"])
    base_tasks = {task: float(base["tasks"][task]["score"]) for task in TASKS}

    if digest(args.reference_base) != plan["reference_base_aggregate_sha256"]:
        raise ValueError("reference four-GPU Base aggregate hash mismatch")
    reference_base = read_json(args.reference_base)
    validate_aggregate(reference_base, plan["protocol_sha256"])
    reference_base_score = float(reference_base["aggregate"]["score"])
    reference_base_tasks = {
        task: float(reference_base["tasks"][task]["score"]) for task in TASKS
    }

    candidates = []
    by_recipe: dict[str, list[dict]] = defaultdict(list)
    input_hashes = {
        "plan": digest(args.plan),
        "matched_two_gpu_base_aggregate": digest(args.base),
        "matched_two_gpu_base_export_receipt": digest(base_receipt_path),
        "reference_four_gpu_base_aggregate": digest(args.reference_base),
    }
    for item in plan["candidates"]:
        candidate_root = args.results_root / item["id"]
        receipt_path = candidate_root / "export-receipt.json"
        aggregate_path = candidate_root / "seven-task-aggregate.json"
        receipt = read_json(receipt_path)
        aggregate = read_json(aggregate_path)
        if receipt.get("status") != "PASS_HF_EXPORT_BITWISE_STATE_AND_BOUNDED_BF16_LOGIT_DRIFT":
            raise ValueError(f"{item['id']} export parity did not pass")
        if receipt.get("checkpoint_sha256") != item["checkpoint_sha256"]:
            raise ValueError(f"{item['id']} export checkpoint hash mismatch")
        if int(receipt.get("checkpoint_step", -1)) != int(item["expected_step"]):
            raise ValueError(f"{item['id']} export checkpoint step mismatch")
        validate_aggregate(aggregate, plan["protocol_sha256"])
        score = float(aggregate["aggregate"]["score"])
        tasks = {task: float(aggregate["tasks"][task]["score"]) for task in TASKS}
        row = {
            "id": item["id"],
            "recipe": item["recipe"],
            "seed": item["seed"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "aggregate_score": score,
            "aggregate_delta_vs_base": score - base_score,
            "tasks": tasks,
            "task_delta_vs_base": {task: tasks[task] - base_tasks[task] for task in TASKS},
            "aggregate_sha256": digest(aggregate_path),
            "export_receipt_sha256": digest(receipt_path),
        }
        candidates.append(row)
        by_recipe[item["recipe"]].append(row)
        input_hashes[f"{item['id']}_aggregate"] = row["aggregate_sha256"]
        input_hashes[f"{item['id']}_export_receipt"] = row["export_receipt_sha256"]

    recipes = []
    for recipe, rows in sorted(by_recipe.items()):
        if len(rows) != 2 or len({row["seed"] for row in rows}) != 2:
            raise ValueError(f"{recipe} does not have two distinct seeds")
        mean_score = sum(row["aggregate_score"] for row in rows) / 2
        mean_tasks = {
            task: sum(row["tasks"][task] for row in rows) / 2 for task in TASKS
        }
        recipes.append(
            {
                "recipe": recipe,
                "mean_two_seed_aggregate": mean_score,
                "worst_seed_aggregate": min(row["aggregate_score"] for row in rows),
                "seed_spread": max(row["aggregate_score"] for row in rows)
                - min(row["aggregate_score"] for row in rows),
                "aggregate_delta_vs_base": mean_score - base_score,
                "mean_tasks": mean_tasks,
                "task_delta_vs_base": {
                    task: mean_tasks[task] - base_tasks[task] for task in TASKS
                },
            }
        )
    recipes.sort(key=lambda row: (-row["worst_seed_aggregate"], -row["mean_two_seed_aggregate"], row["recipe"]))
    for rank, row in enumerate(recipes, 1):
        row["rank"] = rank

    output = {
        "schema": "p529m-seven-task-confirmation-2gpu-summary-v1",
        "status": "PASS_MATCHED_TWO_GPU_BASE_AND_FOUR_CANDIDATE_SEVEN_TASK_EVALUATION",
        "protocol_id": plan["protocol_id"],
        "protocol_sha256": plan["protocol_sha256"],
        "selection_rule": "higher worst-seed seven-task unweighted mean, then higher two-seed mean, then recipe id",
        "base": {"aggregate_score": base_score, "tasks": base_tasks},
        "reference_four_gpu_base": {
            "aggregate_score": reference_base_score,
            "tasks": reference_base_tasks,
        },
        "two_gpu_base_delta_vs_reference": {
            "aggregate": base_score - reference_base_score,
            "tasks": {
                task: base_tasks[task] - reference_base_tasks[task] for task in TASKS
            },
        },
        "candidates": candidates,
        "recipes": recipes,
        "selected_recipe_by_seven_task_accuracy": recipes[0]["recipe"],
        "formal_promotion": False,
        "claim_boundary": "matched two-GPU English base-model multiple-choice accuracy and retention only; instruction following, multilingual generation, safety and market superiority remain unverified",
        "input_sha256": input_hashes,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
