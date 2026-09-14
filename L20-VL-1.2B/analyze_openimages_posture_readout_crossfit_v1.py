#!/usr/bin/env python3
"""Audit and summarize the frozen Open Images posture readout cross-fit run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
VIEWS = ("full", "marked", "crop")


def pair_joint_map(records: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_id"]].append(record)
    output = {}
    for pair_id, rows in grouped.items():
        if len(rows) != 2:
            raise ValueError(f"incomplete pair: {pair_id}")
        output[pair_id] = 100.0 * float(all(row["correct"] for row in rows))
    return output


def paired_interval(left: dict[str, float], right: dict[str, float], protocol: dict[str, Any]):
    from evaluate_openimages_posture_zero_shot_v1 import interval

    if left.keys() != right.keys():
        raise ValueError("paired maps have different keys")
    keys = sorted(left)
    return interval([left[key] - right[key] for key in keys], keys, protocol)


def address_map(records: list[dict[str, Any]], field: str, scale: float) -> dict[str, float]:
    output = {}
    for record in records:
        address = record.get("address")
        if address is None:
            raise ValueError("address record is missing")
        output[record["target_id"]] = scale * float(address[field])
    return output


def target_interval(left, right, field, scale, protocol):
    from evaluate_openimages_posture_zero_shot_v1 import interval

    left_map = address_map(left, field, scale)
    right_map = address_map(right, field, scale)
    if left_map.keys() != right_map.keys():
        raise ValueError("address target sets differ")
    by_target = {row["target_id"]: row["pair_id"] for row in left}
    keys = sorted(left_map)
    return interval(
        [left_map[key] - right_map[key] for key in keys],
        [by_target[key] for key in keys],
        protocol,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    protocol = json.loads(args.protocol.read_text())
    run = json.loads(args.run.read_text())
    if run.get("status") != "complete_crossfit_development_only":
        raise RuntimeError("cross-fit run is not complete")
    if run["protocol"]["sha256"] != sha256_file(args.protocol):
        raise RuntimeError("run/protocol hash mismatch")
    if len(run.get("fold_receipts", [])) != 10:
        raise RuntimeError("expected two arms times five folds")

    expected_pairs = {
        row["pair_id"]
        for line in Path(protocol["data"]["manifest"]["path"]).read_text().splitlines()
        if line
        for row in [json.loads(line)]
    }
    fold_audit = []
    for receipt in run["fold_receipts"]:
        train_ids = set(receipt["train_pair_ids"])
        test_ids = set(receipt["test_pair_ids"])
        if train_ids & test_ids or train_ids | test_ids != expected_pairs:
            raise RuntimeError("cross-fit receipt contains leakage or missing pairs")
        fold_audit.append(
            {
                "arm": receipt["arm"],
                "fold": receipt["fold"],
                "train_pairs": len(train_ids),
                "test_pairs": len(test_ids),
                "optimizer_steps": receipt["optimizer_steps"],
                "trainable_parameters": receipt["trainable_parameters"],
            }
        )
    for arm in ("query_ce", "grounded_query"):
        tested = [
            pair_id
            for receipt in run["fold_receipts"]
            if receipt["arm"] == arm
            for pair_id in receipt["test_pair_ids"]
        ]
        if len(tested) != len(expected_pairs) or set(tested) != expected_pairs:
            raise RuntimeError(f"each pair must be tested exactly once for {arm}")

    parent = json.loads(Path(protocol["data"]["zero_shot_receipt"]["path"]).read_text())
    pair_joint_comparisons = {}
    for left_name, right_name in (
        ("grounded_query", "query_ce"),
        ("query_ce", "stage_d_parent"),
        ("grounded_query", "stage_d_parent"),
    ):
        pair_joint_comparisons[f"{left_name}_minus_{right_name}"] = {}
        for view in VIEWS:
            left_records = (
                run["arms"][left_name]["records"][view]["true_image"]
                if left_name != "stage_d_parent"
                else parent["arms"]["stage_d_parent"]["records"][view]["true_image"]
            )
            right_records = (
                run["arms"][right_name]["records"][view]["true_image"]
                if right_name != "stage_d_parent"
                else parent["arms"]["stage_d_parent"]["records"][view]["true_image"]
            )
            pair_joint_comparisons[f"{left_name}_minus_{right_name}"][view] = paired_interval(
                pair_joint_map(left_records), pair_joint_map(right_records), protocol
            )

    address_effect = {}
    for view in ("full", "marked"):
        grounded = run["arms"]["grounded_query"]["records"][view]["true_image"]
        control = run["arms"]["query_ce"]["records"][view]["true_image"]
        address_effect[view] = {
            "bbox_region_top1_difference_pp": target_interval(
                grounded, control, "bbox_region_top1", 100.0, protocol
            ),
            "bbox_region_attention_mass_difference_pp": target_interval(
                grounded, control, "bbox_region_attention_mass", 100.0, protocol
            ),
        }

    compact_arms = {}
    for arm_name, arm in run["arms"].items():
        compact_arms[arm_name] = {}
        for view in VIEWS:
            true_image = arm["metrics"][view]["conditions"]["true_image"]
            compact_arms[arm_name][view] = {
                "per_image_accuracy_percent": true_image["per_image_accuracy_percent"],
                "pair_joint_accuracy_percent": true_image["pair_joint_accuracy_percent"],
                "true_minus_paired_swap_accuracy_pp": arm["metrics"][view][
                    "true_minus_paired_swap_accuracy_pp"
                ],
                "strict_bidirectional_answer_flip_percent": arm["metrics"][view][
                    "strict_bidirectional_answer_flip_percent"
                ],
                "address": true_image["address"],
            }

    result = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_audited_method_development_only",
        "completed_at": utc_now(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "run": {"path": str(args.run.resolve()), "sha256": sha256_file(args.run)},
        "analyzer_sha256": sha256_file(Path(__file__)),
        "test_sha256": sha256_file(ROOT / "test_analyze_openimages_posture_readout_crossfit_v1.py"),
        "crossfit_audit": {
            "passed": True,
            "pairs": len(expected_pairs),
            "fold_receipts": fold_audit,
            "each_pair_scored_once_per_arm": True,
            "pair_leakage_detected": False,
        },
        "execution": {
            "device": run["device"],
            "wall_seconds": run["wall_seconds"],
            "peak_allocated_gib": run["peak_allocated_gib"],
            "optimizer_steps": sum(row["optimizer_steps"] for row in fold_audit),
            "trainable_parameters": sorted({row["trainable_parameters"] for row in fold_audit}),
        },
        "arms": compact_arms,
        "per_image_accuracy_comparisons": run["comparisons"],
        "pair_joint_accuracy_comparisons": pair_joint_comparisons,
        "grounded_attention_effect": address_effect,
        "decision": {
            "selected_for_new_sealed_confirmation": "query_ce",
            "reason": "The equal-capacity query-CE readout produced the highest answer accuracy. Region supervision strongly improved localization attention but did not improve answer accuracy on the frozen method-development set.",
            "do_not_select": "grounded_query",
            "next_required_evidence": "A newly acquired, disjoint, sealed real-image confirmation set with unambiguous targets, frozen prompts, no hyperparameter changes, and paired cluster confidence intervals.",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
