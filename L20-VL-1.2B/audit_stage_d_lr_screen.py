#!/usr/bin/env python3
"""Fail-closed audit of the Stage-D answer-balanced LR micro-search."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    summary = json.loads(args.summary.read_text())
    protocol_sha = sha256_file(args.protocol)
    if protocol.get("status") != "authorized_stage_d_lr_microsearch":
        raise RuntimeError("unexpected Stage-D LR protocol status")
    if summary.get("status") != "complete" or summary.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("LR summary is incomplete or belongs to another protocol")
    expected = {
        (arm, float(multiplier))
        for arm in protocol["arms"]
        for multiplier in protocol["allowed_learning_rate_multipliers"]
    }
    observed = {
        (item["arm"], float(item["learning_rate_multiplier"]))
        for item in summary["results"]
    }
    if observed != expected or len(summary["results"]) != len(expected):
        raise RuntimeError("LR summary does not contain exactly the frozen trial matrix")

    trial_audits = []
    total_training_seconds = 0.0
    total_evaluation_seconds = 0.0
    for item in summary["results"]:
        trial = item["trial"]
        run_dir = ROOT / "runs" / f"stage-d-lr-screen-{trial}-v1"
        run = json.loads((run_dir / "run.json").read_text())
        if (
            run.get("status") != "complete"
            or run.get("optimizer_step") != protocol["screen_optimizer_steps"]
            or run.get("protocol_sha256") != protocol_sha
            or run.get("teacher_cache_sha256") is not None
        ):
            raise RuntimeError(f"invalid training receipt for {trial}")
        evaluation_path = Path(item["evaluation"])
        if sha256_file(evaluation_path) != item["evaluation_sha256"]:
            raise RuntimeError(f"evaluation hash mismatch for {trial}")
        evaluation = json.loads(evaluation_path.read_text())
        if (
            evaluation.get("split") != "development"
            or evaluation.get("scene_families") != 250
            or evaluation.get("conditions") != ["true_image"]
            or evaluation.get("selection", {}).get("stratified_by") != ["task", "base_answer"]
        ):
            raise RuntimeError(f"invalid development-only evaluation scope for {trial}")
        strata = Counter(
            (row["task"], row["expected"]["base"])
            for row in evaluation["predictions"]
        )
        tasks = set(protocol["development"].get("tasks", ())) or {
            row["task"] for row in evaluation["predictions"]
        }
        expected_strata = {
            (task, answer): protocol["development"]["families_per_task_answer_stratum"]
            for task in tasks
            for answer in ("no", "yes")
        }
        if dict(strata) != expected_strata:
            raise RuntimeError(f"development selection is not exactly balanced for {trial}")
        selection = ROOT / "evidence" / f"stage-d-lr-screen-{trial}-selection-v1.json"
        prune = ROOT / "evidence" / f"stage-d-lr-screen-{trial}-prune-v1.json"
        if not selection.exists() or not prune.exists():
            raise RuntimeError(f"selection/prune receipt missing for {trial}")
        prune_receipt = json.loads(prune.read_text())
        if (
            prune_receipt.get("status") != "complete"
            or prune_receipt.get("expected_optimizer_steps") != protocol["screen_optimizer_steps"]
        ):
            raise RuntimeError(f"invalid prune receipt for {trial}")
        training_seconds = float(run.get("total_wall_seconds", run["wall_seconds"]))
        evaluation_seconds = float(evaluation["total_wall_seconds"])
        total_training_seconds += training_seconds
        total_evaluation_seconds += evaluation_seconds
        trial_audits.append({
            "trial": trial,
            "training_total_wall_seconds": training_seconds,
            "evaluation_total_wall_seconds": evaluation_seconds,
            "development_task_answer_strata": {
                f"{task}|{answer}": count
                for (task, answer), count in sorted(strata.items())
            },
            "run_json_sha256": sha256_file(run_dir / "run.json"),
            "selection_sha256": sha256_file(selection),
            "prune_sha256": sha256_file(prune),
        })

    recomputed = {}
    for arm in protocol["arms"]:
        candidates = [item for item in summary["results"] if item["arm"] == arm]
        best = max(candidates, key=lambda item: (
            item["family_joint_accuracy_percent"],
            item["ordinary_primary_query_accuracy_percent"],
            -abs(item["learning_rate_multiplier"] - 1.0),
            -item["learning_rate_multiplier"],
        ))
        recomputed[arm] = best["learning_rate_multiplier"]
    if recomputed != summary["selected_learning_rate_multipliers"]:
        raise RuntimeError("development-selected LR multipliers do not match the frozen rule")
    if any(summary.get(key) != 0 for key in (
        "prior_test_predictions_produced",
        "new_iid_test_predictions_produced",
        "ood_test_predictions_produced",
    )):
        raise RuntimeError("test predictions were produced during LR selection")
    total_seconds = total_training_seconds + total_evaluation_seconds
    if total_seconds / 3600.0 > protocol["budget"]["lr_screen_training_gpu_hours_cap"]:
        raise RuntimeError("LR screen exceeded its declared GPU-hour cap")
    receipt = {
        "schema_version": "2026-09-13-v1",
        "status": "pass",
        "completed_at": utc_now(),
        "protocol_sha256": protocol_sha,
        "summary_sha256": sha256_file(args.summary),
        "trial_audits": trial_audits,
        "selected_learning_rate_multipliers": recomputed,
        "training_gpu_hours": total_training_seconds / 3600.0,
        "development_evaluation_gpu_hours": total_evaluation_seconds / 3600.0,
        "total_gpu_hours": total_seconds / 3600.0,
        "development_families_per_trial": 250,
        "development_selection_stratified_by": ["task", "base_answer"],
        "prior_test_predictions_produced": 0,
        "new_iid_test_predictions_produced": 0,
        "ood_test_predictions_produced": 0,
        "next_gate": "freeze a new two-arm five-seed D1 protocol before any additional model update",
        "claim_boundary": "This audit validates LR selection mechanics and cost only. The one-seed micro-search is not an efficacy result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
