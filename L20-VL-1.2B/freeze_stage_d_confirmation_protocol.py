#!/usr/bin/env python3
"""Freeze the one-time Stage-D IID/OOD confirmation before test inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from stage_d_confirmation_contract import (
    ARMS,
    CONDITIONS,
    SPLITS,
    confirmation_execution_order,
    sha256_file,
    stable_id_sha256,
)
from train_stage_a_full_token import utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replication-protocol", type=Path, required=True)
    parser.add_argument("--replication-summary", type=Path, required=True)
    parser.add_argument("--data-protocol", type=Path, required=True)
    parser.add_argument("--data-admission", type=Path, required=True)
    parser.add_argument("--human-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if list((ROOT / "evidence").glob("stage-d-confirmation-*-seed*-*-v1.json")):
        raise RuntimeError("confirmation predictions already exist; refusing to freeze post hoc")

    replication_protocol_path = args.replication_protocol.resolve()
    replication_summary_path = args.replication_summary.resolve()
    replication_protocol = json.loads(replication_protocol_path.read_text())
    replication_summary = json.loads(replication_summary_path.read_text())
    if replication_summary.get("status") != "complete_checkpoints_frozen_tests_still_sealed":
        raise RuntimeError("D1 did not finish with tests sealed")
    for key in (
        "prior_test_predictions_produced",
        "new_iid_test_predictions_produced",
        "ood_test_predictions_produced",
    ):
        if replication_summary.get(key) != 0:
            raise RuntimeError(f"D1 summary reports prior test use: {key}")
    seeds = list(replication_protocol["allowed_seeds"])
    if len(seeds) != 5 or set(replication_protocol["arms"]) != set(ARMS):
        raise RuntimeError("confirmation requires the frozen two-arm, five-seed D1 matrix")
    if len(replication_summary["results"]) != 10:
        raise RuntimeError("D1 summary must contain exactly ten completed trials")

    checkpoints: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    observed_eval_seconds_per_family: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for result in replication_summary["results"]:
        arm = result["arm"]
        seed = int(result["seed"])
        selection_path = Path(result["selection"])
        development_path = Path(result["development"])
        selection = json.loads(selection_path.read_text())
        checkpoint = Path(selection["selected_checkpoint"])
        bridge = checkpoint / "bridge.safetensors"
        adapter = checkpoint / "language_adapter"
        prune = ROOT / "evidence" / f"stage-d-d1-{arm}-seed{seed}-prune-v1.json"
        record = {
            "step": int(selection["selected_step"]),
            "checkpoint": str(checkpoint),
            "bridge": str(bridge),
            "bridge_sha256": sha256_file(bridge),
            "adapter": str(adapter),
            "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "selection_evidence": str(selection_path),
            "selection_evidence_sha256": sha256_file(selection_path),
            "development_evidence": str(development_path),
            "development_evidence_sha256": sha256_file(development_path),
            "prune_receipt": str(prune),
            "prune_receipt_sha256": sha256_file(prune),
        }
        checkpoints[arm][str(seed)] = record
        development = json.loads(development_path.read_text())
        observed_eval_seconds_per_family[arm].append(
            float(development["total_wall_seconds"]) / int(development["scene_families"])
        )

    manifest = Path(replication_protocol["development"]["manifest"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    split_records = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            raise RuntimeError(f"empty confirmation split: {split}")
        split_records[split] = {
            "scene_families": len(selected),
            "family_id_sha256": stable_id_sha256(row["scene_family_id"] for row in selected),
            "tasks": dict(sorted(Counter(row["task"] for row in selected).items())),
            "base_answers": dict(sorted(Counter(row["base_answer"] for row in selected).items())),
        }

    family_evaluations_per_arm = sum(split_records[split]["scene_families"] for split in SPLITS)
    projected_raw_seconds = sum(
        max(observed_eval_seconds_per_family[arm]) * family_evaluations_per_arm * len(seeds)
        for arm in ARMS
    )
    safety_factor = 1.10
    projected_with_margin_hours = projected_raw_seconds * safety_factor / 3600.0
    if projected_with_margin_hours > 4.0:
        raise RuntimeError("projected confirmation exceeds the frozen four-hour GPU cap")

    source_files = {
        "replication_protocol": replication_protocol_path,
        "replication_summary": replication_summary_path,
        "data_protocol": args.data_protocol.resolve(),
        "data_admission": args.data_admission.resolve(),
        "human_audit": args.human_audit.resolve(),
    }
    code_files = {
        name: ROOT / name
        for name in (
            "evaluate_stage_a_visual_floor.py",
            "scoring_contract.py",
            "counterfactual_losses.py",
            "modeling.py",
            "train_stage_a_full_token.py",
            "stage_d_confirmation_contract.py",
            "validate_stage_d_confirmation_protocol.py",
            "run_stage_d_confirmation.py",
            "analyze_stage_d_confirmation.py",
        )
    }
    order = confirmation_execution_order(seeds)
    protocol = {
        "schema_version": "2026-09-13-v1",
        "status": "authorized_one_time_stage_d_iid_ood_confirmation_after_all_checkpoints_frozen",
        "authorized_by_user": True,
        "authorized_at": utc_now(),
        "authorization_scope": "one evaluation per frozen arm/seed/split on iid_test and ood_binding",
        "source_files": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in source_files.items()
        },
        "scorer_files": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in code_files.items()
        },
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "splits": split_records,
        "arms": list(ARMS),
        "seeds": seeds,
        "checkpoints": checkpoints,
        "conditions": list(CONDITIONS),
        "random_control": "within_task_answer_matched",
        "candidate_order": "manifest",
        "scoring_precision": "float32",
        "tf32_enabled": False,
        "verify_image_hashes": True,
        "execution_order": order,
        "expected_evaluations": len(order),
        "test_execution": {
            "each_frozen_arm_seed_split_once": True,
            "checkpoint_or_weight_changes_after_unseal_allowed": False,
            "all_results_including_failures_reported": True,
            "resume_may_adopt_only_a_complete_hash_valid_existing_result": True,
        },
        "statistics": {
            "primary_split": "iid_test",
            "primary_metric": "true_image_base_and_edited_family_joint_accuracy",
            "primary_comparison": "A_spatial_49_minus_F_matched_196",
            "primary_unit": "scene_family",
            "paired_training_seeds": True,
            "crossed_bootstrap_resamples": 10000,
            "crossed_bootstrap_seed": 20260913,
            "primary_inference_rule": (
                "report crossed bootstrap and paired-t intervals; superiority or noninferiority "
                "requires the corresponding lower bound from both intervals to pass"
            ),
            "superiority_point_threshold_pp": 2.0,
            "superiority_requires_both_lower_95_ci_gt_zero": True,
            "noninferiority_margin_pp": -1.0,
            "noninferiority_requires_both_lower_95_ci_gt_margin": True,
            "secondary_results": "report OOD, tasks and controls with intervals; no multiplicity-adjusted superiority claims",
            "margin_disclosure": "The -1pp engineering margin was frozen after D1 development and before any Stage-D test inference.",
        },
        "budget": {
            "confirmation_l20_gpu_hours_cap": 4.0,
            "projection_safety_factor": safety_factor,
            "projected_raw_gpu_hours": projected_raw_seconds / 3600.0,
            "projected_with_margin_gpu_hours": projected_with_margin_hours,
            "observed_d1_eval_seconds_per_family_max": {
                arm: max(values) for arm, values in observed_eval_seconds_per_family.items()
            },
        },
        "claim_boundary": (
            "The primary result is controlled synthetic new-IID confirmation across five paired adaptation seeds. "
            "OOD binding is prespecified secondary evidence. Neither is natural-image or second-backbone evidence, "
            "and this two-arm comparison is a package-level compressor effect rather than a token-count-only effect."
        ),
    }
    write_json_atomic(output, protocol)
    print(json.dumps({
        "status": protocol["status"],
        "output": str(output),
        "sha256": sha256_file(output),
        "evaluations": len(order),
        "projected_with_margin_gpu_hours": projected_with_margin_hours,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
