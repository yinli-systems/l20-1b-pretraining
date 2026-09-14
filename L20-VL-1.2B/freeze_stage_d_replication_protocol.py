#!/usr/bin/env python3
"""Freeze D1 only after the answer-balanced LR screen passes its audit."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-protocol", type=Path, required=True)
    parser.add_argument("--lr-summary", type=Path, required=True)
    parser.add_argument("--lr-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=5, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.seeds)) != 5:
        raise RuntimeError("D1 requires five unique seeds")
    lr_protocol = json.loads(args.lr_protocol.read_text())
    summary = json.loads(args.lr_summary.read_text())
    audit = json.loads(args.lr_audit.read_text())
    lr_protocol_sha = sha256_file(args.lr_protocol)
    summary_sha = sha256_file(args.lr_summary)
    audit_sha = sha256_file(args.lr_audit)
    if lr_protocol.get("status") != "authorized_stage_d_lr_microsearch":
        raise RuntimeError("unexpected LR protocol status")
    if (
        summary.get("status") != "complete"
        or summary.get("protocol_sha256") != lr_protocol_sha
        or audit.get("status") != "pass"
        or audit.get("protocol_sha256") != lr_protocol_sha
        or audit.get("summary_sha256") != summary_sha
    ):
        raise RuntimeError("LR screen and audit chain is incomplete")
    if lr_protocol["optimization"]["seed"] in args.seeds:
        raise RuntimeError("D1 seeds must be independent of the LR-selection seed")
    if any(summary.get(key) != 0 for key in (
        "prior_test_predictions_produced",
        "new_iid_test_predictions_produced",
        "ood_test_predictions_produced",
    )):
        raise RuntimeError("test predictions exist before D1 protocol freeze")
    selected = summary["selected_learning_rate_multipliers"]
    if set(selected) != set(lr_protocol["arms"]):
        raise RuntimeError("LR selection does not cover both D1 arms")
    arms = deepcopy(lr_protocol["arms"])
    for arm, config in arms.items():
        multiplier = float(selected[arm])
        if multiplier not in lr_protocol["allowed_learning_rate_multipliers"]:
            raise RuntimeError(f"unfrozen selected multiplier for {arm}")
        config["selected_learning_rate_multiplier"] = multiplier

    required_audits = deepcopy(lr_protocol["required_audits"])
    required_audits[str(args.lr_protocol.resolve())] = lr_protocol_sha
    required_audits[str(args.lr_summary.resolve())] = summary_sha
    required_audits[str(args.lr_audit.resolve())] = audit_sha
    smoke = Path(lr_protocol["training_smoke_receipt"])
    required_audits[str(smoke.resolve())] = sha256_file(smoke)
    protocol = {
        "schema_version": "2026-09-13-v1",
        "status": "authorized_stage_d_two_arm_five_seed_replication",
        "authorized_by_user": True,
        "frozen_at": utc_now(),
        "purpose": "estimate the package-level 49-token versus matched full-token effect across five paired training seeds before any confirmation-test inference",
        "test_split_use_authorized": False,
        "prior_stage_c_test_role": "regression_only_not_used",
        "new_iid_test_status": "sealed",
        "new_ood_binding_test_status": "sealed",
        "parents": deepcopy(lr_protocol["parents"]),
        "data": deepcopy(lr_protocol["data"]),
        "development": deepcopy(lr_protocol["development"]),
        "arms": arms,
        "allowed_seeds": args.seeds,
        "lr_selection_seed": lr_protocol["optimization"]["seed"],
        "distillation_temperature": lr_protocol["distillation_temperature"],
        "preference_scale": lr_protocol["preference_scale"],
        "huber_beta": lr_protocol["huber_beta"],
        "optimization": deepcopy(lr_protocol["optimization"]),
        "selection": {
            "split": "development",
            "families_per_task": lr_protocol["development"]["families_per_task_for_screen"],
            "families_per_task_answer_stratum": lr_protocol["development"]["families_per_task_answer_stratum"],
            "candidate_steps": [100, 300, 500, 700, 875],
            "rule": "maximize family-joint accuracy, then binding-task macro, then ordinary primary-query accuracy, then prefer the earlier optimizer step"
        },
        "equalization": {
            "same_parent_bridge_and_language_adapter": True,
            "same_14000_training_families": True,
            "same_875_optimizer_updates": True,
            "same_answer_only_objective": True,
            "same_paired_seed_for_each_arm": True,
            "same_development_subset_and_checkpoint_candidates": True,
            "same_lr_search_allowance": True,
            "wall_clock_reported_not_equalized": True
        },
        "seed_boundary": "Each paired seed changes deterministic data order and stochastic adaptation. A_spatial_49 also initializes its new compressor from that seed; F_matched_196 has no new compressor. This is adaptation-seed evidence, not parent-pretraining-seed evidence.",
        "capacity_boundary": lr_protocol["capacity_boundary"],
        "budget": {
            "d1_training_gpu_hours_cap": 10.0,
            "d1_total_gpu_hours_cap": 10.0,
            "d0_d1_d2_total_l20_gpu_hours_cap": lr_protocol["budget"]["d0_d1_d2_total_l20_gpu_hours_cap"]
        },
        "required_audits": required_audits,
        "post_d1_gate": "freeze all ten selected checkpoints and a separately hashed test-unseal protocol before exactly one evaluation per arm/seed on new IID and OOD tests",
        "prohibited": [
            "using any test split for training, LR selection or checkpoint selection",
            "calling development scores confirmatory efficacy",
            "calling the two-arm contrast a token-count-only or capacity-only effect",
            "claiming natural-image generalization",
            "claiming seed robustness before all ten runs and confirmation tests complete"
        ],
        "claim_boundary": "D1 training and development selection estimate adaptation stability while keeping all tests sealed. They do not authorize a compression-caused gain or final efficacy claim."
    }
    write_json_atomic(args.output, protocol)
    print(json.dumps({
        "status": "frozen",
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "selected_learning_rate_multipliers": selected,
        "seeds": args.seeds,
        "tests_sealed": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
