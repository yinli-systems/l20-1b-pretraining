#!/usr/bin/env python3
"""Select and compare query-reader CE and CAVR without opening final test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_binding_tuned_ce_evaluation import (
    compact_metrics,
    paired_difference,
    run_evaluator,
)
from train_stage_a_full_token import sha256_file


def resolved_result(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def select_checkpoint(screens: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        screens,
        key=lambda item: (
            -item["affected_question_joint_accuracy_percent"],
            -item["scene_pair_selective_all_six_correct_percent"],
            item["step"],
        ),
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    allowed_statuses = {
        "authorized_counterfactual_address_value_routing_evaluation_v1",
        "authorized_counterfactual_address_value_routing_evaluation_v2",
    }
    if protocol.get("status") not in allowed_statuses:
        raise SystemExit("CAVR evaluation protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for filename, key in (
        ("run_counterfactual_address_value_routing_evaluation.py", "orchestrator_sha256"),
        ("run_binding_tuned_ce_evaluation.py", "evaluation_helpers_sha256"),
        ("evaluate_stage_a_visual_floor.py", "evaluator_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("scoring_contract.py", "scoring_contract_sha256"),
    ):
        if sha256_file(root / filename) != protocol["source_code"][key]:
            raise SystemExit(f"CAVR evaluation source hash mismatch: {filename}")

    training_path = Path(protocol["training_protocol"]["path"])
    if sha256_file(training_path) != protocol["training_protocol"]["sha256"]:
        raise SystemExit("CAVR training protocol hash mismatch")
    training = json.loads(training_path.read_text())
    if training.get("test_split_use_authorized") is not False:
        raise SystemExit("CAVR training protocol used final test")

    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("CAVR evaluation manifest mismatch")
    args.output_dir.mkdir(parents=True)
    arm_results: dict[str, dict[str, Any]] = {}
    full_payloads: dict[str, dict[str, Any]] = {}
    for arm_name in protocol["execution_order"]:
        arm = protocol["arms"][arm_name]
        child_protocol_path = Path(arm["evaluation_protocol"])
        if sha256_file(child_protocol_path) != arm["evaluation_protocol_sha256"]:
            raise SystemExit(f"{arm_name} evaluation protocol hash mismatch")
        child_protocol = json.loads(child_protocol_path.read_text())
        if child_protocol["architecture"] != training["arms"][arm_name]["architecture"]:
            raise SystemExit(f"{arm_name} evaluation/training architecture mismatch")
        run_root = Path(arm["run"])
        run = json.loads((run_root / "run.json").read_text())
        if (
            run.get("status") != "complete"
            or run.get("arm") != arm_name
            or run.get("protocol_sha256") != protocol["training_protocol"]["sha256"]
            or run.get("optimizer_step") != training["optimization"]["max_optimizer_steps"]
            or "parent_after" not in run
        ):
            raise SystemExit(f"{arm_name} training receipt mismatch")

        arm_output = args.output_dir / arm_name
        arm_output.mkdir()
        screens = []
        for step in protocol["checkpoint_selection"]["screen_checkpoints"]:
            checkpoint_root = run_root / f"step-{step:06d}"
            output = arm_output / f"screen-step-{step:06d}.json"
            result = run_evaluator(
                root,
                child_protocol_path,
                manifest,
                checkpoint_root / "bridge.safetensors",
                checkpoint_root / "language_adapter",
                output,
                "mechanism_dev",
                protocol["checkpoint_selection"]["screen_scene_pairs"],
                ["true_image"],
            )
            screens.append({
                "step": step,
                "result": str(output),
                "result_sha256": sha256_file(output),
                **compact_metrics(result),
            })
        selected = select_checkpoint(screens)
        selected_root = run_root / f"step-{selected['step']:06d}"
        full_path = arm_output / f"full-mechanism-step-{selected['step']:06d}.json"
        full = run_evaluator(
            root,
            child_protocol_path,
            manifest,
            selected_root / "bridge.safetensors",
            selected_root / "language_adapter",
            full_path,
            "mechanism_dev",
            None,
            ["true_image", "random_image", "no_image"],
        )
        selection_path = arm_output / f"selection-step-{selected['step']:06d}.json"
        selection = run_evaluator(
            root,
            child_protocol_path,
            manifest,
            selected_root / "bridge.safetensors",
            selected_root / "language_adapter",
            selection_path,
            "selection_dev",
            None,
            ["true_image", "random_image", "no_image"],
        )
        full_payloads[arm_name] = full
        arm_results[arm_name] = {
            "training": {
                "run": str(run_root / "run.json"),
                "run_sha256": sha256_file(run_root / "run.json"),
                "wall_seconds": run["wall_seconds"],
                "language_forward_passes_per_batch": run[
                    "language_forward_passes_per_batch"
                ],
                "peak_allocated_gib": run["peak_allocated_gib"],
            },
            "screens": screens,
            "selected": selected,
            "full_mechanism": {
                "result": str(full_path),
                "result_sha256": sha256_file(full_path),
                **compact_metrics(full),
            },
            "selection_dev": {
                "result": str(selection_path),
                "result_sha256": sha256_file(selection_path),
                **compact_metrics(selection),
            },
        }

    tuned_path = Path(protocol["comparison"]["tuned_plain_ce_summary"])
    if sha256_file(tuned_path) != protocol["comparison"]["tuned_plain_ce_summary_sha256"]:
        raise SystemExit("tuned plain-CE summary hash mismatch")
    tuned_summary = json.loads(tuned_path.read_text())
    tuned_full_path = resolved_result(root, tuned_summary["full_mechanism"]["result"])
    if sha256_file(tuned_full_path) != tuned_summary["full_mechanism"]["result_sha256"]:
        raise SystemExit("tuned plain-CE full result hash mismatch")
    tuned_full = json.loads(tuned_full_path.read_text())
    comparisons = {
        "cavr_minus_query_ce_affected_joint": paired_difference(
            full_payloads["cavr"], full_payloads["query_ce"]
        ),
        "query_ce_minus_tuned_plain_ce_affected_joint": paired_difference(
            full_payloads["query_ce"], tuned_full
        ),
        "cavr_minus_tuned_plain_ce_affected_joint": paired_difference(
            full_payloads["cavr"], tuned_full
        ),
    }
    cavr_full = arm_results["cavr"]["full_mechanism"]
    cavr_vs_query = comparisons["cavr_minus_query_ce_affected_joint"]
    rule = protocol["decision_rule"]
    promising = (
        cavr_full["affected_question_joint_accuracy_percent"]
        >= rule["minimum_mechanism_affected_joint_percent"]
        and cavr_full["visual_floor"]["passes_all"]
        and cavr_full["scene_pair_selective_all_six_correct_percent"] > 0
        and cavr_vs_query["estimate_pp"] >= rule["minimum_cavr_minus_query_ce_pp"]
        and cavr_vs_query["lower_95_ci_pp"] > 0
    )
    summary = {
        "schema_version": protocol["schema_version"],
        "status": "complete_development_only",
        "decision": "promising_cavr_requires_external_validation" if promising else "cavr_not_yet_sufficient",
        "evaluation_protocol_sha256": sha256_file(protocol_path),
        "training_protocol_sha256": protocol["training_protocol"]["sha256"],
        "arms": arm_results,
        "comparisons": comparisons,
        "gpu_time_matched_control_required_if_promising": True,
        "final_test_used": False,
        "novelty_boundary": protocol["novelty_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
