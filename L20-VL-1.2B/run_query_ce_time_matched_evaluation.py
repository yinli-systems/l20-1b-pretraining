#!/usr/bin/env python3
"""Evaluate the pre-registered GPU-time-matched query-CE control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_binding_tuned_ce_evaluation import compact_metrics, paired_difference, run_evaluator
from run_counterfactual_address_value_routing_evaluation import resolved_result, select_checkpoint
from train_stage_a_full_token import sha256_file


def decision_flags(
    cavr_minus_control: dict[str, float],
    wall_ratio: float,
    ratio_range: list[float],
    minimum_advantage_pp: float,
) -> dict[str, bool]:
    return {
        "wall_time_matched": ratio_range[0] <= wall_ratio <= ratio_range[1],
        "minimum_advantage_met": cavr_minus_control["estimate_pp"] >= minimum_advantage_pp,
        "paired_ci_lower_gt_zero": cavr_minus_control["lower_95_ci_pp"] > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    root = Path(__file__).resolve().parent
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_query_ce_gpu_time_matched_evaluation_v1":
        raise SystemExit("time-matched evaluation protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    for filename, key in (
        ("run_query_ce_time_matched_evaluation.py", "orchestrator_sha256"),
        ("run_binding_tuned_ce_evaluation.py", "evaluation_helpers_sha256"),
        ("run_counterfactual_address_value_routing_evaluation.py", "cavr_orchestrator_sha256"),
        ("evaluate_stage_a_visual_floor.py", "evaluator_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("scoring_contract.py", "scoring_contract_sha256"),
    ):
        if sha256_file(root / filename) != protocol["source_code"][key]:
            raise SystemExit(f"time-matched evaluation source hash mismatch: {filename}")

    training_path = Path(protocol["training_protocol"]["path"])
    if sha256_file(training_path) != protocol["training_protocol"]["sha256"]:
        raise SystemExit("time-matched training protocol hash mismatch")
    run_root = Path(protocol["run"])
    run = json.loads((run_root / "run.json").read_text())
    if (
        run.get("status") != "complete"
        or run.get("arm") != "query_ce"
        or run.get("protocol_sha256") != protocol["training_protocol"]["sha256"]
        or run.get("optimizer_step") != 800
        or run.get("parent_before") != run.get("parent_after")
    ):
        raise SystemExit("time-matched training receipt mismatch")
    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("time-matched evaluation manifest mismatch")

    args.output_dir.mkdir(parents=True)
    screens: list[dict[str, Any]] = []
    for step in protocol["checkpoint_selection"]["screen_checkpoints"]:
        checkpoint_root = run_root / f"step-{step:06d}"
        output = args.output_dir / f"screen-step-{step:06d}.json"
        result = run_evaluator(
            root,
            protocol_path,
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
    full_path = args.output_dir / f"full-mechanism-step-{selected['step']:06d}.json"
    full = run_evaluator(
        root, protocol_path, manifest,
        selected_root / "bridge.safetensors",
        selected_root / "language_adapter",
        full_path, "mechanism_dev", None,
        ["true_image", "random_image", "no_image"],
    )
    selection_path = args.output_dir / f"selection-step-{selected['step']:06d}.json"
    selection = run_evaluator(
        root, protocol_path, manifest,
        selected_root / "bridge.safetensors",
        selected_root / "language_adapter",
        selection_path, "selection_dev", None,
        ["true_image", "random_image", "no_image"],
    )

    cavr_summary_path = Path(protocol["comparison"]["cavr_summary"])
    if sha256_file(cavr_summary_path) != protocol["comparison"]["cavr_summary_sha256"]:
        raise SystemExit("CAVR summary hash mismatch")
    cavr_summary = json.loads(cavr_summary_path.read_text())
    cavr_full_path = resolved_result(root, cavr_summary["arms"]["cavr"]["full_mechanism"]["result"])
    if sha256_file(cavr_full_path) != cavr_summary["arms"]["cavr"]["full_mechanism"]["result_sha256"]:
        raise SystemExit("CAVR full result hash mismatch")
    cavr_full = json.loads(cavr_full_path.read_text())
    cavr_minus_control = paired_difference(cavr_full, full)
    original_query_path = resolved_result(
        root, cavr_summary["arms"]["query_ce"]["full_mechanism"]["result"]
    )
    if sha256_file(original_query_path) != cavr_summary["arms"]["query_ce"]["full_mechanism"]["result_sha256"]:
        raise SystemExit("original query-CE full result hash mismatch")
    original_query = json.loads(original_query_path.read_text())

    reference_wall = float(cavr_summary["arms"]["cavr"]["training"]["wall_seconds"])
    wall_ratio = float(run["wall_seconds"]) / reference_wall
    flags = decision_flags(
        cavr_minus_control,
        wall_ratio,
        protocol["comparison"]["accepted_wall_time_ratio"],
        protocol["decision_rule"]["minimum_cavr_advantage_pp"],
    )
    advantage_survives = all(flags.values())
    summary = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_development_only",
        "decision": (
            "cavr_advantage_survives_gpu_time_match"
            if advantage_survives
            else "cavr_advantage_not_confirmed_under_gpu_time_match"
        ),
        "evaluation_protocol_sha256": sha256_file(protocol_path),
        "training_protocol_sha256": protocol["training_protocol"]["sha256"],
        "training": {
            "run": str(run_root / "run.json"),
            "run_sha256": sha256_file(run_root / "run.json"),
            "wall_seconds": run["wall_seconds"],
            "reference_cavr_wall_seconds": reference_wall,
            "wall_time_ratio": wall_ratio,
            "language_forward_passes": 800,
            "peak_allocated_gib": run["peak_allocated_gib"],
        },
        "screens": screens,
        "selected": selected,
        "full_mechanism": {"result": str(full_path), "result_sha256": sha256_file(full_path), **compact_metrics(full)},
        "selection_dev": {"result": str(selection_path), "result_sha256": sha256_file(selection_path), **compact_metrics(selection)},
        "comparisons": {
            "cavr_minus_time_matched_query_ce_affected_joint": cavr_minus_control,
            "time_matched_minus_original_query_ce_affected_joint": paired_difference(full, original_query),
        },
        "decision_flags": flags,
        "final_test_used": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
