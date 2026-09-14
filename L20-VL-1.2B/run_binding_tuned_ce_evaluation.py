#!/usr/bin/env python3
"""Select and evaluate the tuned plain-CE checkpoint without opening final test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from counterfactual_losses import paired_cluster_bootstrap
from train_stage_a_full_token import sha256_file


def compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    binding = result["binding_selective_metrics"]
    true_metrics = binding["per_condition"]["true_image"]
    payload = {
        "affected_question_joint_accuracy_percent": true_metrics[
            "affected_question_joint_accuracy_percent"
        ],
        "invariant_question_joint_accuracy_percent": true_metrics[
            "invariant_question_joint_accuracy_percent"
        ],
        "scene_pair_selective_all_six_correct_percent": true_metrics[
            "scene_pair_selective_all_six_correct_percent"
        ],
    }
    if "random_image" in binding["per_condition"]:
        payload.update({
            "random_image_affected_question_joint_accuracy_percent": binding[
                "per_condition"
            ]["random_image"]["affected_question_joint_accuracy_percent"],
            "affected_true_minus_no_image": binding["affected_true_minus_no_image"],
            "affected_true_minus_random_image": binding[
                "affected_true_minus_random_image"
            ],
            "visual_floor": result["visual_floor"],
        })
    return payload


def paired_difference(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = {row["scene_family_id"]: row for row in candidate["predictions"]}
    baseline_rows = {row["scene_family_id"]: row for row in baseline["predictions"]}
    if set(candidate_rows) != set(baseline_rows):
        raise RuntimeError("paired tuned-CE/baseline family sets differ")
    differences = []
    clusters = []
    for family_id in sorted(candidate_rows):
        row = candidate_rows[family_id]
        if not row["question_role"].startswith("affected_"):
            continue
        other = baseline_rows[family_id]
        expected = row["expected"]
        if expected != other["expected"]:
            raise RuntimeError("paired tuned-CE/baseline labels differ")

        def joint(item: dict[str, Any]) -> float:
            predicted = item["predictions"]["true_image"]
            return float(
                predicted["base"] == expected["base"]
                and predicted["edited"] == expected["edited"]
            )

        differences.append(joint(row) - joint(other))
        clusters.append(row["statistical_cluster_id"])
    interval = paired_cluster_bootstrap(
        differences, clusters, resamples=10_000, seed=20260914
    )
    return {
        "estimate_pp": 100 * interval.estimate,
        "lower_95_ci_pp": 100 * interval.lower,
        "upper_95_ci_pp": 100 * interval.upper,
        "clusters": interval.clusters,
        "resamples": interval.resamples,
    }


def run_evaluator(
    root: Path,
    protocol_path: Path,
    manifest: Path,
    checkpoint: Path,
    adapter: Path,
    output: Path,
    split: str,
    scene_pairs: int | None,
    conditions: list[str],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    command = [
        sys.executable,
        str(root / "evaluate_stage_a_visual_floor.py"),
        "--protocol", str(protocol_path),
        "--checkpoint", str(checkpoint),
        "--language-adapter", str(adapter),
        "--manifest", str(manifest),
        "--output", str(output),
        "--split", split,
        "--batch-families", "24",
        "--max-tokens", "96",
        "--random-control", "within_task_answer_cluster_deranged",
        "--candidate-order", "manifest",
        "--scoring-precision", "bfloat16",
        "--verify-image-hashes",
        "--conditions", *conditions,
    ]
    if scene_pairs is not None:
        command.extend(["--scene-pairs", str(scene_pairs)])
    subprocess.run(command, cwd=root, check=True)
    return json.loads(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_binding_answer_only_development_v1":
        raise SystemExit("tuned plain-CE evaluation protocol is not authorized")
    if protocol.get("experiment_id") != "binding_tuned_plain_ce_evaluation_v1":
        raise SystemExit("unexpected tuned plain-CE evaluation id")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for filename, key in (
        ("run_binding_tuned_ce_evaluation.py", "orchestrator_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("evaluate_stage_a_visual_floor.py", "evaluator_sha256"),
        ("scoring_contract.py", "scoring_contract_sha256"),
    ):
        if sha256_file(root / filename) != protocol["source_code"][key]:
            raise SystemExit(f"evaluation source hash mismatch: {filename}")
    training_path = Path(protocol["training_protocol"]["path"])
    if sha256_file(training_path) != protocol["training_protocol"]["sha256"]:
        raise SystemExit("tuned plain-CE training protocol hash mismatch")
    training = json.loads(training_path.read_text())
    run_root = Path(training["resource_cap"]["full_run_output"])
    run = json.loads((run_root / "run.json").read_text())
    if run.get("status") != "complete" or run.get("protocol_sha256") != protocol["training_protocol"]["sha256"]:
        raise SystemExit("tuned plain-CE training receipt mismatch")
    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("tuned plain-CE evaluation manifest mismatch")
    args.output_dir.mkdir(parents=True)
    screens = []
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
    selected = sorted(
        screens,
        key=lambda item: (
            -item["affected_question_joint_accuracy_percent"],
            -item["scene_pair_selective_all_six_correct_percent"],
            item["step"],
        ),
    )[0]
    selected_root = run_root / f"step-{selected['step']:06d}"
    full_path = args.output_dir / f"full-mechanism-step-{selected['step']:06d}.json"
    full = run_evaluator(
        root,
        protocol_path,
        manifest,
        selected_root / "bridge.safetensors",
        selected_root / "language_adapter",
        full_path,
        "mechanism_dev",
        None,
        ["true_image", "random_image", "no_image"],
    )
    selection_path = args.output_dir / f"selection-step-{selected['step']:06d}.json"
    selection = run_evaluator(
        root,
        protocol_path,
        manifest,
        selected_root / "bridge.safetensors",
        selected_root / "language_adapter",
        selection_path,
        "selection_dev",
        None,
        ["true_image", "random_image", "no_image"],
    )
    e0_summary_path = Path(protocol["comparison"]["e0_summary"])
    if sha256_file(e0_summary_path) != protocol["comparison"]["e0_summary_sha256"]:
        raise SystemExit("E0 comparison summary hash mismatch")
    e0_summary = json.loads(e0_summary_path.read_text())
    e0_full = json.loads(Path(e0_summary["full_mechanism"]["result"]).read_text())
    comparison = paired_difference(full, e0_full)
    decision = (
        "sufficient_tuned_plain_ce"
        if full["visual_floor"]["passes_all"]
        else "insufficient_tuned_plain_ce"
    )
    summary = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_development_only",
        "decision": decision,
        "evaluation_protocol_sha256": sha256_file(protocol_path),
        "training_protocol_sha256": protocol["training_protocol"]["sha256"],
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
        "tuned_plain_ce_minus_e0_affected_joint": comparison,
        "final_test_used": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
