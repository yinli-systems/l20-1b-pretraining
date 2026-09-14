#!/usr/bin/env python3
"""Evaluate residual-CE and IIBR under the frozen E1 development protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from counterfactual_losses import paired_cluster_bootstrap


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(result: dict[str, Any]) -> dict[str, Any]:
    binding = result["binding_selective_metrics"]
    true_metrics = binding["per_condition"]["true_image"]
    return {
        "affected_question_joint_accuracy_percent": true_metrics[
            "affected_question_joint_accuracy_percent"
        ],
        "invariant_question_joint_accuracy_percent": true_metrics[
            "invariant_question_joint_accuracy_percent"
        ],
        "scene_pair_selective_all_six_correct_percent": true_metrics[
            "scene_pair_selective_all_six_correct_percent"
        ],
        "random_image_affected_question_joint_accuracy_percent": binding["per_condition"][
            "random_image"
        ]["affected_question_joint_accuracy_percent"],
        "affected_true_minus_random_image": binding["affected_true_minus_random_image"],
        "visual_floor": result["visual_floor"],
    }


def paired_difference(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = {row["scene_family_id"]: row for row in candidate["predictions"]}
    baseline_rows = {row["scene_family_id"]: row for row in baseline["predictions"]}
    if set(candidate_rows) != set(baseline_rows):
        raise RuntimeError("paired result family sets differ")
    differences = []
    clusters = []
    for family_id in sorted(candidate_rows):
        row = candidate_rows[family_id]
        if not row["question_role"].startswith("affected_"):
            continue
        other = baseline_rows[family_id]
        if row["expected"] != other["expected"]:
            raise RuntimeError("paired result labels differ")
        expected = row["expected"]

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
    random_control: str,
    conditions: list[str],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    command = [
        sys.executable,
        str(root / "evaluate_stage_a_visual_floor.py"),
        "--protocol",
        str(protocol_path),
        "--checkpoint",
        str(checkpoint),
        "--language-adapter",
        str(adapter),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--split",
        split,
        "--batch-families",
        "24",
        "--max-tokens",
        "96",
        "--random-control",
        random_control,
        "--candidate-order",
        "manifest",
        "--scoring-precision",
        "bfloat16",
        "--verify-image-hashes",
        "--conditions",
        *conditions,
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
        raise SystemExit("residual evaluation protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source_paths = {
        "orchestrator_sha256": Path(__file__).resolve(),
        "modeling_sha256": root / "modeling.py",
        "evaluator_sha256": root / "evaluate_stage_a_visual_floor.py",
        "scoring_contract_sha256": root / "scoring_contract.py",
    }
    for label, path in source_paths.items():
        if sha256_file(path) != protocol["source_code"][label]:
            raise SystemExit(f"evaluation source hash mismatch: {label}")
    training_protocol_path = Path(protocol["training_protocol"]["path"])
    if sha256_file(training_protocol_path) != protocol["training_protocol"]["sha256"]:
        raise SystemExit("residual training protocol hash mismatch")
    training_protocol = json.loads(training_protocol_path.read_text())
    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("evaluation manifest hash mismatch")
    adapter = Path(training_protocol["parents"]["language_adapter"])
    screen = protocol["checkpoint_selection"]
    conditions = protocol["scoring"]["conditions"]
    random_control = protocol["scoring"]["random_control"]
    arm_screens: dict[str, list[dict[str, Any]]] = {}
    selected_checkpoints: dict[str, dict[str, Any]] = {}
    selected_full_results: dict[str, dict[str, Any]] = {}
    for arm in ("residual_ce", "iibr"):
        run_root = Path(training_protocol["resource_cap"]["output_roots"][arm])
        run_receipt = json.loads((run_root / "run.json").read_text())
        if run_receipt.get("status") != "complete":
            raise SystemExit(f"training run is not complete: {arm}")
        if run_receipt.get("protocol_sha256") != protocol["training_protocol"]["sha256"]:
            raise SystemExit(f"training protocol receipt mismatch: {arm}")
        arm_screens[arm] = []
        for step in screen["screen_checkpoints_per_arm"]:
            checkpoint = run_root / f"step-{step:06d}" / "bridge.safetensors"
            output = args.output_dir / f"screen-{arm}-step-{step:06d}.json"
            result = run_evaluator(
                root,
                protocol_path,
                manifest,
                checkpoint,
                adapter,
                output,
                "mechanism_dev",
                screen["screen_scene_pairs"],
                random_control,
                conditions,
            )
            arm_screens[arm].append({
                "step": step,
                "result": str(output),
                "result_sha256": sha256_file(output),
                **metrics(result),
            })
        selected = sorted(
            arm_screens[arm],
            key=lambda item: (
                -item["affected_question_joint_accuracy_percent"],
                -item["scene_pair_selective_all_six_correct_percent"],
                item["step"],
            ),
        )[0]
        selected_checkpoints[arm] = selected
        checkpoint = run_root / f"step-{selected['step']:06d}" / "bridge.safetensors"
        full_output = args.output_dir / f"full-mechanism-{arm}-step-{selected['step']:06d}.json"
        full_result = run_evaluator(
            root,
            protocol_path,
            manifest,
            checkpoint,
            adapter,
            full_output,
            "mechanism_dev",
            None,
            random_control,
            conditions,
        )
        selected_full_results[arm] = {
            "step": selected["step"],
            "result": str(full_output),
            "result_sha256": sha256_file(full_output),
            "payload": full_result,
            **metrics(full_result),
        }
    selected_arm = sorted(
        ("residual_ce", "iibr"),
        key=lambda arm: (
            -selected_full_results[arm]["affected_question_joint_accuracy_percent"],
            -selected_full_results[arm]["scene_pair_selective_all_six_correct_percent"],
            0 if arm == "residual_ce" else 1,
        ),
    )[0]
    selected_run = Path(training_protocol["resource_cap"]["output_roots"][selected_arm])
    selected_step = selected_checkpoints[selected_arm]["step"]
    selection_output = args.output_dir / f"selection-{selected_arm}-step-{selected_step:06d}.json"
    selection_result = run_evaluator(
        root,
        protocol_path,
        manifest,
        selected_run / f"step-{selected_step:06d}" / "bridge.safetensors",
        adapter,
        selection_output,
        "selection_dev",
        None,
        random_control,
        conditions,
    )
    e0_summary = json.loads(
        Path(training_protocol["prerequisite"]["plain_ce_summary"]).read_text()
    )
    e0_full = json.loads(Path(e0_summary["full_mechanism"]["result"]).read_text())
    comparison = {
        "iibr_minus_residual_ce_affected_joint": paired_difference(
            selected_full_results["iibr"]["payload"],
            selected_full_results["residual_ce"]["payload"],
        ),
        "selected_residual_minus_full_adaptation_plain_ce_affected_joint": paired_difference(
            selected_full_results[selected_arm]["payload"], e0_full
        ),
        "compute_boundary": training_protocol["comparison_contract"]["compute_boundary"],
    }
    serializable_full = {
        arm: {key: value for key, value in result.items() if key != "payload"}
        for arm, result in selected_full_results.items()
    }
    summary = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_development_only",
        "evaluation_protocol_sha256": sha256_file(protocol_path),
        "training_protocol_sha256": protocol["training_protocol"]["sha256"],
        "screens": arm_screens,
        "selected_checkpoints": selected_checkpoints,
        "full_mechanism": serializable_full,
        "selected_arm": selected_arm,
        "selection_dev": {
            "result": str(selection_output),
            "result_sha256": sha256_file(selection_output),
            **metrics(selection_result),
        },
        "paired_comparisons": comparison,
        "final_test_used": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_dir / "summary.json")
    print(json.dumps({
        "selected_checkpoints": selected_checkpoints,
        "full_mechanism": serializable_full,
        "selected_arm": selected_arm,
        "selection_dev": summary["selection_dev"],
        "paired_comparisons": comparison,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
