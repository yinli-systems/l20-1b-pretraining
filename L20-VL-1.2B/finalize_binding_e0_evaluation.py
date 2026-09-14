#!/usr/bin/env python3
"""Finalize already-completed E0 evaluations after the original summary-only failure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(result: dict[str, Any]) -> dict[str, Any]:
    binding = result["binding_selective_metrics"]
    return {
        "affected_question_joint_accuracy_percent": binding["per_condition"]["true_image"][
            "affected_question_joint_accuracy_percent"
        ],
        "random_image_affected_question_joint_accuracy_percent": binding["per_condition"]["random_image"][
            "affected_question_joint_accuracy_percent"
        ],
        "invariant_question_joint_accuracy_percent": binding["per_condition"]["true_image"][
            "invariant_question_joint_accuracy_percent"
        ],
        "scene_pair_selective_all_six_correct_percent": binding["per_condition"]["true_image"][
            "scene_pair_selective_all_six_correct_percent"
        ],
        "affected_true_minus_no_image": binding["affected_true_minus_no_image"],
        "affected_true_minus_random_image": binding["affected_true_minus_random_image"],
        "visual_floor": result["visual_floor"],
    }


def validate_result(
    path: Path,
    evaluation_protocol_sha256: str,
    manifest_sha256: str,
    checkpoint: Path,
    split: str,
    families: int,
    random_control: str,
) -> dict[str, Any]:
    result = json.loads(path.read_text())
    expected = {
        "protocol_sha256": evaluation_protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": split,
        "scene_families": families,
    }
    actual = {name: result.get(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"result binding mismatch for {path}: {actual} != {expected}")
    if result.get("random_control", {}).get("policy") != random_control:
        raise RuntimeError(f"random-control policy mismatch for {path}")
    if result.get("image_integrity_audit", {}).get("all_match_manifest") is not True:
        raise RuntimeError(f"image-integrity audit did not pass for {path}")
    if result.get("scope") != "binding_intervention_development":
        raise RuntimeError(f"unexpected evaluation scope for {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalization-protocol", type=Path, required=True)
    args = parser.parse_args()
    finalization_path = args.finalization_protocol.resolve()
    finalization = json.loads(finalization_path.read_text())
    if finalization.get("status") != "authorized_binding_e0_summary_recovery_v1":
        raise SystemExit("summary recovery is not authorized")
    if sha256_file(Path(__file__).resolve()) != finalization["finalizer_sha256"]:
        raise SystemExit("finalizer source hash mismatch")
    evaluation_protocol_path = Path(finalization["evaluation_protocol"]["path"])
    evaluation_protocol_sha256 = sha256_file(evaluation_protocol_path)
    if evaluation_protocol_sha256 != finalization["evaluation_protocol"]["sha256"]:
        raise SystemExit("evaluation protocol hash mismatch")
    evaluation_protocol = json.loads(evaluation_protocol_path.read_text())
    training_protocol_path = Path(evaluation_protocol["training_protocol"]["path"])
    if sha256_file(training_protocol_path) != evaluation_protocol["training_protocol"]["sha256"]:
        raise SystemExit("training protocol hash mismatch")
    training_protocol = json.loads(training_protocol_path.read_text())
    manifest_sha256 = evaluation_protocol["data"]["manifest_sha256"]
    output_dir = Path(finalization["evaluation_output_dir"])
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    run_root = Path(training_protocol["resource_cap"]["full_run_output"])
    random_control = evaluation_protocol["scoring"]["random_control"]
    candidate_specs = [{
        "step": 0,
        "label": "frozen_parent",
        "checkpoint": Path(training_protocol["parents"]["bridge"]),
    }]
    for step in evaluation_protocol["checkpoint_selection"]["screen_checkpoints"]:
        candidate_specs.append({
            "step": step,
            "label": f"step_{step:06d}",
            "checkpoint": run_root / f"step-{step:06d}" / "bridge.safetensors",
        })
    screens = []
    for candidate in candidate_specs:
        path = output_dir / f"screen-{candidate['label']}.json"
        result = validate_result(
            path,
            evaluation_protocol_sha256,
            manifest_sha256,
            candidate["checkpoint"],
            "mechanism_dev",
            evaluation_protocol["checkpoint_selection"]["screen_scene_pairs"] * 6,
            random_control,
        )
        screens.append({
            "step": candidate["step"],
            "label": candidate["label"],
            "result": str(path),
            "result_sha256": sha256_file(path),
            **metrics(result),
        })
    selected = sorted(
        screens,
        key=lambda item: (
            -item["affected_question_joint_accuracy_percent"],
            -item["scene_pair_selective_all_six_correct_percent"],
            item["step"],
        ),
    )[0]
    selected_spec = next(item for item in candidate_specs if item["step"] == selected["step"])
    full_path = output_dir / f"full-mechanism-{selected_spec['label']}.json"
    selection_path = output_dir / f"selection-{selected_spec['label']}.json"
    full = validate_result(
        full_path,
        evaluation_protocol_sha256,
        manifest_sha256,
        selected_spec["checkpoint"],
        "mechanism_dev",
        160 * 6,
        random_control,
    )
    selection = validate_result(
        selection_path,
        evaluation_protocol_sha256,
        manifest_sha256,
        selected_spec["checkpoint"],
        "selection_dev",
        160 * 6,
        random_control,
    )
    payload = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_development_only",
        "recovery_reason": "All GPU evaluations completed; the original orchestrator failed only while serializing final_test_used with a lowercase Python literal.",
        "finalization_protocol_sha256": sha256_file(finalization_path),
        "evaluation_protocol_sha256": evaluation_protocol_sha256,
        "training_protocol_sha256": evaluation_protocol["training_protocol"]["sha256"],
        "screens": screens,
        "selected": selected,
        "full_mechanism": {
            "result": str(full_path),
            "result_sha256": sha256_file(full_path),
            **metrics(full),
        },
        "selection_dev": {
            "result": str(selection_path),
            "result_sha256": sha256_file(selection_path),
            **metrics(selection),
        },
        "plain_ce_decision": "insufficient_binding_repair",
        "final_test_used": False,
        "claim_boundary": evaluation_protocol["claim_boundary"],
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    print(json.dumps({
        "status": payload["status"],
        "selected": payload["selected"],
        "full_mechanism": payload["full_mechanism"],
        "selection_dev": payload["selection_dev"],
        "plain_ce_decision": payload["plain_ce_decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
