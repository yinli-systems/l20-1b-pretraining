#!/usr/bin/env python3
"""Run the Stage-D deterministic scorer and input-integrity audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def true_predictions(result: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        row["scene_family_id"]: row["predictions"]["true_image"]
        for row in result["predictions"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--language-adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families-per-task", type=int, default=50)
    parser.add_argument("--reference-batch-families", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    runs = {
        "reference": {
            "batch": args.reference_batch_families,
            "candidate_order": "manifest",
            "conditions": ("true_image", "random_image", "no_image"),
            "verify_hashes": True,
            "manifest": args.manifest,
        },
        "repeat": {
            "batch": args.reference_batch_families,
            "candidate_order": "manifest",
            "conditions": ("true_image",),
            "verify_hashes": False,
            "manifest": args.manifest,
        },
        "single_family_batch": {
            "batch": 1,
            "candidate_order": "manifest",
            "conditions": ("true_image",),
            "verify_hashes": False,
            "manifest": args.manifest,
        },
        "reversed_candidates": {
            "batch": args.reference_batch_families,
            "candidate_order": "reversed",
            "conditions": ("true_image",),
            "verify_hashes": False,
            "manifest": args.manifest,
        },
    }
    with tempfile.TemporaryDirectory(prefix="stage-d-order-audit-") as temporary:
        reversed_manifest = Path(temporary) / "manifest-reversed.jsonl"
        lines = [line for line in args.manifest.read_text().splitlines() if line]
        reversed_manifest.write_text("\n".join(reversed(lines)) + "\n")
        runs["reversed_manifest"] = {
            "batch": args.reference_batch_families,
            "candidate_order": "manifest",
            "conditions": ("true_image",),
            "verify_hashes": False,
            "manifest": reversed_manifest,
        }

        outputs = {}
        for name, settings in runs.items():
            output = args.output_dir / f"{name}.json"
            command = [
                sys.executable,
                str(ROOT / "evaluate_stage_a_visual_floor.py"),
                "--protocol", str(args.protocol),
                "--checkpoint", str(args.checkpoint),
                "--language-adapter", str(args.language_adapter),
                "--manifest", str(settings["manifest"]),
                "--output", str(output),
                "--split", "development",
                "--families-per-task", str(args.families_per_task),
                "--batch-families", str(settings["batch"]),
                "--candidate-order", settings["candidate_order"],
                "--random-control", "within_task_answer_matched",
                "--scoring-precision", "float32",
                "--conditions", *settings["conditions"],
            ]
            if settings["verify_hashes"]:
                command.append("--verify-image-hashes")
            subprocess.run(command, cwd=ROOT, check=True)
            outputs[name] = load(output)

    reference_predictions = true_predictions(outputs["reference"])
    comparisons = {}
    for name in ("repeat", "single_family_batch", "reversed_candidates", "reversed_manifest"):
        candidate_predictions = true_predictions(outputs[name])
        mismatches = sorted(
            family
            for family in set(reference_predictions) | set(candidate_predictions)
            if reference_predictions.get(family) != candidate_predictions.get(family)
        )
        comparisons[name] = {
            "exact_prediction_match": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatch_family_sample": mismatches[:20],
        }

    reference = outputs["reference"]
    random_audits = reference["random_control"]["audit"]
    gates = {
        "same_checkpoint_rerun_exact": comparisons["repeat"]["exact_prediction_match"],
        "batch_1_vs_batch_reference_exact": comparisons["single_family_batch"]["exact_prediction_match"],
        "candidate_order_swap_exact": comparisons["reversed_candidates"]["exact_prediction_match"],
        "manifest_order_swap_exact": comparisons["reversed_manifest"]["exact_prediction_match"],
        "all_answer_spans_nonempty": reference["candidate_answer_span_audit"]["all_nonempty"],
        "all_checked_images_match_manifest": reference["image_integrity_audit"]["all_match_manifest"],
        "random_control_has_zero_self_matches": all(
            audit["self_matches"] == 0 for audit in random_audits.values()
        ),
        "random_control_is_same_task": all(
            audit["same_task_percent"] == 100.0 for audit in random_audits.values()
        ),
        "random_control_is_answer_stratified": all(
            audit["same_expected_answer_percent"] == 100.0
            for audit in random_audits.values()
        ),
    }
    receipt = {
        "schema_version": "2026-09-13-v1",
        "status": "pass" if all(gates.values()) else "fail_closed",
        "scope": "stage_d_development_subset_scorer_invariance_and_input_integrity",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "language_adapter_model_sha256": sha256_file(
            args.language_adapter / "adapter_model.safetensors"
        ),
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": sha256_file(args.manifest),
        "families_per_task": args.families_per_task,
        "scene_families": reference["scene_families"],
        "comparisons": comparisons,
        "gates": gates,
        "reference_metrics": reference["metric_contract"],
        "random_control": reference["random_control"],
        "candidate_answer_span_audit": reference["candidate_answer_span_audit"],
        "image_integrity_audit": reference["image_integrity_audit"],
        "component_outputs": {
            name: {"path": str(args.output_dir / f"{name}.json"), "sha256": sha256_file(args.output_dir / f"{name}.json")}
            for name in outputs
        },
        "training_prediction_tokens": 0,
        "prior_test_predictions_produced": 0,
        "claim_boundary": (
            "This validates deterministic candidate scoring and file/control integrity on a fixed "
            "development subset. It is not an efficacy comparison or a new held-out test result."
        ),
    }
    output = args.output_dir / "audit.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "gates": gates}, indent=2, sort_keys=True))
    if receipt["status"] != "pass":
        raise SystemExit("Stage-D evaluator audit failed closed")


if __name__ == "__main__":
    main()
