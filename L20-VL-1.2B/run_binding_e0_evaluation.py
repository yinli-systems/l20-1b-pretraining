#!/usr/bin/env python3
"""Run the frozen binding-E0 development evaluation sequence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def result_metrics(path: Path) -> dict[str, float]:
    result = json.loads(path.read_text())
    binding = result["binding_selective_metrics"]["per_condition"]["true_image"]
    return {
        "affected_question_joint_accuracy_percent": binding[
            "affected_question_joint_accuracy_percent"
        ],
        "scene_pair_selective_all_six_correct_percent": binding[
            "scene_pair_selective_all_six_correct_percent"
        ],
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
    reuse_existing: bool,
) -> None:
    if output.exists():
        if not reuse_existing:
            raise FileExistsError(output)
        existing = json.loads(output.read_text())
        expected = {
            "protocol_sha256": sha256_file(protocol_path),
            "manifest_sha256": sha256_file(manifest),
            "checkpoint_sha256": sha256_file(checkpoint),
            "split": split,
        }
        actual = {name: existing.get(name) for name in expected}
        if actual != expected:
            raise RuntimeError(
                f"existing evaluation artifact does not match resumed request: {actual} != {expected}"
            )
        expected_families = None if scene_pairs is None else scene_pairs * 6
        if expected_families is not None and existing.get("scene_families") != expected_families:
            raise RuntimeError("existing evaluation scene-pair selection mismatch")
        return
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_binding_answer_only_development_v1":
        raise SystemExit("evaluation protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    sources = protocol["source_code"]
    source_paths = {
        "orchestrator_sha256": Path(__file__).resolve(),
        "evaluator_sha256": root / "evaluate_stage_a_visual_floor.py",
        "scoring_contract_sha256": root / "scoring_contract.py",
        "test_sha256": root / "test_counterfactual_evaluation.py",
    }
    for label, path in source_paths.items():
        if sha256_file(path) != sources[label]:
            raise SystemExit(f"evaluation source hash mismatch: {label}")
    training_protocol_path = Path(protocol["training_protocol"]["path"])
    if sha256_file(training_protocol_path) != protocol["training_protocol"]["sha256"]:
        raise SystemExit("training protocol hash mismatch")
    training_protocol = json.loads(training_protocol_path.read_text())
    run_root = Path(training_protocol["resource_cap"]["full_run_output"])
    run_receipt_path = run_root / "run.json"
    run_receipt = json.loads(run_receipt_path.read_text())
    if run_receipt.get("status") != "complete":
        raise SystemExit("training run is not complete")
    if run_receipt.get("protocol_sha256") != protocol["training_protocol"]["sha256"]:
        raise SystemExit("training receipt does not bind the frozen protocol")
    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("evaluation manifest hash mismatch")

    screen = protocol["checkpoint_selection"]
    candidates = [
        {
            "step": 0,
            "label": "frozen_parent",
            "checkpoint": Path(training_protocol["parents"]["bridge"]),
            "adapter": Path(training_protocol["parents"]["language_adapter"]),
        }
    ]
    for step in screen["screen_checkpoints"]:
        checkpoint_dir = run_root / f"step-{step:06d}"
        candidates.append({
            "step": step,
            "label": f"step_{step:06d}",
            "checkpoint": checkpoint_dir / "bridge.safetensors",
            "adapter": checkpoint_dir / "language_adapter",
        })
    conditions = protocol["scoring"]["conditions"]
    random_control = protocol["scoring"]["random_control"]
    screens = []
    for candidate in candidates:
        output = args.output_dir / f"screen-{candidate['label']}.json"
        run_evaluator(
            root,
            protocol_path,
            manifest,
            candidate["checkpoint"],
            candidate["adapter"],
            output,
            screen["split"],
            screen["screen_scene_pairs"],
            random_control,
            conditions,
            args.resume,
        )
        screens.append({
            "step": candidate["step"],
            "label": candidate["label"],
            "result": str(output),
            "result_sha256": sha256_file(output),
            **result_metrics(output),
        })
    best = sorted(
        screens,
        key=lambda item: (
            -item["affected_question_joint_accuracy_percent"],
            -item["scene_pair_selective_all_six_correct_percent"],
            item["step"],
        ),
    )[0]
    best_candidate = next(candidate for candidate in candidates if candidate["step"] == best["step"])
    full_mechanism = args.output_dir / f"full-mechanism-{best_candidate['label']}.json"
    run_evaluator(
        root,
        protocol_path,
        manifest,
        best_candidate["checkpoint"],
        best_candidate["adapter"],
        full_mechanism,
        "mechanism_dev",
        None,
        random_control,
        conditions,
        args.resume,
    )
    selection = args.output_dir / f"selection-{best_candidate['label']}.json"
    run_evaluator(
        root,
        protocol_path,
        manifest,
        best_candidate["checkpoint"],
        best_candidate["adapter"],
        selection,
        "selection_dev",
        None,
        random_control,
        conditions,
        args.resume,
    )
    write_json_atomic(args.output_dir / "summary.json", {
        "schema_version": "2026-09-14-v1",
        "status": "complete_development_only",
        "evaluation_protocol_sha256": sha256_file(protocol_path),
        "training_protocol_sha256": protocol["training_protocol"]["sha256"],
        "training_run_receipt_sha256": sha256_file(run_receipt_path),
        "screens": screens,
        "selected": best,
        "full_mechanism": {
            "result": str(full_mechanism),
            "result_sha256": sha256_file(full_mechanism),
            **result_metrics(full_mechanism),
        },
        "selection_dev": {
            "result": str(selection),
            "result_sha256": sha256_file(selection),
            **result_metrics(selection),
        },
        "final_test_used": False,
        "claim_boundary": protocol["claim_boundary"],
    })


if __name__ == "__main__":
    main()
