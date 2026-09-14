#!/usr/bin/env python3
"""Select a recovery checkpoint on a frozen development subset."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "stage_a_training_protocol.json")
    parser.add_argument("--arm")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families-per-task", type=int, default=30)
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 300, 500, 700, 820])
    parser.add_argument("--with-language-adapter", action="store_true")
    parser.add_argument("--result-prefix", default="stage-a-recovery-dev-select")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    manifest = args.manifest or Path(protocol["data"]["manifest"])
    evaluations = []
    for step in args.steps:
        checkpoint = args.run / f"step-{step:06d}" / "bridge.safetensors"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        result_path = ROOT / "evidence" / f"{args.result_prefix}-step-{step:06d}.json"
        if result_path.exists():
            raise FileExistsError(result_path)
        command = [
            sys.executable,
            str(ROOT / "evaluate_stage_a_visual_floor.py"),
            "--protocol", str(args.protocol),
            "--checkpoint", str(checkpoint),
            "--manifest", str(manifest),
            "--output", str(result_path),
            "--families-per-task", str(args.families_per_task),
        ]
        if args.arm is not None:
            command.extend(["--arm", args.arm])
        adapter = checkpoint.parent / "language_adapter"
        if args.with_language_adapter:
            if not (adapter / "adapter_model.safetensors").exists():
                raise FileNotFoundError(adapter / "adapter_model.safetensors")
            command.extend(["--language-adapter", str(adapter)])
        subprocess.run(command, check=True)
        result = json.loads(result_path.read_text())
        evaluations.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "language_adapter": None if not args.with_language_adapter else str(adapter),
                "language_adapter_model_sha256": None if not args.with_language_adapter else sha256_file(adapter / "adapter_model.safetensors"),
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "paired_joint_accuracy_percent": result["paired_joint_accuracy_percent"],
                "true_minus_no_image": result["true_minus_no_image"],
                "true_minus_random_image": result["true_minus_random_image"],
                "visual_floor": result["visual_floor"],
            }
        )
    best = max(
        evaluations,
        key=lambda item: (
            item["visual_floor"]["passes_all"],
            item["paired_joint_accuracy_percent"]["true_image"],
            item["true_minus_random_image"]["lower_95_ci_pp"],
            item["true_minus_no_image"]["lower_95_ci_pp"],
            item["step"],
        ),
    )
    receipt = {
        "schema_version": "2026-09-13-v1",
        "status": "complete",
        "completed_at": utc_now(),
        "selection_split": "development",
        "protocol_arm": args.arm,
        "families_per_task": args.families_per_task,
        "scene_families_per_checkpoint": args.families_per_task * 5,
        "evaluations": evaluations,
        "selected_step": best["step"],
        "selected_checkpoint": best["checkpoint"],
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "selected_language_adapter": best["language_adapter"],
        "selected_language_adapter_model_sha256": best["language_adapter_model_sha256"],
        "next_action": "evaluate the selected checkpoint on the complete development split",
        "test_split_predictions_produced": 0,
        "claim_boundary": "This deterministic development subset selects a checkpoint only; it is not a full development gate or test result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, receipt)
    print(json.dumps({"selected_step": best["step"], "evaluations": evaluations}, indent=2), flush=True)


if __name__ == "__main__":
    main()
