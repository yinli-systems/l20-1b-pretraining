#!/usr/bin/env python3
"""Run, select, prune, and evaluate the frozen development-only token sweep."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
DEFAULT_ORDER = [
    "native_196",
    "spatial_196",
    "spatial_49",
    "spatial_100",
    "spatial_25",
    "spatial_144",
    "spatial_16",
    "spatial_64",
    "spatial_9",
    "spatial_36",
    "spatial_4",
    "spatial_1",
]


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\nCOMMAND {json.dumps(command)}\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def update_status(path: Path, status: dict, **updates) -> None:
    status.update(updates)
    status["updated_at"] = utc_now()
    write_json_atomic(path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "visual_token_scaling_protocol.json")
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ORDER)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_visual_token_scaling_sweep":
        raise SystemExit("sweep protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("sweep must not access the prior test split")
    if len(args.arms) != len(set(args.arms)):
        raise SystemExit("duplicate sweep arms")
    unknown = sorted(set(args.arms) - set(protocol["arms"]))
    if unknown:
        raise SystemExit(f"unknown sweep arms: {unknown}")

    evidence = ROOT / "evidence"
    status_path = evidence / "visual-token-sweep-status-v1.json"
    status = {
        "schema_version": "2026-09-13-v1",
        "status": "running",
        "started_at": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "orchestrator_sha256": sha256_file(Path(__file__)),
        "arms": args.arms,
        "completed_arms": [],
        "active_arm": None,
        "active_stage": None,
        "test_split_predictions_produced": 0,
    }
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if prior.get("protocol_sha256") != status["protocol_sha256"] or prior.get("arms") != args.arms:
            raise RuntimeError("existing status belongs to a different frozen sweep")
        status = prior
        status["status"] = "running"
    update_status(status_path, status)
    try:
        for arm in args.arms:
            run_dir = ROOT / "runs" / f"visual-token-sweep-{arm}-seed20260921-v1"
            selection = evidence / f"visual-token-sweep-{arm}-selection-v1.json"
            prune_receipt = evidence / f"visual-token-sweep-{arm}-prune-v1.json"
            development = evidence / f"visual-token-sweep-{arm}-development-v1.json"
            if development.exists() and prune_receipt.exists():
                if arm not in status["completed_arms"]:
                    status["completed_arms"].append(arm)
                update_status(status_path, status, active_arm=None, active_stage=None)
                continue

            update_status(status_path, status, active_arm=arm, active_stage="training")
            run_receipt = run_dir / "run.json"
            if not run_receipt.exists():
                run_command([
                    sys.executable,
                    str(ROOT / "train_stage_c_counterfactual.py"),
                    "--protocol", str(protocol_path),
                    "--arm", arm,
                    "--output", str(run_dir),
                    "--num-workers", "4",
                ], ROOT / "logs" / f"visual-token-sweep-{arm}-train-v1.log")
            run_state = json.loads(run_receipt.read_text())
            if run_state.get("status") != "complete" or run_state.get("optimizer_step") != 875:
                raise RuntimeError(f"{arm} training is not complete")

            update_status(status_path, status, active_stage="checkpoint_selection")
            if not selection.exists():
                run_command([
                    sys.executable,
                    str(ROOT / "select_stage_a_recovery_checkpoint.py"),
                    "--run", str(run_dir),
                    "--protocol", str(protocol_path),
                    "--arm", arm,
                    "--output", str(selection),
                    "--families-per-task", "50",
                    "--steps", "100", "300", "500", "700", "875",
                    "--with-language-adapter",
                    "--result-prefix", f"visual-token-sweep-{arm}-dev-select",
                ], ROOT / "logs" / f"visual-token-sweep-{arm}-selection-v1.log")
            selected = json.loads(selection.read_text())

            update_status(status_path, status, active_stage="pruning_nonselected_checkpoints")
            if not prune_receipt.exists():
                run_command([
                    sys.executable,
                    str(ROOT / "prune_completed_sweep_run.py"),
                    "--run", str(run_dir),
                    "--selection", str(selection),
                    "--output", str(prune_receipt),
                ], ROOT / "logs" / f"visual-token-sweep-{arm}-prune-v1.log")

            update_status(status_path, status, active_stage="full_development_evaluation")
            if not development.exists():
                selected_checkpoint = Path(selected["selected_checkpoint"])
                run_command([
                    sys.executable,
                    str(ROOT / "evaluate_stage_a_visual_floor.py"),
                    "--protocol", str(protocol_path),
                    "--arm", arm,
                    "--checkpoint", str(selected_checkpoint / "bridge.safetensors"),
                    "--language-adapter", str(selected_checkpoint / "language_adapter"),
                    "--manifest", protocol["data"]["manifest"],
                    "--split", "development",
                    "--output", str(development),
                    "--batch-families", "16",
                ], ROOT / "logs" / f"visual-token-sweep-{arm}-development-v1.log")
            if arm not in status["completed_arms"]:
                status["completed_arms"].append(arm)
            update_status(status_path, status, active_arm=None, active_stage=None)

        results = []
        for arm in args.arms:
            development = evidence / f"visual-token-sweep-{arm}-development-v1.json"
            selection = evidence / f"visual-token-sweep-{arm}-selection-v1.json"
            run_receipt = ROOT / "runs" / f"visual-token-sweep-{arm}-seed20260921-v1" / "run.json"
            result = json.loads(development.read_text())
            selected = json.loads(selection.read_text())
            run_state = json.loads(run_receipt.read_text())
            results.append({
                "arm": arm,
                "output_visual_tokens": result["output_visual_tokens"],
                "selected_step": selected["selected_step"],
                "true_image_paired_joint_accuracy_percent": result["paired_joint_accuracy_percent"]["true_image"],
                "random_image_paired_joint_accuracy_percent": result["paired_joint_accuracy_percent"]["random_image"],
                "no_image_paired_joint_accuracy_percent": result["paired_joint_accuracy_percent"]["no_image"],
                "true_minus_random_lower_95_ci_pp": result["true_minus_random_image"]["lower_95_ci_pp"],
                "training_wall_seconds": run_state["wall_seconds"],
                "training_input_and_visual_tokens_per_second": run_state["total_input_and_visual_tokens_per_second"],
                "development_receipt": str(development),
                "development_receipt_sha256": sha256_file(development),
                "selection_receipt": str(selection),
                "selection_receipt_sha256": sha256_file(selection),
            })
        ranked = sorted(results, key=lambda item: (-item["true_image_paired_joint_accuracy_percent"], item["output_visual_tokens"], item["arm"]))
        summary = {
            "schema_version": "2026-09-13-v1",
            "status": "complete",
            "completed_at": utc_now(),
            "protocol_sha256": sha256_file(protocol_path),
            "selection_split": "development",
            "results": results,
            "ranking": [item["arm"] for item in ranked],
            "selected_exploratory_arm": ranked[0]["arm"],
            "selected_exploratory_accuracy_percent": ranked[0]["true_image_paired_joint_accuracy_percent"],
            "test_split_predictions_produced": 0,
            "next_gate": "freeze the selected K and compare it once against matched 196-token controls on a newly generated untouched confirmation holdout",
            "claim_boundary": "Development sweep is exploratory and cannot establish a confirmatory compression-induced gain.",
        }
        summary_path = evidence / "visual-token-sweep-development-summary-v1.json"
        if summary_path.exists():
            raise FileExistsError(summary_path)
        write_json_atomic(summary_path, summary)
        update_status(
            status_path,
            status,
            status="complete",
            active_arm=None,
            active_stage=None,
            summary=str(summary_path),
            summary_sha256=sha256_file(summary_path),
            completed_at=utc_now(),
        )
        print(json.dumps({
            "status": "complete",
            "selected_exploratory_arm": summary["selected_exploratory_arm"],
            "selected_exploratory_accuracy_percent": summary["selected_exploratory_accuracy_percent"],
            "test_split_predictions_produced": 0,
        }, indent=2))
    except Exception as error:
        update_status(
            status_path,
            status,
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
