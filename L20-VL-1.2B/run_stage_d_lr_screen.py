#!/usr/bin/env python3
"""Run the frozen two-arm, three-rate Stage-D development screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent


def tag(multiplier: float) -> str:
    return str(multiplier).replace(".", "p")


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"COMMAND {json.dumps(command)}\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_stage_d_lr_microsearch":
        raise SystemExit("Stage-D LR-screen protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("Stage-D tests must remain sealed during LR selection")
    order = [tuple(item) for item in protocol["screen_order"]]
    expected = {
        (arm, float(multiplier))
        for arm in protocol["arms"]
        for multiplier in protocol["allowed_learning_rate_multipliers"]
    }
    if set(order) != expected or len(order) != len(expected):
        raise RuntimeError("screen order must contain every arm/rate exactly once")
    development_manifest = Path(protocol["development"]["manifest"])
    if sha256_file(development_manifest) != protocol["development"]["manifest_sha256"]:
        raise RuntimeError("Stage-D development manifest hash mismatch")
    for path, expected in protocol["required_audits"].items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"required Stage-D audit hash mismatch: {path}")
    smoke_path = Path(protocol["training_smoke_receipt"])
    smoke = json.loads(smoke_path.read_text())
    if (
        smoke.get("status") != "pass"
        or smoke.get("protocol_sha256") != sha256_file(protocol_path)
        or smoke.get("smoke_directories_removed") is not True
    ):
        raise RuntimeError("Stage-D training smoke is missing or does not bind this protocol")
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_pass",
            "protocol_sha256": sha256_file(protocol_path),
            "trials": len(order),
            "tests_sealed": True,
        }, indent=2, sort_keys=True))
        return

    evidence = ROOT / "evidence"
    status_path = evidence / "stage-d-lr-screen-status-v1.json"
    status = {
        "schema_version": "2026-09-13-v1",
        "status": "running",
        "started_at": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "screen_order": order,
        "completed_trials": [],
        "active_trial": None,
        "prior_test_predictions_produced": 0,
        "new_iid_test_predictions_produced": 0,
        "ood_test_predictions_produced": 0,
    }
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if prior["protocol_sha256"] != status["protocol_sha256"]:
            raise RuntimeError("existing LR-screen status belongs to another protocol")
        status = prior
        status["status"] = "running"
    write_json_atomic(status_path, status)

    results = []
    try:
        for arm, multiplier in order:
            trial = f"{arm}-lr{tag(multiplier)}"
            run_dir = ROOT / "runs" / f"stage-d-lr-screen-{trial}-v1"
            evaluation = evidence / f"stage-d-lr-screen-{trial}-development-v1.json"
            selection = evidence / f"stage-d-lr-screen-{trial}-selection-v1.json"
            prune_receipt = evidence / f"stage-d-lr-screen-{trial}-prune-v1.json"
            status["active_trial"] = trial
            status["updated_at"] = utc_now()
            write_json_atomic(status_path, status)
            if not (run_dir / "run.json").exists():
                run_command([
                    sys.executable,
                    str(ROOT / "train_stage_c_counterfactual.py"),
                    "--protocol", str(protocol_path),
                    "--arm", arm,
                    "--seed", str(protocol["optimization"]["seed"]),
                    "--learning-rate-multiplier", str(multiplier),
                    "--max-optimizer-steps", str(protocol["screen_optimizer_steps"]),
                    "--output", str(run_dir),
                    "--num-workers", "4",
                ], ROOT / "logs" / f"stage-d-lr-screen-{trial}-train-v1.log")
            run_state = json.loads((run_dir / "run.json").read_text())
            if run_state.get("status") != "complete" or run_state.get("optimizer_step") != protocol["screen_optimizer_steps"]:
                raise RuntimeError(f"incomplete LR trial: {trial}")
            accounted_trials = set(status["completed_trials"])
            accounted_trials.add(trial)
            completed_seconds = sum(
                json.loads(
                    (ROOT / "runs" / f"stage-d-lr-screen-{done}-v1" / "run.json").read_text()
                )["wall_seconds"]
                for done in accounted_trials
                if (ROOT / "runs" / f"stage-d-lr-screen-{done}-v1" / "run.json").exists()
            )
            if completed_seconds / 3600.0 > protocol["budget"]["lr_screen_training_gpu_hours_cap"]:
                raise RuntimeError("Stage-D LR-screen training GPU-hour cap exceeded")
            checkpoint = run_dir / f"step-{protocol['screen_optimizer_steps']:06d}"
            if not evaluation.exists():
                run_command([
                    sys.executable,
                    str(ROOT / "evaluate_stage_a_visual_floor.py"),
                    "--protocol", str(protocol_path),
                    "--arm", arm,
                    "--checkpoint", str(checkpoint / "bridge.safetensors"),
                    "--language-adapter", str(checkpoint / "language_adapter"),
                    "--manifest", str(development_manifest),
                    "--split", "development",
                    "--families-per-task", str(protocol["development"]["families_per_task_for_screen"]),
                    "--batch-families", "8",
                    "--conditions", "true_image",
                    "--candidate-order", "manifest",
                    "--scoring-precision", "float32",
                    "--verify-image-hashes",
                    "--output", str(evaluation),
                ], ROOT / "logs" / f"stage-d-lr-screen-{trial}-eval-v1.log")
            evaluated = json.loads(evaluation.read_text())
            metric = evaluated["metric_contract"]["overall"]["true_image"]
            if not selection.exists():
                write_json_atomic(selection, {
                    "schema_version": "2026-09-13-v1",
                    "status": "complete",
                    "protocol_arm": arm,
                    "selected_step": protocol["screen_optimizer_steps"],
                    "selected_checkpoint": str(checkpoint),
                    "selected_checkpoint_sha256": sha256_file(checkpoint / "bridge.safetensors"),
                    "selected_language_adapter": str(checkpoint / "language_adapter"),
                    "selected_language_adapter_model_sha256": sha256_file(checkpoint / "language_adapter" / "adapter_model.safetensors"),
                    "selection_split": "development",
                    "scene_families_per_checkpoint": evaluated["scene_families"],
                    "evaluations": [{"step": protocol["screen_optimizer_steps"]}],
                    "test_split_predictions_produced": 0,
                })
            if not prune_receipt.exists():
                run_command([
                    sys.executable,
                    str(ROOT / "prune_completed_sweep_run.py"),
                    "--run", str(run_dir),
                    "--selection", str(selection),
                    "--output", str(prune_receipt),
                    "--expected-optimizer-steps", str(protocol["screen_optimizer_steps"]),
                ], ROOT / "logs" / f"stage-d-lr-screen-{trial}-prune-v1.log")
            result = {
                "trial": trial,
                "arm": arm,
                "learning_rate_multiplier": multiplier,
                "seed": run_state["seed"],
                "optimizer_steps": run_state["optimizer_step"],
                "family_joint_accuracy_percent": metric["family_joint_accuracy_percent"],
                "ordinary_primary_query_accuracy_percent": metric["ordinary_primary_query_accuracy_percent"],
                "training_wall_seconds": run_state["wall_seconds"],
                "training_input_and_visual_tokens_per_second": run_state["total_input_and_visual_tokens_per_second"],
                "training_peak_allocated_gib": run_state["peak_allocated_gib"],
                "evaluation": str(evaluation),
                "evaluation_sha256": sha256_file(evaluation),
            }
            results.append(result)
            if trial not in status["completed_trials"]:
                status["completed_trials"].append(trial)
            status["updated_at"] = utc_now()
            write_json_atomic(status_path, status)

        selected = {}
        for arm in protocol["arms"]:
            candidates = [item for item in results if item["arm"] == arm]
            best = max(candidates, key=lambda item: (
                item["family_joint_accuracy_percent"],
                item["ordinary_primary_query_accuracy_percent"],
                -abs(item["learning_rate_multiplier"] - 1.0),
                -item["learning_rate_multiplier"],
            ))
            selected[arm] = best["learning_rate_multiplier"]
        summary = {
            "schema_version": "2026-09-13-v1",
            "status": "complete",
            "completed_at": utc_now(),
            "protocol_sha256": sha256_file(protocol_path),
            "selection_split": "development",
            "selection_rule": protocol["selection_rule"],
            "results": results,
            "selected_learning_rate_multipliers": selected,
            "total_training_gpu_seconds": sum(item["training_wall_seconds"] for item in results),
            "prior_test_predictions_produced": 0,
            "new_iid_test_predictions_produced": 0,
            "ood_test_predictions_produced": 0,
            "next_action": "freeze the two-arm five-seed D1 protocol with these development-selected multipliers",
            "claim_boundary": "A one-seed 150-step development LR screen selects optimization settings only; it is not an efficacy result.",
        }
        summary_path = evidence / "stage-d-lr-screen-summary-v1.json"
        if summary_path.exists():
            raise FileExistsError(summary_path)
        write_json_atomic(summary_path, summary)
        status.update({
            "status": "complete",
            "active_trial": None,
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "completed_at": utc_now(),
        })
        write_json_atomic(status_path, status)
        print(json.dumps({"status": "complete", "selected": selected}, indent=2))
    except Exception as error:
        status.update({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "updated_at": utc_now(),
        })
        write_json_atomic(status_path, status)
        raise


if __name__ == "__main__":
    main()
