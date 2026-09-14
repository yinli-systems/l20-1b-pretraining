#!/usr/bin/env python3
"""Run the frozen two-arm, five-seed Stage-D equal-update replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"COMMAND {json.dumps(command)}\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def binding_macro(result: dict) -> float:
    contract = result["metric_contract"]["by_task"]
    return sum(
        contract[task]["true_image"]["family_joint_accuracy_percent"]
        for task in ("color_binding", "shape_binding")
    ) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_stage_d_two_arm_five_seed_replication":
        raise SystemExit("Stage-D D1 replication protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("all Stage-D tests must remain sealed during replication")
    arms = tuple(protocol["arms"])
    seeds = protocol["allowed_seeds"]
    if set(arms) != {"F_matched_196", "A_spatial_49"} or len(seeds) != 5:
        raise RuntimeError("D1 requires exactly two declared arms and five seeds")
    execution_order = []
    for index, seed in enumerate(seeds):
        pair = arms if index % 2 == 0 else tuple(reversed(arms))
        execution_order.extend((arm, seed) for arm in pair)
    development_manifest = Path(protocol["development"]["manifest"])
    if sha256_file(development_manifest) != protocol["development"]["manifest_sha256"]:
        raise RuntimeError("Stage-D development manifest hash mismatch")
    for path, expected in protocol["required_audits"].items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"required Stage-D audit hash mismatch: {path}")
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_pass",
            "protocol_sha256": sha256_file(protocol_path),
            "trials": len(execution_order),
            "tests_sealed": True,
        }, indent=2, sort_keys=True))
        return

    evidence = ROOT / "evidence"
    status_path = evidence / "stage-d-d1-replication-status-v1.json"
    status = {
        "schema_version": "2026-09-13-v1",
        "status": "running",
        "started_at": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "execution_order": execution_order,
        "completed_trials": [],
        "active_trial": None,
        "active_stage": None,
        "training_gpu_seconds": 0.0,
        "prior_test_predictions_produced": 0,
        "new_iid_test_predictions_produced": 0,
        "ood_test_predictions_produced": 0,
    }
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if prior["protocol_sha256"] != status["protocol_sha256"]:
            raise RuntimeError("existing D1 status belongs to a different protocol")
        status = prior
        status["status"] = "running"
    write_json_atomic(status_path, status)

    results = []
    try:
        for arm, seed in execution_order:
            trial = f"{arm}-seed{seed}"
            run_dir = ROOT / "runs" / f"stage-d-d1-{trial}-v1"
            selection = evidence / f"stage-d-d1-{trial}-selection-v1.json"
            prune_receipt = evidence / f"stage-d-d1-{trial}-prune-v1.json"
            development = evidence / f"stage-d-d1-{trial}-development-v1.json"
            status.update({"active_trial": trial, "active_stage": "training", "updated_at": utc_now()})
            write_json_atomic(status_path, status)
            multiplier = protocol["arms"][arm]["selected_learning_rate_multiplier"]
            if not (run_dir / "run.json").exists():
                run_command([
                    sys.executable,
                    str(ROOT / "train_stage_c_counterfactual.py"),
                    "--protocol", str(protocol_path),
                    "--arm", arm,
                    "--seed", str(seed),
                    "--learning-rate-multiplier", str(multiplier),
                    "--output", str(run_dir),
                    "--num-workers", "4",
                ], ROOT / "logs" / f"stage-d-d1-{trial}-train-v1.log")
            run_state = json.loads((run_dir / "run.json").read_text())
            expected_steps = protocol["optimization"]["max_optimizer_steps"]
            if run_state.get("status") != "complete" or run_state.get("optimizer_step") != expected_steps:
                raise RuntimeError(f"incomplete D1 training: {trial}")
            accounted_trials = set(status["completed_trials"])
            accounted_trials.add(trial)
            completed_seconds = sum(
                json.loads(
                    (ROOT / "runs" / f"stage-d-d1-{done}-v1" / "run.json").read_text()
                )["wall_seconds"]
                for done in accounted_trials
                if (ROOT / "runs" / f"stage-d-d1-{done}-v1" / "run.json").exists()
            )
            if completed_seconds / 3600.0 > protocol["budget"]["d1_training_gpu_hours_cap"]:
                raise RuntimeError("D1 training GPU-hour cap exceeded")

            status.update({"active_stage": "checkpoint_selection", "updated_at": utc_now()})
            write_json_atomic(status_path, status)
            if not selection.exists():
                evaluations = []
                for step in protocol["selection"]["candidate_steps"]:
                    checkpoint = run_dir / f"step-{step:06d}"
                    output = evidence / f"stage-d-d1-{trial}-select-step-{step:06d}-v1.json"
                    if not output.exists():
                        run_command([
                            sys.executable,
                            str(ROOT / "evaluate_stage_a_visual_floor.py"),
                            "--protocol", str(protocol_path),
                            "--arm", arm,
                            "--checkpoint", str(checkpoint / "bridge.safetensors"),
                            "--language-adapter", str(checkpoint / "language_adapter"),
                            "--manifest", str(development_manifest),
                            "--split", "development",
                            "--families-per-task", str(protocol["selection"]["families_per_task"]),
                            "--batch-families", "8",
                            "--conditions", "true_image",
                            "--candidate-order", "manifest",
                            "--scoring-precision", "float32",
                            "--output", str(output),
                        ], ROOT / "logs" / f"stage-d-d1-{trial}-select-step-{step:06d}-v1.log")
                    result = json.loads(output.read_text())
                    metric = result["metric_contract"]["overall"]["true_image"]
                    evaluations.append({
                        "step": step,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint / "bridge.safetensors"),
                        "language_adapter_model_sha256": sha256_file(checkpoint / "language_adapter" / "adapter_model.safetensors"),
                        "family_joint_accuracy_percent": metric["family_joint_accuracy_percent"],
                        "ordinary_primary_query_accuracy_percent": metric["ordinary_primary_query_accuracy_percent"],
                        "binding_macro_percent": binding_macro(result),
                        "evaluation_wall_seconds": result["total_wall_seconds"],
                        "result": str(output),
                        "result_sha256": sha256_file(output),
                    })
                best = max(evaluations, key=lambda item: (
                    item["family_joint_accuracy_percent"],
                    item["binding_macro_percent"],
                    item["ordinary_primary_query_accuracy_percent"],
                    -item["step"],
                ))
                write_json_atomic(selection, {
                    "schema_version": "2026-09-13-v1",
                    "status": "complete",
                    "protocol_arm": arm,
                    "seed": seed,
                    "selection_split": "development",
                    "families_per_task": protocol["selection"]["families_per_task"],
                    "scene_families_per_checkpoint": protocol["selection"]["families_per_task"] * 5,
                    "selection_rule": protocol["selection"]["rule"],
                    "evaluations": evaluations,
                    "selected_step": best["step"],
                    "selected_checkpoint": best["checkpoint"],
                    "selected_checkpoint_sha256": best["checkpoint_sha256"],
                    "selected_language_adapter": str(Path(best["checkpoint"]) / "language_adapter"),
                    "selected_language_adapter_model_sha256": best["language_adapter_model_sha256"],
                    "test_split_predictions_produced": 0,
                })
            selected = json.loads(selection.read_text())

            status.update({"active_stage": "pruning", "updated_at": utc_now()})
            write_json_atomic(status_path, status)
            if not prune_receipt.exists():
                run_command([
                    sys.executable,
                    str(ROOT / "prune_completed_sweep_run.py"),
                    "--run", str(run_dir),
                    "--selection", str(selection),
                    "--output", str(prune_receipt),
                ], ROOT / "logs" / f"stage-d-d1-{trial}-prune-v1.log")

            status.update({"active_stage": "full_development_evaluation", "updated_at": utc_now()})
            write_json_atomic(status_path, status)
            checkpoint = Path(selected["selected_checkpoint"])
            if not development.exists():
                run_command([
                    sys.executable,
                    str(ROOT / "evaluate_stage_a_visual_floor.py"),
                    "--protocol", str(protocol_path),
                    "--arm", arm,
                    "--checkpoint", str(checkpoint / "bridge.safetensors"),
                    "--language-adapter", str(checkpoint / "language_adapter"),
                    "--manifest", str(development_manifest),
                    "--split", "development",
                    "--batch-families", "8",
                    "--conditions", "true_image", "random_image", "no_image",
                    "--random-control", "within_task_answer_matched",
                    "--candidate-order", "manifest",
                    "--scoring-precision", "float32",
                    "--verify-image-hashes",
                    "--output", str(development),
                ], ROOT / "logs" / f"stage-d-d1-{trial}-development-v1.log")
            evaluated = json.loads(development.read_text())
            selection_gpu_seconds = sum(
                item["evaluation_wall_seconds"] for item in selected["evaluations"]
            )
            result = {
                "trial": trial,
                "arm": arm,
                "seed": seed,
                "selected_step": selected["selected_step"],
                "learning_rate_multiplier": multiplier,
                "development_family_joint_accuracy_percent": evaluated["metric_contract"]["overall"]["true_image"]["family_joint_accuracy_percent"],
                "training_wall_seconds": run_state["wall_seconds"],
                "training_total_wall_seconds": run_state["total_wall_seconds"],
                "training_input_and_visual_tokens_per_second": run_state["total_input_and_visual_tokens_per_second"],
                "training_peak_allocated_gib": run_state["peak_allocated_gib"],
                "checkpoint_selection_gpu_seconds": selection_gpu_seconds,
                "development_evaluation_gpu_seconds": evaluated["total_wall_seconds"],
                "selection": str(selection),
                "selection_sha256": sha256_file(selection),
                "development": str(development),
                "development_sha256": sha256_file(development),
            }
            results.append(result)
            consumed_gpu_seconds = sum(
                item["training_total_wall_seconds"]
                + item["checkpoint_selection_gpu_seconds"]
                + item["development_evaluation_gpu_seconds"]
                for item in results
            )
            if consumed_gpu_seconds / 3600.0 > protocol["budget"]["d1_total_gpu_hours_cap"]:
                raise RuntimeError("D1 total GPU-hour cap exceeded")
            if trial not in status["completed_trials"]:
                status["completed_trials"].append(trial)
            status.update({
                "active_trial": None,
                "active_stage": None,
                "training_gpu_seconds": sum(item["training_total_wall_seconds"] for item in results),
                "total_gpu_seconds": consumed_gpu_seconds,
                "updated_at": utc_now(),
            })
            write_json_atomic(status_path, status)

        summary = {
            "schema_version": "2026-09-13-v1",
            "status": "complete_checkpoints_frozen_tests_still_sealed",
            "completed_at": utc_now(),
            "protocol_sha256": sha256_file(protocol_path),
            "regime": "equal_data_equal_875_optimizer_updates",
            "results": results,
            "training_gpu_hours": sum(item["training_total_wall_seconds"] for item in results) / 3600.0,
            "selection_and_development_evaluation_gpu_hours": sum(
                item["checkpoint_selection_gpu_seconds"] + item["development_evaluation_gpu_seconds"]
                for item in results
            ) / 3600.0,
            "total_gpu_hours": sum(
                item["training_total_wall_seconds"]
                + item["checkpoint_selection_gpu_seconds"]
                + item["development_evaluation_gpu_seconds"]
                for item in results
            ) / 3600.0,
            "prior_test_predictions_produced": 0,
            "new_iid_test_predictions_produced": 0,
            "ood_test_predictions_produced": 0,
            "next_gate": "freeze a confirmation-unseal receipt, then evaluate every frozen seed/arm once on new IID and OOD splits",
            "claim_boundary": "Development results select checkpoints; they are not confirmatory efficacy estimates.",
        }
        summary_path = evidence / "stage-d-d1-replication-summary-v1.json"
        if summary_path.exists():
            raise FileExistsError(summary_path)
        write_json_atomic(summary_path, summary)
        status.update({
            "status": "complete_checkpoints_frozen_tests_still_sealed",
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "completed_at": utc_now(),
        })
        write_json_atomic(status_path, status)
        print(json.dumps({"status": status["status"], "trials": len(results)}, indent=2))
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
