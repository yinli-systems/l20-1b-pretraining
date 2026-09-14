#!/usr/bin/env python3
"""Run the frozen one-time Stage-D IID/OOD confirmation matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from analyze_stage_d_confirmation import (
    evaluation_key,
    evaluation_path,
    load_confirmation_result,
    load_manifest,
)
from stage_d_confirmation_contract import sha256_file
from train_stage_a_full_token import utc_now, write_json_atomic
from validate_stage_d_confirmation_protocol import validate_protocol


ROOT = Path(__file__).resolve().parent


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
    validation = validate_protocol(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    manifest_by_id, split_ids = load_manifest(protocol)
    if args.preflight_only:
        existing = [str(evaluation_path(item)) for item in protocol["execution_order"] if evaluation_path(item).exists()]
        if existing:
            raise RuntimeError(f"confirmation outputs already exist before first launch: {existing}")
        print(json.dumps({**validation, "test_outputs_present": 0}, indent=2, sort_keys=True))
        return

    status_path = ROOT / "evidence" / "stage-d-confirmation-status-v1.json"
    summary_path = ROOT / "evidence" / "stage-d-confirmation-summary-v1.json"
    if summary_path.exists():
        raise RuntimeError("confirmation summary already exists; refusing to rerun tests")
    protocol_sha = sha256_file(protocol_path)
    status = {
        "schema_version": "2026-09-13-v1",
        "status": "running_tests_unsealed",
        "started_at": utc_now(),
        "tests_unsealed_at": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "expected_evaluations": protocol["expected_evaluations"],
        "completed_evaluations": [],
        "active_evaluation": None,
        "evaluation_gpu_seconds": 0.0,
        "failures": [],
    }
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if prior.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("existing confirmation status belongs to another protocol")
        if prior.get("failures"):
            raise RuntimeError("a confirmation evaluation failed; automatic retry is prohibited")
        active = prior.get("active_evaluation")
        completed = set(prior.get("completed_evaluations", []))
        if active and active not in completed:
            active_item = next(item for item in protocol["execution_order"] if evaluation_key(item) == active)
            if not evaluation_path(active_item).exists():
                raise RuntimeError("ambiguous partial test inference; automatic retry is prohibited")
        status = prior
        status["status"] = "running_tests_unsealed"
    write_json_atomic(status_path, status)

    try:
        for item in protocol["execution_order"]:
            key = evaluation_key(item)
            output = evaluation_path(item)
            status.update({"active_evaluation": key, "active_stage": "confirmation_evaluation", "updated_at": utc_now()})
            write_json_atomic(status_path, status)
            arm, seed, split = item["arm"], int(item["seed"]), item["split"]
            checkpoint = protocol["checkpoints"][arm][str(seed)]
            if not output.exists():
                run_command([
                    sys.executable,
                    str(ROOT / "evaluate_stage_a_visual_floor.py"),
                    "--protocol", protocol["source_files"]["replication_protocol"]["path"],
                    "--arm", arm,
                    "--checkpoint", checkpoint["bridge"],
                    "--language-adapter", checkpoint["adapter"],
                    "--manifest", protocol["manifest"]["path"],
                    "--split", split,
                    "--batch-families", "8",
                    "--conditions", *protocol["conditions"],
                    "--random-control", protocol["random_control"],
                    "--candidate-order", protocol["candidate_order"],
                    "--scoring-precision", protocol["scoring_precision"],
                    "--verify-image-hashes",
                    "--output", str(output),
                ], ROOT / "logs" / f"stage-d-confirmation-{key}-v1.log")
            result = load_confirmation_result(
                protocol, item, manifest_by_id, split_ids[split]
            )
            completed = list(status["completed_evaluations"])
            if key not in completed:
                completed.append(key)
            result_seconds = {
                record["key"]: float(record["wall_seconds"])
                for record in status.get("result_records", [])
            }
            result_seconds[key] = result["wall_seconds"]
            result_records = [
                {
                    "key": name,
                    "path": str(evaluation_path(next(
                        candidate for candidate in protocol["execution_order"] if evaluation_key(candidate) == name
                    ))),
                    "sha256": sha256_file(evaluation_path(next(
                        candidate for candidate in protocol["execution_order"] if evaluation_key(candidate) == name
                    ))),
                    "wall_seconds": seconds,
                }
                for name, seconds in result_seconds.items()
            ]
            consumed = sum(record["wall_seconds"] for record in result_records)
            if consumed / 3600.0 > protocol["budget"]["confirmation_l20_gpu_hours_cap"]:
                raise RuntimeError("confirmation GPU-hour cap exceeded")
            status.update({
                "completed_evaluations": completed,
                "result_records": result_records,
                "active_evaluation": None,
                "active_stage": None,
                "evaluation_gpu_seconds": consumed,
                "iid_test_evaluations_completed": sum(name.endswith("-iid_test") for name in completed),
                "ood_binding_evaluations_completed": sum(name.endswith("-ood_binding") for name in completed),
                "updated_at": utc_now(),
            })
            write_json_atomic(status_path, status)

        status.update({"active_stage": "analysis", "updated_at": utc_now()})
        write_json_atomic(status_path, status)
        run_command([
            sys.executable,
            str(ROOT / "analyze_stage_d_confirmation.py"),
            "--protocol", str(protocol_path),
            "--output", str(summary_path),
        ], ROOT / "logs" / "stage-d-confirmation-analysis-v1.log")
        summary = json.loads(summary_path.read_text())
        status.update({
            "status": "complete_confirmation_results_frozen",
            "active_evaluation": None,
            "active_stage": None,
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "completed_at": utc_now(),
            "updated_at": utc_now(),
        })
        write_json_atomic(status_path, status)
        print(json.dumps({
            "status": status["status"],
            "completed_evaluations": len(status["completed_evaluations"]),
            "summary": str(summary_path),
            "summary_sha256": status["summary_sha256"],
            "primary_decision": summary["primary_decision"],
        }, indent=2, sort_keys=True))
    except Exception as error:
        status.setdefault("failures", []).append({
            "evaluation": status.get("active_evaluation"),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "at": utc_now(),
        })
        status.update({"status": "failed_no_automatic_retry", "updated_at": utc_now()})
        write_json_atomic(status_path, status)
        raise


if __name__ == "__main__":
    main()
