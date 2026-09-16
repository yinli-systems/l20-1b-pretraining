#!/usr/bin/env python3
"""Submit the seven frozen four-RTX-5090 new-pool pilots exactly once."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
INPUTS = ROOT / "source/new-pool-mixture-pilot-inputs-v1"
RUNNER = ROOT / "source/new-pool-mixture-pilots-v1/run.sbatch"
RECEIPT = ROOT / "receipts/new-pool-mixture-pilots-v1-submission.json"
FAILURE_RECEIPT = ROOT / "receipts/new-pool-mixture-pilots-v1-submit-attempt.json"
STORAGE_GATE = 50 * 1024**3
EXPECTED_STATUS = "PASS_NEW_POOL_INPUTS_ADMITTED_FOR_EXPLORATORY_PILOTS"
JOB_PREFIX = "p529m-np1-"
OWNER = "pretrain500m new-pool pilots; Codex task 44ca"
COMMAND_TIMEOUT_SECONDS = 20


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


def write_receipt(path: Path, status: str, jobs: list[dict], free: int, artifacts: dict, **extra) -> None:
    capacity = ""
    capacity_error = None
    try:
        capacity = run_command(["sinfo", "-h", "-p", "gpu_5090", "-o", "%P|%a|%D|%t|%G"]).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        capacity_error = f"{type(exc).__name__}: {exc}"
    receipt = {
        "schema": "p529m-new-pool-mixture-pilot-submission-v1",
        "status": status,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "owner": OWNER,
        "jobs": jobs,
        "world_size_per_job": 4,
        "submitted_gpu_total": 4 * len(jobs),
        "target_prediction_tokens_per_run": 536_870_912,
        "mfu_rule": "after five-step grace, every checked rolling ten-step median must be strictly greater than 0.70",
        "free_bytes_before_submit": free,
        "storage_gate_bytes": STORAGE_GATE,
        "storage_gate_passed": free >= STORAGE_GATE,
        "artifacts": artifacts,
        "capacity_snapshot": capacity,
        "capacity_snapshot_error": capacity_error,
        "training_started": False,
        "claim_boundary": "scheduler submission only; allocation, live GPUs, MFU, completion, capability and superiority remain unverified",
        **extra,
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if RECEIPT.exists():
        raise ValueError("successful or partial submission receipt already exists")
    build_path = INPUTS / "build-receipt.json"
    build = json.loads(build_path.read_text())
    if build.get("status") != EXPECTED_STATUS or build.get("training_launched") is not False:
        raise ValueError("new-pool inputs are not in the expected pre-launch state")
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free < STORAGE_GATE:
        raise ValueError(f"aggregate model-output storage gate failed: {free} < {STORAGE_GATE}")

    artifacts = {"runner": sha(RUNNER), "build_receipt": sha(build_path), "protocol": sha(INPUTS / "protocol.json")}
    for recipe, item in build["recipes"].items():
        manifest = INPUTS / "manifests" / f"{recipe}.json"
        admission = INPUTS / "admissions" / f"{recipe}.admission.json"
        if sha(manifest) != item["manifest_sha256"] or sha(admission) != item["admission_sha256"]:
            raise ValueError(f"generated input identity changed: {recipe}")
        artifacts[f"{recipe}_manifest"] = item["manifest_sha256"]
        artifacts[f"{recipe}_admission"] = item["admission_sha256"]

    try:
        queue = run_command(["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%j"]).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        write_receipt(
            FAILURE_RECEIPT,
            "SCHEDULER_PREFLIGHT_UNAVAILABLE_NO_SUBMISSION",
            [],
            free,
            artifacts,
            scheduler_error=f"{type(exc).__name__}: {exc}",
        )
        raise
    active = [line for line in queue.splitlines() if line.split("|")[-1].startswith(JOB_PREFIX)]
    if active:
        raise ValueError("active new-pool job already exists: " + ";".join(active))

    jobs: list[dict] = []
    for arm in build["arms"]:
        name = JOB_PREFIX + arm["arm_id"].replace("lr", "l")
        exports = ",".join([
            "ALL", f"RECIPE={arm['recipe']}", f"ARM_ID={arm['arm_id']}",
            f"PEAK_LR={arm['peak_learning_rate']}", "RUN_SEED=20260916",
        ])
        command = [
            "sbatch", "--parsable", "--partition=gpu_5090", "--qos=gpugpu",
            f"--job-name={name}", f"--export={exports}", str(RUNNER),
        ]
        last_error = None
        for attempt in range(3):
            try:
                result = run_command(command, check=False)
                if result.returncode == 0:
                    job_id = int(result.stdout.strip().split(";")[0])
                    jobs.append({
                        **arm, "job_id": job_id, "job_name": name,
                        "seed": 20260916, "submit_attempt": attempt + 1,
                    })
                    break
                last_error = result.stderr.strip() or result.stdout.strip()
            except (subprocess.SubprocessError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
        else:
            write_receipt(
                RECEIPT,
                "PARTIAL_SUBMISSION_REQUIRES_RECOVERY" if jobs else "SCHEDULER_SUBMISSION_UNAVAILABLE",
                jobs,
                free,
                artifacts,
                scheduler_error=last_error,
                failed_arm=arm,
            )
            raise RuntimeError(last_error)

    write_receipt(RECEIPT, "SUBMITTED_PENDING_ALLOCATION", jobs, free, artifacts)
    print(json.dumps({"status": "SUBMITTED_PENDING_ALLOCATION", "jobs": jobs}, sort_keys=True))


if __name__ == "__main__":
    main()
