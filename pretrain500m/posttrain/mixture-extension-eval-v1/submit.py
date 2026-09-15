#!/usr/bin/env python3
"""Submit two independent two-GPU R4 extension evaluations exactly once."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
SOURCE = ROOT / "source/mixture-extension-eval-v1"
PLAN = ROOT / "source/mixture-extension-eval-inputs-v1/candidates.json"
RUNNER = SOURCE / "run.sbatch"
RECEIPT = ROOT / "receipts/mixture-extension-eval-v1-submission.json"
JOB_PREFIX = "p529m-r4eval-"
STORAGE_GATE = 10 * 1024**3


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write_receipt(status: str, jobs: list[dict], free: int, **extra) -> None:
    capacity = subprocess.run(
        ["sinfo", "-p", "gpu_5090"], text=True, capture_output=True
    ).stdout
    document = {
        "schema": "p529m-r4-extension-capability-submission-v1",
        "status": status,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "jobs": jobs,
        "world_size_per_job": 2,
        "submitted_gpu_total": 2 * len(jobs),
        "plan_sha256": digest(PLAN),
        "runner_sha256": digest(RUNNER),
        "free_bytes_before_submit": free,
        "storage_gate_bytes": STORAGE_GATE,
        "capacity_snapshot": capacity,
        "formal_promotion": False,
        "claim_boundary": "submitted adaptive R4 two-seed seven-task capability screens; allocation, completion and candidate quality remain unverified",
        **extra,
    }
    RECEIPT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if RECEIPT.exists():
        raise ValueError("submission receipt already exists")
    plan = json.loads(PLAN.read_text())
    if plan.get("status") != "FROZEN_BEFORE_EXTENSION_CAPABILITY_SCREEN":
        raise ValueError("evaluation plan is not frozen")
    build_receipt_path = PLAN.parent / "build-receipt.json"
    build_receipt = json.loads(build_receipt_path.read_text())
    if build_receipt.get("status") != "PASS_TWO_VERIFIED_EXTENSION_CHECKPOINTS_FROZEN_FOR_EVALUATION":
        raise ValueError("extension capability plan build did not pass")
    if build_receipt.get("plan_sha256") != digest(PLAN) or build_receipt.get("evaluation_launched") is not False:
        raise ValueError("capability plan hash or pre-launch state differs")
    candidates = plan.get("candidates", [])
    if len(candidates) != 2 or len({item["id"] for item in candidates}) != 2:
        raise ValueError("evaluation plan must contain two unique candidates")
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free < STORAGE_GATE:
        raise ValueError(f"evaluation storage gate failed: {free} < {STORAGE_GATE}")
    queue = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%j"],
        check=True, text=True, capture_output=True,
    ).stdout
    active = [line for line in queue.splitlines() if line.split("|")[-1].startswith(JOB_PREFIX)]
    if active:
        raise ValueError("active capability-screen job already exists: " + ";".join(active))

    plan_sha = digest(PLAN)
    jobs = []
    for item in candidates:
        short = "s" + str(item["seed"])
        name = JOB_PREFIX + short
        command = [
            "sbatch", "--parsable", "--partition=gpu_5090", "--qos=gpugpu",
            "--job-name=" + name,
            "--export=ALL,CANDIDATE_ID=" + item["id"] + ",EXPECTED_PLAN_SHA256=" + plan_sha,
            str(RUNNER),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            write_receipt(
                "PARTIAL_SUBMISSION_REQUIRES_RECOVERY", jobs, free,
                failed_candidate=item["id"],
                scheduler_error=result.stderr.strip() or result.stdout.strip(),
            )
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        job_id = int(result.stdout.strip().split(";")[0])
        jobs.append({"candidate_id": item["id"], "job_id": job_id, "job_name": name})

    write_receipt("SUBMITTED_PENDING_ALLOCATION", jobs, free)
    print(json.dumps({"status": "SUBMITTED_PENDING_ALLOCATION", "jobs": jobs}, sort_keys=True))


if __name__ == "__main__":
    main()
