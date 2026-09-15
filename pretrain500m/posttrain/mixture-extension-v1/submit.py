#!/usr/bin/env python3
"""Submit the two frozen R4 extension seeds exactly once."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
INPUTS = ROOT / "source/mixture-extension-inputs-v1"
RUNNER = ROOT / "source/mixture-extension-v1/run.sbatch"
RECEIPT = ROOT / "receipts/mixture-extension-v1-submission.json"
STORAGE_GATE = 75 * 1024**3
JOB_PREFIX = "p529m-r4-t1b-"
SEEDS = (20260916, 20260917)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def runnable_four_gpu_nodes(capacity: str) -> int:
    rows = [line.split() for line in capacity.splitlines() if line.strip().startswith("gpu_5090")]
    runnable = [row for row in rows if len(row) >= 9]
    if len(runnable) != 1:
        raise ValueError("could not identify the gpu_5090 runnable-layout row")
    return int(runnable[0][4])


def write_receipt(status: str, jobs: list[dict], free: int, capacity: str, **extra) -> None:
    build_path = INPUTS / "build-receipt.json"
    document = {
        "schema": "p529m-r4-two-seed-extension-submission-v1",
        "status": status,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "jobs": jobs,
        "world_size_per_job": 4,
        "submitted_gpu_total": 4 * len(jobs),
        "target_prediction_tokens_per_run": 1_073_741_824,
        "free_bytes_before_submit": free,
        "storage_gate_bytes": STORAGE_GATE,
        "four_gpu_runnable_nodes_before_submit": runnable_four_gpu_nodes(capacity),
        "capacity_snapshot": capacity,
        "build_receipt_sha256": digest(build_path),
        "runner_sha256": digest(RUNNER),
        "formal_promotion": False,
        "claim_boundary": "submitted R4 current-corpus extensions; allocation, MFU, completion and capability remain unverified",
        **extra,
    }
    RECEIPT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if RECEIPT.exists():
        raise ValueError("extension submission receipt already exists")
    build_path = INPUTS / "build-receipt.json"
    build = json.loads(build_path.read_text())
    if build.get("status") != "PASS_EXTENSION_INPUTS_ADMITTED":
        raise ValueError("extension inputs are not admitted")
    if build.get("seeds") != list(SEEDS) or build.get("training_launched") is not False:
        raise ValueError("extension seed set or pre-launch state differs")
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free < STORAGE_GATE:
        raise ValueError(f"extension storage gate failed: {free} < {STORAGE_GATE}")
    queue = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%j"],
        check=True, text=True, capture_output=True,
    ).stdout
    active = [line for line in queue.splitlines() if line.split("|")[-1].startswith(JOB_PREFIX)]
    if active:
        raise ValueError("active extension job already exists: " + ";".join(active))
    capacity = subprocess.run(
        ["sinfo", "-p", "gpu_5090"], check=True, text=True, capture_output=True
    ).stdout
    if runnable_four_gpu_nodes(capacity) < 1:
        raise ValueError("no public single-node four-GPU layout is currently available")

    manifest_sha = build["manifest_sha256"]
    admission_sha = build["admission_sha256"]
    protocol_sha = build["protocol_sha256"]
    build_sha = digest(build_path)
    jobs = []
    for seed in SEEDS:
        name = f"{JOB_PREFIX}s{seed}"
        exports = ",".join([
            "ALL",
            f"RUN_SEED={seed}",
            f"EXPECTED_MANIFEST_SHA256={manifest_sha}",
            f"EXPECTED_ADMISSION_SHA256={admission_sha}",
            f"EXPECTED_PROTOCOL_SHA256={protocol_sha}",
            f"EXPECTED_BUILD_RECEIPT_SHA256={build_sha}",
        ])
        result = subprocess.run([
            "sbatch", "--parsable", "--partition=gpu_5090", "--qos=gpugpu",
            "--job-name=" + name, "--export=" + exports, str(RUNNER),
        ], text=True, capture_output=True)
        if result.returncode != 0:
            write_receipt(
                "PARTIAL_SUBMISSION_REQUIRES_RECOVERY", jobs, free, capacity,
                failed_seed=seed,
                scheduler_error=result.stderr.strip() or result.stdout.strip(),
            )
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        jobs.append({"seed": seed, "job_id": int(result.stdout.strip().split(";")[0]), "job_name": name})
    write_receipt("SUBMITTED_PENDING_ALLOCATION", jobs, free, capacity)
    print(json.dumps({"status": "SUBMITTED_PENDING_ALLOCATION", "jobs": jobs}, sort_keys=True))


if __name__ == "__main__":
    main()
