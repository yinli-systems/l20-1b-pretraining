#!/usr/bin/env python3
"""Submit the two frozen F2 fresh-N1 learning-rate arms exactly once."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
BUNDLE = ROOT / "source/f2-fresh-n1-continuation-v1"
RUNNER = BUNDLE / "run.sbatch"
RECEIPT = ROOT / "receipts/f2-fresh-n1-continuation-v1-submission.json"
PARENT = ROOT / "posttrain/confirmations-v1/F2_reasoning_no_synthetic-seed20260915-1590004/train/model-final.pt"
EXPECTED_PARENT_SHA = "f6a4ff43aeeb892ce7dd3dd7ed69f022cef6af688f21b3f005f67850f6041cbc"
STORAGE_GATE = 30 * 1024**3
JOB_PREFIX = "p529m-f2n1-"
ARMS = (("f2n1-lr3e5", "0.00003"), ("f2n1-lr6e5", "0.00006"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True, timeout=30)


def main() -> None:
    if RECEIPT.exists():
        raise ValueError(f"submission receipt already exists: {RECEIPT}")
    if sha(PARENT) != EXPECTED_PARENT_SHA:
        raise ValueError("F2 parent checkpoint identity mismatch")
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free < STORAGE_GATE:
        raise ValueError(f"storage gate failed: {free} < {STORAGE_GATE}")
    queue_before = run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%j|%R"]).stdout
    if any(line.split("|")[2].startswith(JOB_PREFIX) for line in queue_before.splitlines()):
        raise ValueError("active F2 fresh-N1 job already exists")

    jobs = []
    for arm, lr in ARMS:
        name = JOB_PREFIX + arm.rsplit("-", 1)[-1]
        exports = f"ALL,ARM_ID={arm},PEAK_LR={lr}"
        command = [
            "sbatch", "--parsable", "--partition=gpu_5090", "--qos=gpugpu",
            f"--job-name={name}", f"--export={exports}", str(RUNNER),
        ]
        result = run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        jobs.append({"arm_id": arm, "peak_learning_rate": lr,
                     "job_id": int(result.stdout.strip().split(";")[0]), "job_name": name})

    queue_after = run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%j|%R"]).stdout
    receipt = {
        "schema": "p529m-f2-fresh-n1-submission-v1",
        "status": "SUBMITTED_PENDING_ALLOCATION",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "jobs": jobs,
        "submitted_gpu_total": 4 * len(jobs),
        "parent_checkpoint_sha256": EXPECTED_PARENT_SHA,
        "runner_sha256": sha(RUNNER),
        "free_bytes_before_submit": free,
        "storage_gate_bytes": STORAGE_GATE,
        "queue_before": queue_before,
        "queue_after": queue_after,
        "claim_boundary": "scheduler submission only; allocation, live GPU identity, MFU, completion, capability gain and promotion remain unverified"
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
