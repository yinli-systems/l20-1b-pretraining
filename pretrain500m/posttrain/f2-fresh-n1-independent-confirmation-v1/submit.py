#!/usr/bin/env python3
"""Verify and submit the four frozen independent confirmation jobs exactly once."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
SOURCE = ROOT / "source/f2-fresh-n1-independent-confirmation-v1"
RUNNER = SOURCE / "run-domain.sbatch"
GATE_PLAN = SOURCE / "gate-plan.json"
RECEIPT = ROOT / "receipts/f2-fresh-n1-independent-confirmation-v1-submission.json"
DOMAINS = ("reading", "math", "code", "knowledge")
JOB_PREFIX = "p529m-f2n1-cf-"
STORAGE_GATE = 12 * 1024**3


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verify_manifest() -> str:
    manifest = SOURCE / "SHA256SUMS"
    result = subprocess.run(
        ["sha256sum", "-c", str(manifest)], cwd=SOURCE, text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError("bundle manifest failed: " + (result.stderr or result.stdout))
    return digest(manifest)


def write_receipt(status: str, jobs: list[dict], free: int, capacity: str, **extra) -> None:
    document = {
        "schema": "p529m-f2-fresh-n1-independent-confirmation-submission-v1",
        "status": status,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "jobs": jobs,
        "domains": list(DOMAINS),
        "world_size_per_job": 2,
        "submitted_gpu_total": 2 * len(jobs),
        "gate_plan_sha256": digest(GATE_PLAN),
        "bundle_manifest_sha256": digest(SOURCE / "SHA256SUMS"),
        "runner_sha256": digest(RUNNER),
        "free_bytes_before_submit": free,
        "storage_gate_bytes": STORAGE_GATE,
        "capacity_snapshot": capacity,
        "formal_promotion": False,
        "claim_boundary": "scheduler submission of four matched RTX 4090 confirmation jobs; allocation, completion, gate passage, replication, and promotion remain unverified",
        **extra,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if RECEIPT.exists():
        raise ValueError("submission receipt already exists")
    manifest_sha = verify_manifest()
    gate = json.loads(GATE_PLAN.read_text())
    if gate.get("status") != "FROZEN_BEFORE_CONFIRMATION_SCORING":
        raise ValueError("gate plan is not frozen")
    if tuple(gate.get("domains", {}).keys()) != DOMAINS:
        raise ValueError("gate plan domain order mismatch")
    expected = gate["candidates"]
    if [item["id"] for item in expected] != ["F2_parent_seed20260915", "f2n1-lr3e5"]:
        raise ValueError("candidate pair mismatch")
    for item in expected:
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_file() or digest(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"checkpoint identity failed: {item['id']}")
    for domain in DOMAINS:
        plan = json.loads((SOURCE / domain / "plan.json").read_text())
        if plan.get("status") != "FROZEN_BEFORE_CONFIRMATION_SCORING":
            raise ValueError(f"domain plan is not frozen: {domain}")
        if plan.get("reference_candidate_id") != "F2_parent_seed20260915":
            raise ValueError(f"domain reference mismatch: {domain}")
        if plan.get("candidates") != expected:
            raise ValueError(f"domain candidate identity mismatch: {domain}")
        role = plan.get("roles", {}).get("confirmation", "")
        if "do not score" not in role:
            raise ValueError(f"confirmation freeze lineage absent: {domain}")

    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free < STORAGE_GATE:
        raise ValueError(f"confirmation storage gate failed: {free} < {STORAGE_GATE}")
    capacity = subprocess.run(
        ["sinfo", "-p", "gpu_4090"], text=True, capture_output=True, check=True
    ).stdout
    queue = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%j"],
        check=True, text=True, capture_output=True,
    ).stdout
    active = [line for line in queue.splitlines() if line.split("|")[-1].startswith(JOB_PREFIX)]
    if active:
        raise ValueError("active confirmation job already exists: " + ";".join(active))

    jobs = []
    for domain in DOMAINS:
        name = JOB_PREFIX + domain[:4]
        command = [
            "sbatch", "--parsable", "--partition=gpu_4090", "--qos=gpugpu",
            "--job-name=" + name,
            "--export=ALL,DOMAIN=" + domain + ",EXPECTED_BUNDLE_SHA256=" + manifest_sha,
            str(RUNNER),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            write_receipt(
                "PARTIAL_SUBMISSION_REQUIRES_RECOVERY", jobs, free, capacity,
                failed_domain=domain,
                scheduler_error=result.stderr.strip() or result.stdout.strip(),
            )
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        job_id = int(result.stdout.strip().split(";")[0])
        jobs.append({"domain": domain, "job_id": job_id, "job_name": name})
    write_receipt("SUBMITTED_PENDING_ALLOCATION", jobs, free, capacity)
    print(json.dumps({"status": "SUBMITTED_PENDING_ALLOCATION", "jobs": jobs}, sort_keys=True))


if __name__ == "__main__":
    main()
