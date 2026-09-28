#!/usr/bin/env python3
"""Fail-closed recovery for one inspected transient evaluation download."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--partial", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-partial-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.evaluation_dir.resolve()
    plan_path = root / "efficiency-evaluation-plan-20260912-v4.json"
    receipt_path = root / "receipt.json"
    queue_log = root / "queue-v4.log"
    queue_lock_path = root / "queue.lock"
    if sha256(plan_path) != args.expected_plan_sha256:
        raise ValueError("Plan hash changed")
    if sha256(receipt_path) != args.expected_receipt_sha256:
        raise ValueError("Failed receipt changed")

    plan = json.loads(plan_path.read_text())
    jobs = {job["id"]: job for job in plan["jobs"]}
    job = jobs[args.job]
    receipt = json.loads(receipt_path.read_text())
    job_receipt = receipt.get("jobs", {}).get(args.job, {})
    expected_error = f"Download failed for {args.job}/{args.partial.removesuffix('.part')}: ConnectionError; partial retained"
    if receipt.get("status") != "failed" or receipt.get("current_job") != args.job:
        raise ValueError("Receipt is not the inspected failed job")
    if job_receipt.get("status") != "failed" or job_receipt.get("error") != expected_error:
        raise ValueError("Failure mode changed")

    directory = (root / "models" / args.job).resolve()
    if directory.parent != (root / "models").resolve():
        raise ValueError("Unsafe job directory")
    identity = {"job": job["id"], "repo": job["repo"], "revision": job["revision"], "files": job["files"]}
    if json.loads((directory / "ownership.json").read_text()) != identity:
        raise ValueError("Ownership mismatch")
    partial = (directory / args.partial).resolve()
    if partial.parent != directory or partial.is_symlink() or not partial.is_file():
        raise ValueError("Unsafe partial path")
    if sha256(partial) != args.expected_partial_sha256 or partial.stat().st_size != 0:
        raise ValueError("Partial content changed or is not the inspected zero-byte artifact")
    if (root / "results" / f"{args.job}.json").exists() or (root / "results" / f"{args.job}-summary.json").exists():
        raise ValueError("Result already exists")

    audit = {
        "schema": 1,
        "status": "verified_recovery_ready",
        "job": args.job,
        "plan_sha256": sha256(plan_path),
        "failed_receipt_sha256": sha256(receipt_path),
        "queue_log_sha256_at_failure": sha256(queue_log),
        "partial_path": str(partial),
        "partial_size": partial.stat().st_size,
        "partial_sha256": sha256(partial),
        "ownership_sha256": sha256(directory / "ownership.json"),
        "failure": expected_error,
        "checked_unix": time.time(),
    }
    if not args.apply:
        print(json.dumps(audit, indent=2))
        return 0

    with queue_lock_path.open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = root / "failures"
        failures.mkdir(mode=0o700, exist_ok=True)
        archive_stem = f"{args.job}-{audit['failed_receipt_sha256'][:12]}"
        archived_receipt = failures / f"{archive_stem}-receipt.json"
        archived_log = failures / f"{archive_stem}-queue.log"
        recovery_receipt = failures / f"{archive_stem}-recovery.json"
        for destination in (archived_receipt, archived_log, recovery_receipt):
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
        shutil.copyfile(receipt_path, archived_receipt)
        shutil.copyfile(queue_log, archived_log)
        if sha256(archived_receipt) != audit["failed_receipt_sha256"] or sha256(archived_log) != audit["queue_log_sha256_at_failure"]:
            raise RuntimeError("Failure archive verification failed")

        partial.unlink()
        if partial.exists() or partial.is_symlink():
            raise RuntimeError("Partial cleanup verification failed")
        audit.update(
            status="verified_recovery_applied",
            applied_unix=time.time(),
            archived_receipt=str(archived_receipt),
            archived_queue_log=str(archived_log),
            partial_verified_absent=True,
        )
        atomic_json(recovery_receipt, audit)

        failed_attempt = dict(job_receipt)
        failed_attempt["archived_receipt_sha256"] = audit["failed_receipt_sha256"]
        failed_attempt["recovery_receipt"] = str(recovery_receipt)
        receipt.setdefault("failed_attempts", []).append({args.job: failed_attempt})
        receipt["jobs"].pop(args.job)
        receipt.update(status="recovery_ready", updated_unix=time.time())
        receipt.pop("current_job", None)
        receipt.pop("error", None)
        receipt.pop("pid", None)
        atomic_json(receipt_path, receipt)

    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
