#!/usr/bin/env python3
"""Remove exact superseded binaries and invalid partial packs after quota failure."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
OWNER = "owner task 01a09290-b43f-7431-be8a-412ea5d37954"
FILES = [
    "pilots/1583433/continuous/resume.pt",
    "pilots/1583433/split/resume.pt",
    "pilots/1583508/continuous/resume.pt",
    "pilots/1583508/split/resume.pt",
    "pilots/1583555/continuous/resume.pt",
    "pilots/1583555/split/resume.pt",
    "pilots/lr-1583515/lr-0.0015/resume.pt",
    "pilots/1583820/continuous/resume.pt",
    "pilots/1583820/split/resume.pt",
    "pilots/1583831/continuous/resume.pt",
    "pilots/1583831/continuous/resume.pt.next",
    "pilots/export-smoke-v2-1583801/model.safetensors",
    "pilots/export-smoke-v2-1583804/model.safetensors",
]
DIRECTORIES = [
    f"data/fineweb-edu-parts/part-{index:02d}"
    for index in (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
]
BLOCKING_JOB_NAMES = {
    "p500m-pack14-v1",
    "p500m-dedup-audit-v1",
    "p529m-lr-long-v2",
    "p529m-resume-det-v2",
    "p529m-export-smoke-v2",
}


def tree_size(path: Path) -> tuple[int, int]:
    size = 0
    files = 0
    for root, _, names in os.walk(path):
        for name in names:
            item = Path(root) / name
            size += item.stat().st_size
            files += 1
    return size, files


def checkpoint_sidecar(path: Path) -> str | None:
    sidecar = path.with_name("resume.sha256")
    if path.name != "resume.pt" or not sidecar.is_file():
        return None
    fields = sidecar.read_text().split()
    return fields[0] if len(fields) == 2 and fields[1] == "resume.pt" else None


def main() -> None:
    if ROOT.resolve() != ROOT or not ROOT.is_dir():
        raise SystemExit(f"unexpected root: {ROOT}")
    owner = (ROOT / "OWNER.txt").read_text()
    if OWNER not in owner:
        raise SystemExit("owner marker mismatch")
    active_text = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%j|%T"], text=True
    )
    active = [line for line in active_text.splitlines() if line]
    conflicts = [line for line in active if line.split("|")[1] in BLOCKING_JOB_NAMES]
    if conflicts:
        raise SystemExit(f"active conflicting jobs: {conflicts}")

    file_records = []
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"expected exact file missing: {path}")
        file_records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "checkpoint_sha256_from_preserved_sidecar": checkpoint_sidecar(path),
            }
        )
    directory_records = []
    for relative in DIRECTORIES:
        path = ROOT / relative
        if not path.is_dir():
            raise SystemExit(f"expected failed partition missing: {path}")
        size, files = tree_size(path)
        if (path / "web" / "manifest.json").exists():
            raise SystemExit(f"refusing to remove completed partition: {path}")
        directory_records.append({"path": str(path), "bytes": size, "files": files})

    for record in file_records:
        Path(record["path"]).unlink()
    for record in directory_records:
        shutil.rmtree(record["path"])
    missing_verified = all(not Path(record["path"]).exists() for record in file_records + directory_records)
    if not missing_verified:
        raise SystemExit("post-cleanup absence verification failed")

    receipt = {
        "status": "VERIFIED_EXACT_TARGETS_ABSENT",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(ROOT),
        "owner_marker": owner.strip(),
        "active_jobs_before_cleanup": active,
        "reason": "quota failure; remove superseded pilot binaries and restart incomplete pack partitions from scratch",
        "preserved": "logs, metrics, manifests, comparison receipts, export receipts, and checkpoint SHA-256 sidecars",
        "deleted_files": file_records,
        "deleted_incomplete_directories": directory_records,
        "bytes_removed": sum(x["bytes"] for x in file_records + directory_records),
        "absence_verified": missing_verified,
    }
    target = ROOT / "receipts" / "quota-cleanup-v1.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
