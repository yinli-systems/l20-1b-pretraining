#!/usr/bin/env python3
"""Remove superseded packed-data intermediates after a strict preflight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
DATA = ROOT / "data"
CANDIDATES = (
    DATA / "fineweb-edu-parts",
    DATA / "fineweb-edu-parts-dedup-v2",
)
FINAL_DATA = DATA / "fineweb-edu-formal-v1"
CHECKPOINT = ROOT / "formal/run-v5/resume.pt"
EXPECTED_MANIFEST_SHA256 = "29f75eca421f26653dd520e650d50e6a7802e9a499944b3b8382688fa3806e1a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_snapshot() -> dict:
    manifest = FINAL_DATA / "manifest.json"
    train = sorted((FINAL_DATA / "train-npy").glob("*.npy"))
    val = sorted((FINAL_DATA / "val-npy").glob("*.npy"))
    result = {
        "manifest_sha256": sha256(manifest),
        "train_files": len(train),
        "train_bytes": sum(path.stat().st_size for path in train),
        "val_files": len(val),
        "val_bytes": sum(path.stat().st_size for path in val),
    }
    if result["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("frozen final-data manifest hash mismatch")
    return result


def main() -> None:
    uid = os.getuid()
    root_real = ROOT.resolve(strict=True)
    data_real = DATA.resolve(strict=True)
    if data_real.parent != root_real:
        raise RuntimeError("unexpected data root")

    job_snapshot = subprocess.run(
        ["squeue", "-u", "scxi253", "-h", "-o", "%i|%T|%j|%R"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    formal_rows = [row for row in job_snapshot if row.startswith("1586840|")]
    if formal_rows != [next(row for row in formal_rows if "|RUNNING|" in row)]:
        raise RuntimeError(f"formal job is not uniquely running: {formal_rows}")

    before_final = final_snapshot()
    checkpoint_before = CHECKPOINT.stat()
    inode_records: dict[tuple[int, int], dict] = {}
    inventory = []
    for candidate in CANDIDATES:
        if candidate.is_symlink() or not candidate.is_dir():
            raise RuntimeError(f"candidate is not a real directory: {candidate}")
        if candidate.resolve(strict=True).parent != data_real:
            raise RuntimeError(f"candidate escaped data root: {candidate}")
        for path in candidate.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"refusing symlink: {path}")
            stat = path.stat()
            if stat.st_uid != uid:
                raise RuntimeError(f"refusing non-owned path: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            record = {
                "path": relative,
                "size": stat.st_size,
                "blocks_bytes": stat.st_blocks * 512,
                "nlink": stat.st_nlink,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
            }
            if path.suffix == ".json":
                record["sha256"] = sha256(path)
            inventory.append(record)
            key = (stat.st_dev, stat.st_ino)
            inode = inode_records.setdefault(
                key,
                {"candidate_names": 0, "nlink": stat.st_nlink, "blocks_bytes": stat.st_blocks * 512},
            )
            inode["candidate_names"] += 1
            if inode["nlink"] != stat.st_nlink:
                raise RuntimeError(f"inconsistent link count for inode {key}")

    expected_freed_bytes = sum(
        record["blocks_bytes"]
        for record in inode_records.values()
        if record["candidate_names"] == record["nlink"]
    )
    metadata_root = ROOT / "receipts/intermediate-data-metadata-v1"
    if metadata_root.exists():
        raise RuntimeError(f"metadata destination already exists: {metadata_root}")
    for candidate in CANDIDATES:
        for source in candidate.rglob("*.json"):
            destination = metadata_root / source.relative_to(DATA)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    receipt_path = ROOT / "receipts/cleanup-intermediate-data-v1.json"
    receipt = {
        "status": "PREPARED",
        "created_unix": time.time(),
        "owner_uid": uid,
        "targets": [path.as_posix() for path in CANDIDATES],
        "reason": "superseded construction intermediates; active formal training uses only fineweb-edu-formal-v1",
        "active_jobs": job_snapshot,
        "expected_freed_bytes": expected_freed_bytes,
        "before_final_data": before_final,
        "checkpoint_before": {
            "size": checkpoint_before.st_size,
            "mtime_ns": checkpoint_before.st_mtime_ns,
        },
        "inventory": inventory,
        "metadata_copy": metadata_root.as_posix(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    for candidate in CANDIDATES:
        shutil.rmtree(candidate)

    absent = [not path.exists() and not path.is_symlink() for path in CANDIDATES]
    after_final = final_snapshot()
    checkpoint_after = CHECKPOINT.stat()
    if not all(absent):
        raise RuntimeError("one or more cleanup targets remain")
    if after_final != before_final:
        raise RuntimeError("frozen final data changed during cleanup")
    if checkpoint_after.st_size != checkpoint_before.st_size:
        raise RuntimeError("formal checkpoint size changed during cleanup")

    usage = shutil.disk_usage(ROOT)
    receipt.update(
        {
            "status": "VERIFIED_ABSENT",
            "completed_unix": time.time(),
            "targets_absent": absent,
            "after_final_data": after_final,
            "checkpoint_after": {
                "size": checkpoint_after.st_size,
                "mtime_ns": checkpoint_after.st_mtime_ns,
            },
            "filesystem_free_bytes_after": usage.free,
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "status", "targets", "targets_absent", "expected_freed_bytes",
        "filesystem_free_bytes_after", "after_final_data", "checkpoint_after"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
