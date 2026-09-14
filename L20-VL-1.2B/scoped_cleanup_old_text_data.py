#!/usr/bin/env python3
"""Remove only superseded/rebuildable text-data intermediates after preflight."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = Path("/home/hhai/pretrain/data")
TARGETS = (
    DATA_ROOT / "raw-cache",
    DATA_ROOT / "full-npy",
)
PRESERVE = (
    DATA_ROOT / "packed",
    Path("/home/hhai/pretrain/manifests/data-full-receipt.json"),
    Path("/home/hhai/pretrain/checkpoints/full/final/lit_model.pth"),
    Path("/home/hhai/pretrain/evaluations/final/hf/pytorch_model.bin"),
    Path("/home/hhai/pretrain/continuation/20260911-v1/pilot-A/resume.pth"),
    Path("/home/hhai/pretrain/continuation/20260911-v1/pilot-B/resume.pth"),
)
RECEIPT = Path("/home/hhai/l20-vl-1.2b/evidence/old-text-data-cleanup.json")


def tree_stats(path: Path) -> dict[str, int]:
    files = 0
    bytes_ = 0
    for root, _, names in os.walk(path):
        for name in names:
            item = Path(root) / name
            try:
                stat = item.lstat()
            except FileNotFoundError:
                continue
            files += 1
            bytes_ += stat.st_size
    return {"files": files, "logical_bytes": bytes_}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def open_file_references(path: Path) -> list[str]:
    result = subprocess.run(
        ["lsof", "+D", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"lsof failed for {path}: {result.stderr.strip()}")
    return result.stdout.splitlines()[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    if DATA_ROOT.resolve() != DATA_ROOT:
        errors.append("data root must resolve to its exact path")
    for path in PRESERVE:
        if not path.exists():
            errors.append(f"required preserved artifact missing: {path}")

    before: dict[str, dict[str, int]] = {}
    for target in TARGETS:
        if target.parent != DATA_ROOT or target.is_symlink():
            errors.append(f"unsafe target: {target}")
            continue
        if not target.is_dir():
            errors.append(f"target is not an existing directory: {target}")
            continue
        if target.stat().st_uid != os.getuid():
            errors.append(f"target is not owned by uid {os.getuid()}: {target}")
        references = open_file_references(target)
        if references:
            errors.append(f"target has {len(references)} open-file references: {target}")
        before[str(target)] = tree_stats(target)

    if errors:
        raise SystemExit("FAIL\n" + "\n".join(errors))

    payload = {
        "schema_version": "2026-09-13-v1",
        "requested_action": "delete rebuildable old text-data intermediates while retaining replay data and evidence",
        "targets": before,
        "preserved": [str(path) for path in PRESERVE],
        "execute": args.execute,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed": False,
    }
    atomic_json(RECEIPT, payload)

    if not args.execute:
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("DRY RUN: pass --execute to remove the two exact targets")
        return

    for target in TARGETS:
        shutil.rmtree(target)

    absent = {str(target): not target.exists() for target in TARGETS}
    preserved = {str(path): path.exists() for path in PRESERVE}
    if not all(absent.values()) or not all(preserved.values()):
        raise SystemExit("FAIL: post-delete absence/preservation verification failed")

    stat = os.statvfs(DATA_ROOT)
    payload.update(
        {
            "completed": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "verified_absent": absent,
            "verified_preserved": preserved,
            "filesystem_available_bytes_after": stat.f_bavail * stat.f_frsize,
            "recoverability": "deleted targets are not recoverable locally; reconstruction requires the frozen source and data receipts",
        }
    )
    atomic_json(RECEIPT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
