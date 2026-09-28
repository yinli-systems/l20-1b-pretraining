"""Delete one verified obsolete 20-step smoke checkpoint and write a receipt.

This program is deliberately not a general-purpose cleanup utility. It cannot
target the production, gate, continuation, evaluation, or release checkpoints.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time


TARGET = Path("/home/hhai/pretrain/checkpoints/smoke-test/smoke/step-00000020/lit_model.pth")
EXPECTED_SHA256 = "a3af3c1766858480216298b0ed4c71f96496dbe7dc44ddc622e5e39bea5dcbb2"
EXPECTED_SIZE = 13_200_786_287
LOCK = Path("/home/hhai/pretrain/continuation/20260911-v1/gpu.lock")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def open_references(target: Path) -> list[str]:
    references = []
    for descriptor in Path("/proc").glob("[0-9]*/fd/*"):
        try:
            if Path(os.readlink(descriptor)) == target:
                references.append(str(descriptor))
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return references


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not args.apply:
        parser.error("--apply is required")
    if args.receipt.exists():
        raise FileExistsError(args.receipt)

    with LOCK.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if gpu_processes:
            raise RuntimeError("CUDA process active")
        resolved = TARGET.resolve(strict=True)
        metadata = TARGET.lstat()
        if resolved != TARGET or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("target is not the exact regular file")
        if metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
            raise ValueError("unexpected target ownership or hard links")
        if metadata.st_size != EXPECTED_SIZE or digest(TARGET) != EXPECTED_SHA256:
            raise ValueError("target identity changed")
        references = open_references(TARGET)
        if references:
            raise RuntimeError(f"target is open: {references}")

        filesystem = Path("/home/hhai/pretrain")
        before = os.statvfs(filesystem).f_bavail * os.statvfs(filesystem).f_frsize
        started = time.time()
        TARGET.unlink()
        if TARGET.exists() or TARGET.is_symlink():
            raise RuntimeError("target deletion was not verified")
        directory = os.open(TARGET.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        after = os.statvfs(filesystem).f_bavail * os.statvfs(filesystem).f_frsize
        write_exclusive(
            args.receipt,
            {
                "status": "verified_complete",
                "target": str(TARGET),
                "sha256": EXPECTED_SHA256,
                "bytes": EXPECTED_SIZE,
                "inode": metadata.st_ino,
                "uid": metadata.st_uid,
                "hard_links": metadata.st_nlink,
                "open_references_before": references,
                "cuda_processes_before": [],
                "free_bytes_before": before,
                "free_bytes_after": after,
                "started_unix": started,
                "completed_unix": time.time(),
                "recovery": "Re-run the 20-step smoke test from the retained source and data.",
                "preserved_classes": [
                    "production_20b_checkpoints",
                    "gate_checkpoints",
                    "continuation_a_and_b",
                    "evaluation_outputs",
                    "published_hf_weights"
                ]
            },
        )
        print(json.dumps({"status": "verified_complete", "freed_bytes": after - before}))


if __name__ == "__main__":
    main()
