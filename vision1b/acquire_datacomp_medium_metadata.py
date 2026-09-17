#!/usr/bin/env python3
"""Stage the pinned DataComp-medium parquet inventory with resume support."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.parse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def acquire(repository: str, revision: str, row: dict, root: Path, floor: int) -> dict:
    if shutil.disk_usage(root).free < floor:
        raise RuntimeError("free-space floor reached")
    destination = root / "metadata" / row["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if not destination.exists():
        quoted = urllib.parse.quote(row["path"], safe="/")
        url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{quoted}?download=true"
        result = subprocess.run(
            [
                "wget",
                "--continue",
                "--tries=20",
                "--timeout=90",
                "--waitretry=5",
                "--output-document",
                str(partial),
                url,
            ],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"wget failed with exit {result.returncode}: {row['path']}")
        os.replace(partial, destination)
    if destination.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"byte-length mismatch: {row['path']}")
    digest = sha256(destination)
    if digest != row["sha256"]:
        raise RuntimeError(f"LFS SHA-256 mismatch: {row['path']}")
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n"
    )
    return {"bytes": destination.stat().st_size, "path": row["path"], "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "METADATA_DOWNLOAD_PLAN_NOT_ADMITTED":
        raise RuntimeError("expected a metadata-only download protocol")
    dataset = protocol["dataset"]
    download = protocol["download"]
    if sha256(arguments.inventory) != dataset["inventory_sha256"]:
        raise RuntimeError("DataComp metadata inventory identity mismatch")
    rows = [json.loads(line) for line in arguments.inventory.read_text().splitlines() if line]
    if len(rows) != int(dataset["expected_parquet_files"]):
        raise RuntimeError("DataComp metadata inventory count mismatch")
    if len({row["path"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate DataComp metadata paths")
    if sum(int(row["bytes"]) for row in rows) != int(dataset["expected_metadata_bytes"]):
        raise RuntimeError("DataComp metadata byte inventory mismatch")
    if arguments.workers < 1 or arguments.workers > int(download["parallel_downloads"]):
        raise RuntimeError("worker count exceeds the frozen concurrency cap")
    if arguments.limit:
        rows = rows[: arguments.limit]
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (arguments.output_root / ".acquire.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    completed: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                acquire,
                dataset["repository"],
                dataset["revision"],
                row,
                arguments.output_root,
                int(download["minimum_free_bytes"]),
            ): row["path"]
            for row in rows
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append({"path": path, "error": repr(error)})
            atomic_json(
                arguments.output_root / "progress.json",
                {
                    "bytes": sum(row["bytes"] for row in completed),
                    "completed": len(completed),
                    "failed": failures,
                    "requested": len(rows),
                    "status": "RUNNING_METADATA_NOT_ADMITTED",
                    "updated_unix": time.time(),
                },
            )
    completed.sort(key=lambda row: row["path"])
    manifest = arguments.output_root / "download-manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in completed))
    receipt = {
        "bytes": sum(row["bytes"] for row in completed),
        "claim_boundary": protocol["claim_boundary"],
        "completed_files": len(completed),
        "elapsed_seconds": time.time() - started,
        "failures": failures,
        "inventory_sha256": sha256(arguments.inventory),
        "manifest_sha256": sha256(manifest),
        "protocol_sha256": sha256(arguments.protocol),
        "requested_files": len(rows),
        "status": (
            "METADATA_STAGED_NOT_ADMITTED"
            if not failures and len(rows) == int(dataset["expected_parquet_files"])
            else "METADATA_SMOKE_COMPLETE_NOT_ADMITTED"
            if not failures
            else "FAILED_NOT_ADMITTED"
        ),
    }
    atomic_json(arguments.output_root / "acquisition-receipt.json", receipt)
    atomic_json(arguments.output_root / "progress.json", receipt)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
