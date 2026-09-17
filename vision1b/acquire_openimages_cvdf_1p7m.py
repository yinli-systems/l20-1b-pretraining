#!/usr/bin/env python3
"""Download and integrity-check the official CVDF train archives with resume support."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


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


def acquire(url: str, destination: Path) -> dict:
    partial = destination.with_suffix(destination.suffix + ".part")
    if not destination.exists():
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
            raise RuntimeError(f"wget failed with exit {result.returncode}: {url}")
        os.replace(partial, destination)
    subprocess.run(["gzip", "--test", str(destination)], check=True)
    digest = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n"
    )
    return {
        "bytes": destination.stat().st_size,
        "filename": destination.name,
        "sha256": digest,
        "url": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "DOWNLOAD_PLAN_NOT_ADMITTED":
        raise RuntimeError("expected a download-only protocol")
    download = protocol["download"]
    if arguments.workers < 1 or arguments.workers > int(download["parallel_downloads"]):
        raise RuntimeError("worker count exceeds the frozen concurrency cap")
    shards = list(download["shards"])
    if len(shards) != int(download["expected_archive_count"]):
        raise RuntimeError("archive-count contract mismatch")
    if arguments.limit:
        shards = shards[: arguments.limit]
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    archive_root = arguments.output_root / "tar"
    archive_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (arguments.output_root / ".acquire.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    completed: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {}
        for shard in shards:
            filename = f"{download['archive_prefix']}{shard}{download['archive_suffix']}"
            url = f"{download['base_url']}/{filename}"
            futures[executor.submit(acquire, url, archive_root / filename)] = filename
        for future in as_completed(futures):
            filename = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append({"filename": filename, "error": repr(error)})
            atomic_json(
                arguments.output_root / "progress.json",
                {
                    "bytes": sum(row["bytes"] for row in completed),
                    "completed": len(completed),
                    "failed": failures,
                    "requested": len(shards),
                    "status": "RUNNING_NOT_ADMITTED",
                    "updated_unix": time.time(),
                },
            )
    receipt = {
        "archives": sorted(completed, key=lambda row: row["filename"]),
        "bytes": sum(row["bytes"] for row in completed),
        "claim_boundary": protocol["claim_boundary"],
        "elapsed_seconds": time.time() - started,
        "failures": failures,
        "protocol_sha256": sha256(arguments.protocol),
        "requested_archives": len(shards),
        "status": (
            "DOWNLOADED_NOT_ADMITTED"
            if not failures and len(shards) == int(download["expected_archive_count"])
            else "SMOKE_COMPLETE_NOT_ADMITTED"
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
