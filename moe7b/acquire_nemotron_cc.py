#!/usr/bin/env python3
"""Stage official Nemotron-CC compressed JSONL objects with resume support."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request


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


def partition_key(path: str) -> str:
    values = {}
    for part in path.split("/"):
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    return "|".join(values[key] for key in ("quality", "kind", "kind2"))


def priority(path: str) -> tuple[int, str]:
    quality = partition_key(path).split("|", 1)[0]
    order = {"high": 0, "medium-high": 1, "medium": 2, "medium-low": 3, "low": 4}
    return order[quality], path


def download_one(base_url: str, relative: str, output_root: Path, minimum_free: int) -> dict:
    if shutil.disk_usage(output_root).free < minimum_free:
        raise RuntimeError("free-space floor reached")
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
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
                f"{base_url}/{relative}",
            ],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"wget failed with exit {result.returncode}: {relative}")
        os.replace(partial, destination)
    subprocess.run(["zstd", "--test", "--quiet", str(destination)], check=True)
    digest = sha256(destination)
    sidecar.write_text(f"{digest}  {destination.name}\n")
    return {
        "bytes": destination.stat().st_size,
        "partition": partition_key(relative),
        "path": relative,
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "DOWNLOAD_PLAN_NOT_ADMITTED":
        raise RuntimeError("expected a download-only protocol")
    download = protocol["download"]
    if arguments.workers < 1 or arguments.workers > int(download["parallel_downloads"]):
        raise RuntimeError("worker count exceeds the frozen concurrency cap")
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (arguments.output_root / ".acquire.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index_path = arguments.output_root / "data-jsonl.paths.gz"
    if not index_path.exists():
        temporary = index_path.with_suffix(index_path.suffix + ".tmp")
        urllib.request.urlretrieve(download["index_url"], temporary)
        os.replace(temporary, index_path)
    if sha256(index_path) != download["expected_index_sha256"]:
        raise RuntimeError("Nemotron-CC path index identity mismatch")
    with gzip.open(index_path, "rt") as handle:
        paths = [line.strip() for line in handle if line.strip()]
    if len(paths) != int(download["expected_objects"]) or len(paths) != len(set(paths)):
        raise RuntimeError("Nemotron-CC path inventory count/uniqueness mismatch")
    counts = Counter(partition_key(path) for path in paths)
    if dict(sorted(counts.items())) != download["partition_object_counts"]:
        raise RuntimeError("Nemotron-CC partition inventory mismatch")
    paths.sort(key=priority)
    if arguments.limit:
        paths = paths[: arguments.limit]
    started = time.time()
    completed: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                download["base_url"],
                relative,
                arguments.output_root,
                int(download["minimum_free_bytes"]),
            ): relative
            for relative in paths
        }
        for future in as_completed(futures):
            relative = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append({"path": relative, "error": repr(error)})
            if len(completed) % 10 == 0 or failures:
                atomic_json(
                    arguments.output_root / "progress.json",
                    {
                        "bytes": sum(row["bytes"] for row in completed),
                        "completed": len(completed),
                        "failed": failures,
                        "requested": len(paths),
                        "status": "RUNNING_NOT_ADMITTED",
                        "updated_unix": time.time(),
                    },
                )
    completed.sort(key=lambda row: row["path"])
    manifest = arguments.output_root / "download-manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in completed))
    receipt = {
        "bytes": sum(row["bytes"] for row in completed),
        "claim_boundary": protocol["claim_boundary"],
        "completed_objects": len(completed),
        "elapsed_seconds": time.time() - started,
        "failures": failures,
        "index_sha256": sha256(index_path),
        "manifest_sha256": sha256(manifest),
        "protocol_sha256": sha256(arguments.protocol),
        "requested_objects": len(paths),
        "status": (
            "DOWNLOADED_NOT_ADMITTED"
            if not failures and len(paths) == int(download["expected_objects"])
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
