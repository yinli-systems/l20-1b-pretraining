#!/usr/bin/env python3
"""Resumably download and integrity-check the official CVDF train archives."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
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


def curl_base() -> list[str]:
    return ["curl", "--fail", "--location", "--show-error", "--silent"]


def content_length(url: str) -> int:
    result = subprocess.run(
        curl_base()
        + ["--head", "--connect-timeout", "20", "--max-time", "45", url],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    lengths = re.findall(r"(?im)^content-length:\s*(\d+)\s*$", result.stdout)
    if not lengths:
        raise RuntimeError(f"missing Content-Length for {url}")
    return int(lengths[-1])


def fetch_segment(url: str, destination: Path, start: int, end: int) -> dict:
    expected = end - start + 1
    if destination.exists() and destination.stat().st_size == expected:
        return {"bytes": expected, "end": end, "path": str(destination), "start": start}
    if destination.exists():
        destination.unlink()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    for attempt in range(1, 11):
        temporary.unlink(missing_ok=True)
        result = subprocess.run(
            curl_base()
            + [
                "--range",
                f"{start}-{end}",
                "--connect-timeout",
                "20",
                "--max-time",
                "1200",
                "--retry",
                "3",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "--output",
                str(temporary),
                url,
            ],
            check=False,
        )
        if result.returncode == 0 and temporary.stat().st_size == expected:
            os.replace(temporary, destination)
            return {
                "bytes": expected,
                "end": end,
                "path": str(destination),
                "start": start,
            }
        observed = temporary.stat().st_size if temporary.exists() else 0
        temporary.unlink(missing_ok=True)
        if attempt == 10:
            raise RuntimeError(
                f"segment failed after 10 attempts: {url} "
                f"range={start}-{end} observed={observed} expected={expected}"
            )
        time.sleep(min(5 * attempt, 60))
    raise AssertionError("unreachable")


def assemble(parts: list[Path], destination: Path, expected_bytes: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".assembling")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, 16 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"assembled byte-length mismatch: {temporary.stat().st_size} != {expected_bytes}"
        )
    os.replace(temporary, destination)


def acquire(
    url: str,
    destination: Path,
    segment_bytes: int,
    segment_workers: int,
) -> dict:
    partial = destination.with_suffix(destination.suffix + ".part")
    expected_bytes = content_length(url)
    segment_root = destination.parent / ".segments" / destination.name
    segment_root.mkdir(parents=True, exist_ok=True)
    spans = [
        (index, start, min(start + segment_bytes, expected_bytes) - 1)
        for index, start in enumerate(range(0, expected_bytes, segment_bytes))
    ]
    segment_paths = [segment_root / f"{index:06d}.part" for index, _, _ in spans]
    if not destination.exists():
        with ThreadPoolExecutor(max_workers=segment_workers) as executor:
            futures = {
                executor.submit(fetch_segment, url, segment_paths[index], start, end): index
                for index, start, end in spans
            }
            for future in as_completed(futures):
                future.result()
        assemble(segment_paths, partial, expected_bytes)
        subprocess.run(["gzip", "--test", str(partial)], check=True)
        os.replace(partial, destination)
    if destination.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"existing byte-length mismatch for {url}: "
            f"{destination.stat().st_size} != {expected_bytes}"
        )
    subprocess.run(["gzip", "--test", str(destination)], check=True)
    digest = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n"
    )
    for path in segment_paths:
        path.unlink(missing_ok=True)
    try:
        segment_root.rmdir()
        segment_root.parent.rmdir()
    except OSError:
        pass
    return {
        "bytes": destination.stat().st_size,
        "content_length": expected_bytes,
        "filename": destination.name,
        "segments": len(spans),
        "sha256": digest,
        "url": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--segment-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "DOWNLOAD_PLAN_NOT_ADMITTED":
        raise RuntimeError("expected a download-only protocol")
    download = protocol["download"]
    if arguments.workers < 1 or arguments.workers > int(download["parallel_downloads"]):
        raise RuntimeError("archive worker count exceeds the frozen concurrency cap")
    if (
        arguments.segment_workers < 1
        or arguments.segment_workers > int(download["max_segment_workers_per_archive"])
    ):
        raise RuntimeError("segment worker count exceeds the frozen concurrency cap")
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
            futures[
                executor.submit(
                    acquire,
                    url,
                    archive_root / filename,
                    int(download["segment_bytes"]),
                    arguments.segment_workers,
                )
            ] = filename
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
