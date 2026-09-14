#!/usr/bin/env python3
"""Acquire only the pinned PixMo-Cap Parquet metadata shards and verify SHA-256."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
ADMISSION = ROOT / "pixmo_cap_metadata_admission.json"
RECEIPT = ROOT / "evidence" / "pixmo-cap-metadata-acquisition.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents:
        raise RuntimeError(f"unsafe destination: {relative}")
    return candidate


def download(urls: list[str], destination: Path, expected_bytes: int) -> str:
    partial = destination.with_suffix(destination.suffix + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_bytes:
                raise RuntimeError(f"oversized partial file: {partial}")
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    partial.unlink()
                    offset = 0
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"size mismatch after download: {partial.stat().st_size} != {expected_bytes}"
                )
            os.replace(partial, destination)
            return url
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("all endpoints failed: " + " | ".join(errors))


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    if admission.get("image_download_authorized") is not False:
        raise SystemExit("image download must remain blocked")
    if admission.get("training_authorized") is not False:
        raise SystemExit("training must remain blocked")
    files = admission["files"]
    expected_total = sum(item["bytes"] for item in files.values())
    if expected_total != admission["max_bytes"]:
        raise SystemExit("frozen byte total mismatch")
    destination_root = Path(admission["destination"])
    free_bytes = shutil.disk_usage(destination_root.parent if destination_root.parent.exists() else ROOT).free
    if free_bytes < expected_total + 5_000_000_000:
        raise SystemExit("insufficient free space for bounded acquisition")

    receipt = {
        "schema_version": "2026-09-13-v1",
        "started_at": utc_now(),
        "repo": admission["repo"],
        "revision": admission["revision"],
        "destination": str(destination_root),
        "image_downloaded": False,
        "training_started": False,
        "files": {},
        "status": "running",
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    try:
        for relative, expected in files.items():
            destination = safe_destination(destination_root, relative)
            urls = [
                f"{base}/{admission['repo']}/resolve/{admission['revision']}/{relative}?download=true"
                for base in admission["download_base_urls"]
            ]
            started = time.monotonic()
            endpoint = "already_present"
            if not destination.exists() or destination.stat().st_size != expected["bytes"]:
                endpoint = download(urls, destination, expected["bytes"])
            actual_hash = sha256_file(destination)
            if actual_hash != expected["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch: {relative}")
            receipt["files"][relative] = {
                "bytes": destination.stat().st_size,
                "sha256": actual_hash,
                "verified": True,
                "endpoint": endpoint,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        receipt["status"] = "complete"
        receipt["completed_at"] = utc_now()
        receipt["verified_bytes"] = expected_total
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["failed_at"] = utc_now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
