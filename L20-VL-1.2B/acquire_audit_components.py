#!/usr/bin/env python3
"""Acquire only pre-authorized audit components, verify hashes, never extract."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib.parse import quote


ROOT = Path("/home/hhai/l20-vl-1.2b")
REGISTRY = ROOT / "component_admission.json"
RECEIPT = ROOT / "evidence/ocr13-acquisition.json"
BASE_URL = (
    "https://hf-mirror.com/datasets/nvidia/Llama-Nemotron-VLM-Dataset-v1/resolve/"
    "13b7986f8266a3cadee8a6eed1e05c438d0fe695/"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def acquire(destination: Path, relative: str, metadata: dict) -> dict:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == metadata["bytes"] and sha256(target) == metadata["sha256"]:
            return {"path": relative, "status": "already_verified", **metadata}
        raise RuntimeError(f"existing target fails size/hash verification: {target}")
    partial = target.with_name(target.name + ".part")
    url = BASE_URL + quote(relative, safe="/")
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--speed-time",
        "120",
        "--speed-limit",
        "10240",
        "-C",
        "-",
        "-o",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    actual_bytes = partial.stat().st_size
    actual_sha = sha256(partial)
    if actual_bytes != metadata["bytes"] or actual_sha != metadata["sha256"]:
        raise RuntimeError(
            f"verification failed for {relative}: bytes={actual_bytes}, sha256={actual_sha}"
        )
    os.replace(partial, target)
    return {"path": relative, "status": "downloaded_and_verified", **metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text())
    if registry.get("formal_training_authorized") is not False:
        raise SystemExit("FAIL: registry unexpectedly authorizes training")
    destination = Path(registry["acquisition_budget"]["destination"])
    if destination != ROOT / "data/audit/nvidia-ocr13":
        raise SystemExit(f"FAIL: unexpected destination: {destination}")

    files: dict[str, dict] = {}
    for name, component in registry["components"].items():
        if name not in {"ocr_1", "ocr_3"}:
            raise SystemExit(f"FAIL: unauthorized component: {name}")
        if component["acquisition_status"] != "authorized_for_integrity_and_quality_audit_only":
            raise SystemExit(f"FAIL: {name} is not acquisition-authorized")
        if not component["training_status"].startswith("blocked_"):
            raise SystemExit(f"FAIL: {name} training is not blocked")
        for relative, metadata in component["files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"FAIL: unsafe relative path: {relative}")
            files[relative] = metadata

    expected_bytes = sum(item["bytes"] for item in files.values())
    if expected_bytes > registry["acquisition_budget"]["max_bytes"]:
        raise SystemExit("FAIL: expected acquisition exceeds frozen budget")
    disk = os.statvfs(destination.parent if destination.parent.exists() else ROOT)
    available = disk.f_bavail * disk.f_frsize
    if available < expected_bytes + 20 * 2**30:
        raise SystemExit("FAIL: less than expected acquisition plus 20GiB safety margin")

    started = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(acquire, destination, relative, metadata): relative
            for relative, metadata in files.items()
        }
        for future in as_completed(futures):
            relative = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"VERIFIED {relative}", flush=True)
            except Exception as error:
                errors.append(f"{relative}: {error!r}")
                print(f"FAILED {relative}: {error!r}", flush=True)

    payload = {
        "schema_version": "2026-09-13-v1",
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": registry["parent_repo"],
        "source_revision": registry["parent_revision"],
        "destination": str(destination),
        "expected_bytes": expected_bytes,
        "verified_files": sorted(results, key=lambda item: item["path"]),
        "errors": errors,
        "status": "complete" if not errors and len(results) == len(files) else "failed",
        "archive_extracted": False,
        "formal_training": False,
        "training_prediction_tokens": 0,
        "next_gate": "safe tar member, decode, JSONL reference, duplicate, label distribution, and human-sample audits",
    }
    atomic_json(RECEIPT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise SystemExit("FAIL: one or more acquisition files failed")


if __name__ == "__main__":
    main()
