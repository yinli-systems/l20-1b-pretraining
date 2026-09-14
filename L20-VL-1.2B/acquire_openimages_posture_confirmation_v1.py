#!/usr/bin/env python3
"""Acquire only the official Open Images validation annotations for confirmation.

This is a fail-closed data-acquisition step.  It cannot build examples, inspect
model outputs, train parameters, or start an evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parent
FILENAMES = {
    "validation_visual_relationships": "oidv6-validation-annotations-vrd.csv",
    "validation_image_metadata": "validation-images-with-rotation.csv",
}
STATUS = "authorized_openimages_validation_annotation_acquisition_only_v1"


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing source artifact: {path}")
    size = path.stat().st_size
    md5 = digest_file(path, "md5")
    if size != int(source["expected_bytes"]):
        raise RuntimeError(f"size mismatch for {path}: {size}")
    if md5 != source["expected_md5_hex"]:
        raise RuntimeError(f"MD5 mismatch for {path}: {md5}")
    return {
        "path": str(path),
        "bytes": size,
        "md5": md5,
        "sha256": digest_file(path, "sha256"),
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != STATUS:
        raise RuntimeError("validation annotation acquisition is not authorized")
    if protocol.get("download_authorized") is not True:
        raise RuntimeError("download authorization is absent")
    for field in (
        "training_authorized",
        "model_evaluation_authorized",
        "automatic_training_start",
        "automatic_evaluation_start",
    ):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    if set(protocol.get("sources", {})) != set(FILENAMES):
        raise RuntimeError("source set changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)

    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    if digest_file(Path(__file__), "sha256") != protocol["source_code"]["acquirer_sha256"]:
        raise RuntimeError("acquisition source hash mismatch")
    test_path = ROOT / "test_acquire_openimages_posture_confirmation_v1.py"
    if digest_file(test_path, "sha256") != protocol["source_code"]["test_sha256"]:
        raise RuntimeError("acquisition test hash mismatch")

    for name, item in protocol["reused_ontology"].items():
        if digest_file(Path(item["path"]), "sha256") != item["sha256"]:
            raise RuntimeError(f"reused ontology hash mismatch: {name}")

    destination = Path(protocol["limits"]["destination"])
    expected_destination = Path(
        "/home/hhai/l20-vl-1.2b/data/openimages-posture-confirmation-v1/sources"
    )
    if destination != expected_destination:
        raise RuntimeError("destination changed")
    destination.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(destination).free
    if free_before < int(protocol["limits"]["minimum_free_bytes_before"]):
        raise RuntimeError(f"free disk gate failed: {free_before}")

    records = {}
    for name in sorted(FILENAMES):
        source = protocol["sources"][name]
        path = destination / FILENAMES[name]
        if path.exists():
            records[name] = validate_file(path, source)
            print(f"OPENIMAGES_CONFIRMATION_REUSE source={name}", flush=True)
            continue
        partial = path.with_suffix(path.suffix + ".part")
        print(f"OPENIMAGES_CONFIRMATION_DOWNLOAD source={name}", flush=True)
        subprocess.run(
            [
                "curl",
                "-4",
                "-fL",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--retry-all-errors",
                "--connect-timeout",
                "15",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                source["url"],
            ],
            check=True,
        )
        records[name] = validate_file(partial, source)
        os.replace(partial, path)
        records[name]["path"] = str(path)
        print(
            f"OPENIMAGES_CONFIRMATION_VERIFIED source={name} sha256={records[name]['sha256']}",
            flush=True,
        )

    total_bytes = sum(record["bytes"] for record in records.values())
    if total_bytes > int(protocol["limits"]["maximum_annotation_bytes"]):
        raise RuntimeError("bounded annotation byte cap exceeded")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_verified_validation_annotation_acquisition_only",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol), "sha256": digest_file(args.protocol, "sha256")},
        "source_records": records,
        "reused_ontology": protocol["reused_ontology"],
        "total_downloaded_bytes": total_bytes,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(destination).free,
        "training_started": False,
        "model_evaluation_started": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    partial_receipt = args.receipt.with_suffix(args.receipt.suffix + ".part")
    partial_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(partial_receipt, args.receipt)
    print("OPENIMAGES_CONFIRMATION_ANNOTATIONS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
