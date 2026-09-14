#!/usr/bin/env python3
"""Acquire and validate official Open Images validation textual narratives."""
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


STATUS = "authorized_openimages_validation_caption_acquisition_only_v1"


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_jsonl(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    size = path.stat().st_size
    md5 = digest_file(path, "md5")
    if size != int(source["expected_bytes"]):
        raise RuntimeError(f"caption size mismatch: {size}")
    if md5 != source["expected_md5_hex"]:
        raise RuntimeError(f"caption MD5 mismatch: {md5}")
    rows = 0
    image_ids = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL at line {line_number}") from error
        if not row.get("image_id") or not str(row.get("caption", "")).strip():
            raise RuntimeError(f"missing image or caption at line {line_number}")
        rows += 1
        image_ids.add(str(row["image_id"]))
    if rows != int(source["expected_rows"]):
        raise RuntimeError(f"caption row count mismatch: {rows}")
    if len(image_ids) != int(source["expected_unique_image_ids"]):
        raise RuntimeError(f"caption unique-image count mismatch: {len(image_ids)}")
    return {
        "path": str(path),
        "bytes": size,
        "md5": md5,
        "sha256": digest_file(path, "sha256"),
        "rows": rows,
        "unique_image_ids": len(image_ids),
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != STATUS or protocol.get("download_authorized") is not True:
        raise RuntimeError("caption acquisition is not authorized")
    for field in (
        "training_authorized",
        "model_evaluation_authorized",
        "automatic_training_start",
        "automatic_evaluation_start",
    ):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")


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
        raise RuntimeError("caption acquirer source hash mismatch")
    for name, item in protocol["prerequisites"].items():
        if digest_file(Path(item["path"]), "sha256") != item["sha256"]:
            raise RuntimeError(f"prerequisite hash mismatch: {name}")

    destination = Path(protocol["limits"]["destination"])
    if destination != Path(
        "/home/hhai/l20-vl-1.2b/data/openimages-posture-confirmation-v1/sources"
    ):
        raise RuntimeError("caption destination changed")
    destination.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(destination).free
    if free_before < int(protocol["limits"]["minimum_free_bytes_before"]):
        raise RuntimeError(f"free disk gate failed: {free_before}")

    source = protocol["source"]
    path = destination / "open_images_validation_captions.jsonl"
    if path.exists():
        record = validate_jsonl(path, source)
        print("OPENIMAGES_CONFIRMATION_CAPTIONS_REUSE", flush=True)
    else:
        partial = path.with_suffix(path.suffix + ".part")
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
        record = validate_jsonl(partial, source)
        os.replace(partial, path)
        record["path"] = str(path)
    if record["bytes"] > int(protocol["limits"]["maximum_new_bytes"]):
        raise RuntimeError("bounded caption byte cap exceeded")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_verified_validation_caption_acquisition_only",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol), "sha256": digest_file(args.protocol, "sha256")},
        "caption_record": record,
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
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
