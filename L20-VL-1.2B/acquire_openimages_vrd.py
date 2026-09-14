#!/usr/bin/env python3
"""Fail-closed acquisition of the bounded Open Images VRD annotation sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FILENAMES = {
    "visual_relationships": "oidv6-train-annotations-vrd.csv",
    "boxable_class_names": "class-descriptions-boxable.csv",
    "attribute_names": "oidv6-attributes-description.csv",
}


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: Path, source: dict) -> dict:
    size = path.stat().st_size
    md5 = digest_file(path, "md5")
    if size != int(source["expected_bytes"]):
        raise RuntimeError(f"size mismatch for {path}: {size}")
    if md5 != source["expected_md5_hex"]:
        raise RuntimeError(f"MD5 mismatch for {path}: {md5}")
    return {"path": str(path), "bytes": size, "md5": md5, "sha256": digest_file(path, "sha256")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "authorized_bounded_openimages_vrd_acquisition_for_audit":
        raise SystemExit("acquisition protocol is not authorized")
    if protocol.get("training_authorized") is not False:
        raise SystemExit("acquisition protocol must keep training disabled")
    if digest_file(Path(__file__), "sha256") != protocol["source_code"]["acquisition_script_sha256"]:
        raise RuntimeError("acquisition script hash mismatch")

    parent = protocol["existing_image_parent"]
    for key in ("manifest", "human_audit"):
        path = Path(parent[key])
        expected = parent[f"{key}_sha256"]
        if digest_file(path, "sha256") != expected:
            raise RuntimeError(f"parent hash mismatch: {key}")

    destination = Path(protocol["limits"]["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    if free < int(protocol["limits"]["minimum_free_bytes_before_acquisition"]):
        raise RuntimeError(f"free disk gate failed: {free}")

    records = {}
    for name, source in protocol["sources"].items():
        path = destination / FILENAMES[name]
        if path.exists():
            try:
                records[name] = validate_file(path, source)
                print(f"OPENIMAGES_VRD_REUSE source={name}", flush=True)
                continue
            except RuntimeError:
                pass
        partial = path.with_suffix(path.suffix + ".part")
        print(f"OPENIMAGES_VRD_DOWNLOAD source={name}", flush=True)
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
        print(f"OPENIMAGES_VRD_VERIFIED source={name} sha256={records[name]['sha256']}", flush=True)

    total_bytes = sum(record["bytes"] for record in records.values())
    if total_bytes > int(protocol["limits"]["maximum_new_bytes"]):
        raise RuntimeError("bounded acquisition byte cap exceeded")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_verified_acquisition_only",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(args.protocol),
        "protocol_sha256": digest_file(args.protocol, "sha256"),
        "training_started": False,
        "source_records": records,
        "total_bytes": total_bytes,
        "free_bytes_after": shutil.disk_usage(destination).free,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".part")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.receipt)
    print("OPENIMAGES_VRD_ACQUISITION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
