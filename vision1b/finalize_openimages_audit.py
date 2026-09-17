#!/usr/bin/env python3
"""Bind a completed fixed visual audit to an Open Images acquisition receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    receipt = json.loads(arguments.receipt.read_text())
    audit = json.loads(arguments.audit.read_text())
    protocol_hash = sha256(arguments.protocol)
    manifest_hash = sha256(arguments.manifest)
    expected_rows = sum(1 for line in arguments.manifest.open() if line.strip())

    if receipt.get("status") != "PENDING_FIXED_VISUAL_AUDIT":
        raise RuntimeError(f"unexpected receipt status: {receipt.get('status')}")
    if receipt.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("receipt/protocol mismatch")
    if receipt.get("manifest") != {
        "path": receipt["manifest"]["path"],
        "rows": expected_rows,
        "sha256": manifest_hash,
    }:
        raise RuntimeError("receipt/manifest mismatch")
    if (
        audit.get("status") != "COMPLETE"
        or audit.get("decision") != "PASS_FOR_BOUNDED_SELF_SUPERVISED_RESEARCH_PILOT"
        or audit.get("protocol_sha256") != protocol_hash
        or audit.get("manifest", {}).get("rows") != expected_rows
        or audit.get("manifest", {}).get("sha256") != manifest_hash
    ):
        raise RuntimeError("audit decision/identity mismatch")

    receipt["status"] = "PASS_FIXED_VISUAL_AUDIT"
    receipt["visual_audit"] = {
        "path": str(arguments.audit.resolve()),
        "sha256": sha256(arguments.audit),
        "decision": audit["decision"],
        "items_reviewed": audit["checks"]["fixed_items_reviewed"],
    }
    temporary = arguments.receipt.with_suffix(arguments.receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, arguments.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
