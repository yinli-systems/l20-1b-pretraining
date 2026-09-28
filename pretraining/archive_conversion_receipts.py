#!/usr/bin/env python3
"""Archive transient conversion receipts while the efficiency queue runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    root = args.evaluation_dir.resolve()
    receipt_path = root / "receipt.json"
    archive_dir = root / "conversion-receipts"
    archive_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    while True:
        for source in sorted((root / "models").glob("*/conversion-receipt.json")):
            job = source.parent.name
            destination = archive_dir / f"{job}.json"
            if not destination.exists():
                atomic_copy(source, destination)
                print(
                    json.dumps(
                        {
                            "status": "archived",
                            "job": job,
                            "sha256": sha256(destination),
                            "path": str(destination),
                        }
                    ),
                    flush=True,
                )

        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") in {"complete", "failed"}:
                print(json.dumps({"status": "queue_terminal", "queue_status": receipt["status"]}), flush=True)
                return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
