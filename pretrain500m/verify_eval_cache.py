#!/usr/bin/env python3
"""Verify every frozen offline evaluation-cache file and alias."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    files = aliases = total_bytes = 0
    for record in manifest["files"]:
        path = args.root / record["path"]
        if record["type"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != record["target"]:
                raise RuntimeError(f"evaluation-cache alias mismatch: {path}")
            aliases += 1
        elif record["type"] == "file":
            if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise RuntimeError(f"evaluation-cache file mismatch: {path}")
            files += 1
            total_bytes += record["bytes"]
        else:
            raise RuntimeError(f"unknown manifest record type: {record}")
    print(json.dumps({"status": "PASS_FROZEN_OFFLINE_EVAL_CACHE", "files": files,
                      "aliases": aliases, "bytes": total_bytes}, sort_keys=True))


if __name__ == "__main__":
    main()
