#!/usr/bin/env python3
"""Fault-inject one aria2 failure and verify resumable outer retry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import source_docs


def main() -> None:
    calls = []
    original_run = source_docs.subprocess.run
    original_sleep = source_docs.time.sleep
    with tempfile.TemporaryDirectory(prefix="parquet-retry-") as directory:
        os.environ["HQ_RAW_CACHE"] = directory
        payload = b"verified parquet fixture"
        item = {"rfilename": "data/train.parquet", "size": len(payload)}

        def injected(command, check):
            calls.append(command)
            if len(calls) == 1:
                raise subprocess.CalledProcessError(5, command)
            out = next(value.split("=", 1)[1] for value in command if value.startswith("--out="))
            target = Path(next(value.split("=", 1)[1] for value in command if value.startswith("--dir="))) / out
            target.write_bytes(payload)
            return subprocess.CompletedProcess(command, 0)

        source_docs.subprocess.run = injected
        source_docs.time.sleep = lambda _seconds: None
        try:
            path = source_docs._download_parquet("owner/dataset", "c" * 40, item)
        finally:
            source_docs.subprocess.run = original_run
            source_docs.time.sleep = original_sleep
        if len(calls) != 2 or path.read_bytes() != payload or not path.with_suffix(path.suffix + ".verified.json").is_file():
            raise RuntimeError(f"retry test failed: calls={len(calls)}, path={path}")
        print(json.dumps({"passed": True, "attempts": len(calls), "bytes": path.stat().st_size}))


def test_parquet_download_retries_after_aria2_failure() -> None:
    main()


if __name__ == "__main__":
    main()
