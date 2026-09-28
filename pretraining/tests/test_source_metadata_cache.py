#!/usr/bin/env python3
"""Verify pinned Hub metadata is atomically cached and reusable offline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import source_docs


class Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def main() -> None:
    repo = "owner/dataset"
    revision = "a" * 40
    payload = {
        "sha": revision,
        "siblings": [
            {"rfilename": "data/train-00001.parquet", "size": 123},
            {"rfilename": "data/train-00002.parquet", "size": 456},
            {"rfilename": "README.md", "size": 10},
        ],
    }
    calls = []
    original_get = source_docs.requests.get
    with tempfile.TemporaryDirectory(prefix="metadata-cache-") as directory:
        os.environ["HQ_METADATA_CACHE"] = directory

        def fetch(url, timeout):
            calls.append((url, timeout))
            return Response(payload)

        source_docs.requests.get = fetch
        first = source_docs._repo_parquet_files(repo, "data/", revision, 1)

        def offline(*_args, **_kwargs):
            raise RuntimeError("network must not be called for a valid cache")

        source_docs.requests.get = offline
        second = source_docs._repo_parquet_files(repo, "data/", revision, 1)
        cache = Path(directory) / "owner--dataset" / f"{revision}.json"
        if first != second or len(first) != 2 or len(calls) != 1 or not cache.is_file():
            raise RuntimeError(
                f"cache test failed: first={first}, second={second}, calls={calls}, cache={cache}"
            )
        cached = json.loads(cache.read_text())
        if cached.get("sha") != revision:
            raise RuntimeError(f"cached revision mismatch: {cached}")
    source_docs.requests.get = original_get
    print(json.dumps({"passed": True, "network_calls": len(calls), "parquet_files": len(first)}))


def test_pinned_metadata_cache_is_reusable_offline() -> None:
    main()


if __name__ == "__main__":
    main()
