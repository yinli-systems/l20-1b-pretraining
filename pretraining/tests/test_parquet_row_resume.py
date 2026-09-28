#!/usr/bin/env python3
"""Verify row-group skipping preserves exact Parquet resume order."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import source_docs


def main() -> None:
    repo = "owner/dataset"
    revision = "b" * 40
    relative = "data/train.parquet"
    with tempfile.TemporaryDirectory(prefix="parquet-resume-") as directory:
        root = Path(directory)
        parquet_path = root / "train.parquet"
        table = pa.table({"index": list(range(1000)), "text": [f"row-{i}" for i in range(1000)]})
        pq.write_table(table, parquet_path, row_group_size=100)
        state = root / "state.sqlite"
        source_docs._set_file_progress(state, repo, revision, relative, 637)

        original_files = source_docs._repo_parquet_files
        original_download = source_docs._download_parquet
        source_docs._repo_parquet_files = lambda *_args, **_kwargs: [{"rfilename": relative}]
        source_docs._download_parquet = lambda *_args, **_kwargs: parquet_path
        try:
            rows = list(source_docs._parquet_rows(repo, "data/", revision, 42, state, False))
        finally:
            source_docs._repo_parquet_files = original_files
            source_docs._download_parquet = original_download

        indices = [row["index"] for row in rows]
        if indices != list(range(637, 1000)):
            raise RuntimeError(f"resume order mismatch: first={indices[:3]}, last={indices[-3:]}")
        if not source_docs._file_is_completed(state, repo, revision, relative):
            raise RuntimeError("completed file was not recorded")
        print(json.dumps({"passed": True, "first": indices[0], "last": indices[-1], "rows": len(indices)}))


def test_parquet_resume_preserves_exact_row_order() -> None:
    main()


if __name__ == "__main__":
    main()
