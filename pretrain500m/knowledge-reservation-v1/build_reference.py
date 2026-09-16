#!/usr/bin/env python3
"""Create a canonical JSONL representation of the pinned SciQ test parquet."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


DATA_SHA256 = "3a719356a29b127fc54ef3c7f51a034db4bd105d5717215e8c85d2aa58d60667"
COLUMNS = ["question", "distractor3", "distractor1", "distractor2",
           "correct_answer", "support"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.data) != DATA_SHA256:
        raise ValueError("SciQ source digest mismatch")
    table = pq.read_table(args.data)
    if table.num_rows != 1000 or table.column_names != COLUMNS:
        raise ValueError("unexpected SciQ test shape")
    lines = []
    for ordinal, row in enumerate(table.to_pylist()):
        if not all(isinstance(row[key], str) for key in COLUMNS) or not all(
            row[key].strip() for key in COLUMNS if key != "support"
        ):
            raise ValueError(f"invalid SciQ row {ordinal}")
        document = {"ordinal": ordinal, **row}
        lines.append(json.dumps(document, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
    payload = ("\n".join(lines) + "\n").encode()
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
