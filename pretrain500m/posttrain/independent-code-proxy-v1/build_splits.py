#!/usr/bin/env python3
"""Build the frozen output-blind MBPP sanitized development split."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


DATA_SHA256 = "e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850"
SPLIT_SALT = "p529m-independent-code-proxy-v1-mbpp-sanitized-20260916"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(row: dict) -> bytes:
    return json.dumps(row, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def atomic(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.data.read_bytes()
    if digest_bytes(raw) != DATA_SHA256:
        raise ValueError("MBPP parquet digest mismatch")
    table = pq.read_table(args.data)
    if table.num_rows != 257 or table.column_names != [
        "source_file", "task_id", "prompt", "code", "test_imports", "test_list"
    ]:
        raise ValueError("unexpected MBPP sanitized test shape")
    records = table.to_pylist()
    if len({row["task_id"] for row in records}) != len(records):
        raise ValueError("duplicate MBPP task IDs")
    for row in records:
        if not isinstance(row["task_id"], int) or not all(
            isinstance(row[key], str) and row[key] for key in ("source_file", "prompt", "code")
        ) or not isinstance(row["test_imports"], list) or not isinstance(row["test_list"], list):
            raise ValueError("invalid MBPP row")
        if not row["test_list"] or not all(isinstance(item, str) and item for item in row["test_list"]):
            raise ValueError("invalid MBPP tests")
    rows_bytes = b"".join(canonical(row) + b"\n" for row in records)
    assignments = []
    for ordinal, row in enumerate(records):
        row_sha = digest_bytes(canonical(row))
        prompt_sha = digest_bytes(row["prompt"].encode())
        split_key = digest_bytes(
            f"{SPLIT_SALT}:{ordinal}:{row['task_id']}:{prompt_sha}".encode()
        )
        assignments.append({
            "ordinal": ordinal,
            "task_id": row["task_id"],
            "row_sha256": row_sha,
            "prompt_sha256": prompt_sha,
            "split_key": split_key,
        })
    development = {
        row["ordinal"] for row in sorted(
            assignments, key=lambda row: (row["split_key"], row["ordinal"])
        )[:128]
    }
    for row in assignments:
        row["role"] = "development" if row["ordinal"] in development else "confirmation"
    split = {
        "schema": "p529m-independent-code-proxy-mbpp-sanitized-splits-v1",
        "source_sha256": DATA_SHA256,
        "rows_jsonl_sha256": digest_bytes(rows_bytes),
        "split_salt": SPLIT_SALT,
        "split_method": "sort sha256(salt:ordinal:task_id:prompt_sha256); first 128 development, remaining 129 confirmation; gold code and tests excluded from ordering",
        "counts": {"development": 128, "confirmation": 129},
        "rows": assignments,
    }
    split_bytes = (json.dumps(split, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n").encode()
    atomic(args.rows_output, rows_bytes)
    atomic(args.split_output, split_bytes)


if __name__ == "__main__":
    main()
