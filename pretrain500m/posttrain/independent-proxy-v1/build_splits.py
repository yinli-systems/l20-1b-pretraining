#!/usr/bin/env python3
"""Build a content-bound, output-blind 450/450 Belebele split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


DATA_SHA256 = "15af884c2b5994fdc71199e79d553a1864b08ce1b31894aa87f6be925a064a31"
SPLIT_SALT = "p529m-independent-proxy-v1-belebele-eng-20260916"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build(data_path: Path) -> dict:
    raw = data_path.read_bytes()
    if digest_bytes(raw) != DATA_SHA256:
        raise ValueError("Belebele source digest mismatch")
    rows = []
    for ordinal, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        required = {
            "link", "question_number", "flores_passage", "question",
            "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
            "correct_answer_num", "dialect",
        }
        if not required.issubset(row) or row["dialect"] != "eng_Latn":
            raise ValueError(f"invalid row schema at ordinal {ordinal}")
        if str(row["correct_answer_num"]) not in {"1", "2", "3", "4"}:
            raise ValueError(f"invalid answer at ordinal {ordinal}")
        row_sha = digest_bytes(line)
        split_key = digest_bytes(f"{SPLIT_SALT}:{ordinal}:{row_sha}".encode())
        rows.append({
            "ordinal": ordinal,
            "row_sha256": row_sha,
            "split_key": split_key,
        })
    if len(rows) != 900:
        raise ValueError(f"expected 900 rows, found {len(rows)}")
    ranked = sorted(rows, key=lambda item: (item["split_key"], item["ordinal"]))
    development = {item["ordinal"] for item in ranked[:450]}
    for item in rows:
        item["role"] = "development" if item["ordinal"] in development else "confirmation"
    return {
        "schema": "p529m-independent-proxy-belebele-splits-v1",
        "source_sha256": DATA_SHA256,
        "split_salt": SPLIT_SALT,
        "split_method": "sort sha256(salt:line_ordinal:raw_line_sha256); first 450 development, remaining 450 confirmation",
        "counts": {"development": 450, "confirmation": 450},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, build(args.data))


if __name__ == "__main__":
    main()

