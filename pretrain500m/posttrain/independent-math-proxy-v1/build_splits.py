#!/usr/bin/env python3
"""Build the frozen output-blind 125/125 MGSM English split."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path


DATA_SHA256 = "50021d0f28cc957edcb44e7806425b1c7fbd648ddcb9e0a8ec689d10e57d40fa"
SPLIT_SALT = "p529m-independent-math-proxy-v1-mgsm-en-20260916"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.data.read_bytes()
    if digest(raw) != DATA_SHA256:
        raise ValueError("MGSM source digest mismatch")
    parsed = list(csv.reader(io.StringIO(raw.decode()), delimiter="\t"))
    if len(parsed) != 250 or any(len(row) != 2 or not all(row) for row in parsed):
        raise ValueError("unexpected MGSM TSV shape")
    rows = []
    for ordinal, line in enumerate(raw.splitlines()):
        row_sha = digest(line)
        rows.append({"ordinal": ordinal, "row_sha256": row_sha,
                     "split_key": digest(f"{SPLIT_SALT}:{ordinal}:{row_sha}".encode())})
    development = {row["ordinal"] for row in sorted(rows, key=lambda row: (row["split_key"], row["ordinal"]))[:125]}
    for row in rows:
        row["role"] = "development" if row["ordinal"] in development else "confirmation"
    document = {
        "schema": "p529m-independent-math-proxy-mgsm-en-splits-v1",
        "source_sha256": DATA_SHA256,
        "split_salt": SPLIT_SALT,
        "split_method": "sort sha256(salt:line_ordinal:raw_line_sha256); first 125 development, remaining 125 confirmation",
        "counts": {"development": 125, "confirmation": 125},
        "rows": rows,
    }
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()

