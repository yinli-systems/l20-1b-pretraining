#!/usr/bin/env python3
"""Freeze a contamination-filtered, output-blind SciQ development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


DATA_SHA256 = "55cc70950fdd8892d6c7c2cbd458d2c3c6fc5f88548b909799d147b68e7c16da"
CONTAMINATION_REPORT_SHA256 = "9950ea90d82ffe460acae573b7d4555c8d8e7fc57ae17554e2924fb6006af5d4"
DUPLICATE_CHOICE_ORDINALS = {67, 124, 318, 469, 526, 659, 718, 884, 926}
SPLIT_SALT = "p529m-independent-knowledge-proxy-v1-sciq-20260916"
OPTION_SALT = "p529m-independent-knowledge-proxy-v1-sciq-option-order-20260916"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def canonical_hash(value: object) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build(data_path: Path, contamination_path: Path) -> dict:
    raw = data_path.read_bytes()
    if digest_bytes(raw) != DATA_SHA256:
        raise ValueError("SciQ canonical JSONL digest mismatch")
    if digest(contamination_path) != CONTAMINATION_REPORT_SHA256:
        raise ValueError("contamination report digest mismatch")
    contamination = json.loads(contamination_path.read_text())
    if contamination["status"] != "PASS_CLEAN_REFERENCE_POOL_READY":
        raise ValueError("contamination gate has not passed")
    contaminated = set(contamination["contaminated_reference_ordinals"])
    if contaminated & DUPLICATE_CHOICE_ORDINALS:
        raise ValueError("exclusion sets unexpectedly overlap")

    rows = []
    eligible = []
    raw_lines = raw.splitlines()
    for line_ordinal, line in enumerate(raw_lines):
        row = json.loads(line)
        if row.get("ordinal") != line_ordinal:
            raise ValueError(f"non-canonical ordinal at row {line_ordinal}")
        required = {"question", "support", "correct_answer", "distractor1", "distractor2", "distractor3"}
        if not required.issubset(row) or not all(isinstance(row[key], str) for key in required):
            raise ValueError(f"invalid row schema at ordinal {line_ordinal}")
        if not all(row[key] for key in required - {"support"}):
            raise ValueError(f"invalid row schema at ordinal {line_ordinal}")
        options = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
        if len(set(options)) != 4 and line_ordinal not in DUPLICATE_CHOICE_ORDINALS:
            raise ValueError(f"unregistered duplicate choice at ordinal {line_ordinal}")
        excluded_reason = None
        if line_ordinal in contaminated:
            excluded_reason = "current_corpus_exact_or_lexical_near_hit"
        elif line_ordinal in DUPLICATE_CHOICE_ORDINALS:
            excluded_reason = "duplicate_choice_text"
        row_record = {
            "ordinal": line_ordinal,
            "row_sha256": digest_bytes(line),
            "eligible": excluded_reason is None,
            "excluded_reason": excluded_reason,
        }
        if excluded_reason is None:
            ordered_options = sorted(
                options,
                key=lambda text: (digest_bytes(f"{OPTION_SALT}:{line_ordinal}:{text}".encode()), text),
            )
            visible_input = {"question": row["question"], "choices": sorted(options)}
            split_key = digest_bytes(
                f"{SPLIT_SALT}:{line_ordinal}:{canonical_hash(visible_input)}".encode()
            )
            row_record.update({
                "choice_text_sha256": [digest_bytes(text.encode()) for text in ordered_options],
                "gold_position": ordered_options.index(row["correct_answer"]),
                "split_key": split_key,
            })
            eligible.append(row_record)
        rows.append(row_record)

    if len(raw_lines) != 1000 or len(contaminated) != 12 or len(eligible) != 979:
        raise ValueError("unexpected source or eligibility count")
    ranked = sorted(eligible, key=lambda item: (item["split_key"], item["ordinal"]))
    development = {item["ordinal"] for item in ranked[:489]}
    for item in eligible:
        item["role"] = "development" if item["ordinal"] in development else "confirmation"
    return {
        "schema": "p529m-independent-knowledge-proxy-sciq-splits-v1",
        "source_sha256": DATA_SHA256,
        "contamination_report_sha256": CONTAMINATION_REPORT_SHA256,
        "split_salt": SPLIT_SALT,
        "option_order_salt": OPTION_SALT,
        "split_method": "exclude frozen contamination and duplicate-choice rows; sort sha256(salt:ordinal:sha256(question plus sorted choice texts)); first 489 development, remaining 490 confirmation",
        "option_order_method": "sort four choice texts by sha256(option_salt:ordinal:choice_text), then choice text; correct-answer identity is not an input",
        "counts": {"source": 1000, "excluded_contamination": 12, "excluded_duplicate_choice": 9, "eligible": 979, "development": 489, "confirmation": 490},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--contamination-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, build(args.data, args.contamination_report))


if __name__ == "__main__":
    main()
