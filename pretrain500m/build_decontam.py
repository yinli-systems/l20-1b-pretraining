#!/usr/bin/env python3
"""Build an exact 13-word benchmark decontamination hash set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


WORD_RE = re.compile(r"[a-z0-9]+")
BENCHMARKS = [
    ("cais/mmlu", "all", "c30699e8356da336a370243923dbaf21066bb9fe", ("validation", "test", "dev")),
    ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", ("validation", "test")),
    ("allenai/ai2_arc", "ARC-Easy", "210d026faf9955653af8916fad021475a3f00453", ("validation", "test")),
    ("Rowan/hellaswag", None, "218ec52e09a7e7462a5400043bb9a69a41d06b76", ("validation", "test")),
    ("allenai/winogrande", "winogrande_xl", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", ("validation", "test")),
    ("ybisk/piqa", None, "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0", ("validation", "test")),
    ("allenai/openbookqa", "main", "388097ea7776314e93a529163e0fea805b8a6454", ("validation", "test")),
    ("google/boolq", None, "35b264d03638db9f4ce671b711558bf7ff0f80d5", ("validation",)),
    ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866", ("test",)),
    ("EleutherAI/lambada_openai", None, "900124bf3b8235c6daf21033af9948b3f07346c4", ("test",)),
    ("truthfulqa/truthful_qa", "generation", "741b8276f2d1982aa3d5b832d3ee81ed3b896490", ("validation",)),
    ("truthfulqa/truthful_qa", "multiple_choice", "741b8276f2d1982aa3d5b832d3ee81ed3b896490", ("validation",)),
]


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngram-size", type=int, default=13)
    args = parser.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output)
    connection.execute("CREATE TABLE IF NOT EXISTS ngrams (hash BLOB PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS provenance (repo TEXT, config TEXT, revision TEXT, split TEXT, rows INTEGER)"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS provenance_key ON provenance(repo, config, revision, split)"
    )

    total_rows = total_ngrams = 0
    for repo, config, revision, splits in BENCHMARKS:
        for split in splits:
            config_key = config or ""
            completed = connection.execute(
                "SELECT rows FROM provenance WHERE repo=? AND config=? AND revision=? AND split=?",
                (repo, config_key, revision, split),
            ).fetchone()
            if completed:
                print(json.dumps({"repo": repo, "config": config, "split": split, "rows": completed[0], "resumed": True}))
                total_rows += completed[0]
                continue
            dataset = load_dataset(
                repo, config, split=split, streaming=True, revision=revision, trust_remote_code=True
            )
            rows = 0
            pending: list[tuple[bytes]] = []
            for row in dataset:
                text = " \n ".join(strings(row))
                words = WORD_RE.findall(text.lower())
                for index in range(len(words) - args.ngram_size + 1):
                    ngram = " ".join(words[index : index + args.ngram_size]).encode()
                    pending.append((hashlib.blake2b(ngram, digest_size=8).digest(),))
                rows += 1
                if len(pending) >= 100_000:
                    before = connection.total_changes
                    connection.executemany("INSERT OR IGNORE INTO ngrams(hash) VALUES (?)", pending)
                    total_ngrams += connection.total_changes - before
                    connection.commit()
                    pending.clear()
            if pending:
                before = connection.total_changes
                connection.executemany("INSERT OR IGNORE INTO ngrams(hash) VALUES (?)", pending)
                total_ngrams += connection.total_changes - before
            connection.execute(
                "INSERT INTO provenance(repo, config, revision, split, rows) VALUES (?, ?, ?, ?, ?)",
                (repo, config_key, revision, split, rows),
            )
            connection.commit()
            total_rows += rows
            print(json.dumps({"repo": repo, "config": config, "split": split, "rows": rows}), flush=True)
    final_count = connection.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
    connection.close()
    print(json.dumps({"output": str(args.output), "rows": total_rows, "unique_13grams": final_count}), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
