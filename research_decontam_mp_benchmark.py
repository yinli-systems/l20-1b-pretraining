#!/usr/bin/env python3
"""Read-only multiprocessing benchmark for 13-gram decontamination."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import re
import sqlite3
import time
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/home/hhai/pretrain")
PARQUET = ROOT / (
    "data/raw-cache/HuggingFaceTB--smollm-corpus/"
    "3ba9d605774198c5868892d7a8deda78031a781f/"
    "fineweb-edu-dedup/train-00119-of-00234.parquet"
)
WORD_RE = re.compile(r"[a-z0-9]+")
HASHES: set[bytes] = set()


def is_contaminated(text: str, ngram_size: int = 13) -> bool:
    words = WORD_RE.findall(text.lower())
    for index in range(len(words) - ngram_size + 1):
        ngram = " ".join(words[index : index + ngram_size]).encode()
        if hashlib.blake2b(ngram, digest_size=8).digest() in HASHES:
            return True
    return False


def run(workers: int, rows: list[str]) -> tuple[list[bool], float]:
    start = time.perf_counter()
    if workers == 1:
        result = [is_contaminated(row) for row in rows]
    else:
        context = mp.get_context("fork")
        with context.Pool(workers) as pool:
            result = pool.map(is_contaminated, rows, chunksize=32)
    return result, time.perf_counter() - start


def main() -> None:
    global HASHES
    parquet = pq.ParquetFile(PARQUET, memory_map=True)
    rows = []
    for batch in parquet.iter_batches(batch_size=2048, columns=["text", "metadata"]):
        for row in batch.to_pylist():
            metadata = row["metadata"] or {}
            if metadata.get("language_score", 1.0) >= 0.90 and metadata.get("int_score", 4) >= 4:
                rows.append(row["text"])
            if len(rows) >= 8192:
                break
        if len(rows) >= 8192:
            break
    parquet.close()
    connection = sqlite3.connect(ROOT / "manifests/decontam-13gram.sqlite")
    HASHES = {row[0] for row in connection.execute("SELECT hash FROM ngrams")}
    connection.close()
    baseline = None
    for workers in (1, 2, 4, 6, 7):
        result, elapsed = run(workers, rows)
        if baseline is None:
            baseline = result
        elif result != baseline:
            raise RuntimeError(f"workers={workers} changed decontamination results")
        print(
            "DECONTAM_MP",
            f"workers={workers}",
            f"seconds={elapsed:.6f}",
            f"docs_s={len(rows) / elapsed:.1f}",
            f"contaminated={sum(result)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
