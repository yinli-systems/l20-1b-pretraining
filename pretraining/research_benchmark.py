#!/usr/bin/env python3
"""Read-only benchmark of the current pretraining document pipeline."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
from tokenizers import Tokenizer


ROOT = Path("/home/hhai/pretrain")
PARQUET = ROOT / (
    "data/raw-cache/HuggingFaceTB--smollm-corpus/"
    "3ba9d605774198c5868892d7a8deda78031a781f/"
    "fineweb-edu-dedup/train-00119-of-00234.parquet"
)
WORD_RE = re.compile(r"[a-z0-9]+")


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def is_contaminated(text: str, hashes: set[bytes], ngram_size: int = 13) -> bool:
    words = WORD_RE.findall(text.lower())
    for index in range(len(words) - ngram_size + 1):
        ngram = " ".join(words[index : index + ngram_size]).encode()
        if hashlib.blake2b(ngram, digest_size=8).digest() in hashes:
            return True
    return False


def timed(label, function):
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    print(f"BENCH {label} seconds={elapsed:.6f}")
    return result, elapsed


def main() -> None:
    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer/tokenizer.json"))
    rows = []
    domains = Counter()
    scores = Counter()
    lang_scores = []
    chars = []
    parquet = pq.ParquetFile(PARQUET, memory_map=True)
    for batch in parquet.iter_batches(batch_size=2048, columns=["text", "metadata"]):
        for row in batch.to_pylist():
            metadata = row["metadata"] or {}
            if metadata.get("language_score", 1.0) < 0.90 or metadata.get("int_score", 4) < 4:
                continue
            text = row["text"]
            if not isinstance(text, str) or len(text.strip()) < 200:
                continue
            rows.append(text)
            domains[urlparse(metadata.get("url", "")).netloc.lower()] += 1
            scores[int(metadata.get("int_score", -1))] += 1
            lang_scores.append(float(metadata.get("language_score", 0)))
            chars.append(len(text))
            if len(rows) >= 4096:
                break
        if len(rows) >= 4096:
            break
    parquet.close()
    print(
        "SAMPLE",
        f"docs={len(rows)}",
        f"chars={sum(chars)}",
        f"median_chars={statistics.median(chars):.1f}",
        f"p95_chars={sorted(chars)[int(len(chars) * .95)]:.0f}",
        f"mean_language_score={statistics.mean(lang_scores):.5f}",
        f"quality_scores={dict(sorted(scores.items()))}",
        f"unique_domains={len(domains)}",
        f"top_domain_share={max(domains.values()) / len(rows):.5f}",
    )

    connection = sqlite3.connect(ROOT / "manifests/decontam-13gram.sqlite")
    hashes, load_seconds = timed(
        "load_decontam_hashes",
        lambda: {row[0] for row in connection.execute("SELECT hash FROM ngrams")},
    )
    connection.close()
    print(f"DECONTAM hashes={len(hashes)}")

    digests, normalize_seconds = timed(
        "normalize_sha256",
        lambda: [hashlib.sha256(normalized_text(text).encode()).digest() for text in rows],
    )
    contaminated, contam_seconds = timed(
        "decontam_13gram",
        lambda: [is_contaminated(text, hashes) for text in rows],
    )
    serial_encodings, serial_seconds = timed(
        "tokenize_serial",
        lambda: [tokenizer.encode(text) for text in rows],
    )
    serial_tokens = sum(len(encoding.ids) for encoding in serial_encodings)
    print(f"TOKENS serial={serial_tokens} serial_tok_s={serial_tokens / serial_seconds:.1f}")

    batch_results = []
    for batch_size in (16, 64, 256, 1024):
        def encode_batches():
            encoded = []
            for start in range(0, len(rows), batch_size):
                encoded.extend(tokenizer.encode_batch(rows[start : start + batch_size]))
            return encoded

        encodings, elapsed = timed(f"tokenize_batch_{batch_size}", encode_batches)
        batch_tokens = sum(len(encoding.ids) for encoding in encodings)
        if [[*encoding.ids] for encoding in encodings] != [[*encoding.ids] for encoding in serial_encodings]:
            raise RuntimeError(f"batch_size={batch_size} changed token IDs")
        batch_results.append((batch_size, elapsed, batch_tokens / elapsed))

    accepted = len(rows) - sum(contaminated)
    print(
        "SUMMARY",
        f"accepted={accepted}",
        f"contaminated={sum(contaminated)}",
        f"normalize_docs_s={len(rows) / normalize_seconds:.1f}",
        f"decontam_docs_s={len(rows) / contam_seconds:.1f}",
        f"serial_tok_s={serial_tokens / serial_seconds:.1f}",
        "batch=" + ",".join(f"{size}:{rate:.1f}" for size, _, rate in batch_results),
        f"digests={len(digests)}",
        f"hash_load_s={load_seconds:.3f}",
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
