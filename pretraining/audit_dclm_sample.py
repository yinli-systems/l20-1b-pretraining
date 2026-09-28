#!/usr/bin/env python3
"""Aggregate quality audit for the pinned first DCLM production shard."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq


ROOT = Path("/home/hhai/pretrain")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IP_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
BOILERPLATE = ("cookie policy", "privacy policy", "enable javascript", "click here", "subscribe now")


def quantile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, int(len(values) * fraction))]


def main() -> None:
    paths = list((ROOT / "data/raw-cache/mlfoundations--dclm-baseline-1.0-parquet").rglob("*.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one DCLM audit shard, found {len(paths)}")
    parquet = pq.ParquetFile(paths[0], memory_map=True)
    scanned = rejected_language = 0
    texts = []
    domains = Counter()
    language_scores = []
    fasttext_scores = []
    hashes = set()
    exact_duplicates = email_documents = ip_documents = boilerplate_documents = 0
    for batch in parquet.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            scanned += 1
            if row.get("language") != "en" or float(row.get("language_score", 0)) < 0.90:
                rejected_language += 1
                continue
            text = row.get("text")
            if not isinstance(text, str) or len(text.strip()) < 200:
                continue
            normalized = " ".join(text.lower().split())
            digest = hashlib.sha256(normalized.encode()).digest()
            if digest in hashes:
                exact_duplicates += 1
            hashes.add(digest)
            texts.append(text)
            domains[urlparse(row.get("url", "")).netloc.lower()] += 1
            language_scores.append(float(row.get("language_score", 0)))
            fasttext_scores.append(float(row.get("fasttext_score", 0)))
            email_documents += bool(EMAIL_RE.search(text))
            ip_documents += bool(IP_RE.search(text))
            lowered = text.lower()
            boilerplate_documents += any(phrase in lowered for phrase in BOILERPLATE)
            if len(texts) >= 20_000:
                break
        if len(texts) >= 20_000:
            break
    parquet.close()
    lengths = [len(text) for text in texts]
    result = {
        "path": str(paths[0]),
        "sample_documents": len(texts),
        "rows_scanned": scanned,
        "language_rejected": rejected_language,
        "unique_domains": len(domains),
        "top_domain_share": max(domains.values()) / len(texts),
        "median_characters": statistics.median(lengths),
        "p95_characters": quantile(lengths, 0.95),
        "language_score_mean": statistics.mean(language_scores),
        "language_score_min": min(language_scores),
        "fasttext_score_median": statistics.median(fasttext_scores),
        "fasttext_score_p10": quantile(fasttext_scores, 0.10),
        "exact_duplicate_documents": exact_duplicates,
        "email_document_rate": email_documents / len(texts),
        "ipv4_document_rate": ip_documents / len(texts),
        "boilerplate_phrase_document_rate": boilerplate_documents / len(texts),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
