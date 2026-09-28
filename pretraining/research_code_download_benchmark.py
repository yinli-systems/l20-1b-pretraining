#!/usr/bin/env python3
"""Benchmark Stack-Edu Software Heritage downloads at several concurrencies."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config


def candidate_ids(paths: list[Path], limit: int, seed: int) -> list[str]:
    values: list[str] = []
    for path in paths:
        table = pq.read_table(path, columns=["blob_id", "license_type", "int_score"])
        for row in table.to_pylist():
            if row.get("blob_id") and row.get("license_type") == "permissive" and row.get("int_score", 0) >= 4:
                values.append(str(row["blob_id"]))
    values = list(dict.fromkeys(values))
    random.Random(seed).shuffle(values)
    if len(values) < limit:
        raise RuntimeError(f"only found {len(values)} unique candidates, need {limit}")
    return values[:limit]


def run(blob_ids: list[str], workers: int) -> dict:
    client = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=workers * 2,
            connect_timeout=20,
            read_timeout=60,
            retries={"max_attempts": 5},
        ),
        region_name="us-east-1",
    )

    def fetch(blob_id: str) -> tuple[bool, int]:
        try:
            obj = client.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
            with gzip.GzipFile(fileobj=obj["Body"]) as stream:
                return True, len(stream.read())
        except Exception:
            return False, 0

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(fetch, blob_ids))
    elapsed = time.perf_counter() - start
    successes = sum(ok for ok, _ in outcomes)
    raw_bytes = sum(size for _, size in outcomes)
    return {
        "workers": workers,
        "requested": len(blob_ids),
        "successes": successes,
        "failures": len(blob_ids) - successes,
        "elapsed_s": elapsed,
        "docs_per_s": successes / elapsed,
        "raw_mib_per_s": raw_bytes / elapsed / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    ids = candidate_ids(args.paths, args.samples, args.seed)
    orders = ([32, 48, 64], [64, 48, 32])
    for repetition, order in enumerate(orders):
        shuffled = ids.copy()
        random.Random(args.seed + repetition).shuffle(shuffled)
        for workers in order:
            print(json.dumps({"repetition": repetition, **run(shuffled, workers)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
