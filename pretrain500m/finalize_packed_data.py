#!/usr/bin/env python3
"""Verify audited partitions and materialize the frozen packed-token corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


BLOCK_SIZE = 2049


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_shard(record: dict) -> Path:
    path = Path(record["path"])
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.uint16 or array.ndim != 1 or array.size % BLOCK_SIZE:
        raise RuntimeError(f"invalid packed shard: {path}")
    actual = digest(path)
    if actual != record["sha256"] or int(array.size) != int(record["tokens"]):
        raise RuntimeError(f"shard receipt mismatch: {path}")
    sidecar = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if sidecar.get("sha256") != actual or int(sidecar.get("tokens", -1)) != int(array.size):
        raise RuntimeError(f"sidecar mismatch: {path}")
    return path


def write_shard(source: Path, output: Path, tokens: int | None = None) -> dict:
    if tokens is None:
        os.link(source, output)
        size = int(np.load(output, mmap_mode="r", allow_pickle=False).size)
    else:
        source_array = np.load(source, mmap_mode="r", allow_pickle=False)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(source_array[:tokens], dtype=np.uint16), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        size = tokens
    record = {"path": str(output), "tokens": size, "sha256": digest(output)}
    atomic_json(output.with_suffix(output.suffix + ".sha256.json"), record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--dedup-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parts", type=int, default=14)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--train-tokens", type=int)
    target.add_argument("--use-all-train-tokens", action="store_true")
    args = parser.parse_args()
    if args.train_tokens is not None and args.train_tokens % BLOCK_SIZE:
        raise SystemExit("--train-tokens must be divisible by 2049")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    receipt = json.loads(args.dedup_receipt.read_text())
    if receipt.get("status") != "PASS_ZERO_CROSS_PART_DUPLICATES":
        raise RuntimeError("cross-part exact-dedup audit did not pass")
    if receipt.get("parts_root") != str(args.parts_root):
        raise RuntimeError("dedup receipt refers to a different parts root")

    manifests = []
    train_sources = []
    val_sources = []
    for index in range(args.expected_parts):
        part = args.parts_root / f"part-{index:02d}"
        manifest_path = part / "web" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        audit = receipt["parts"][index]
        if audit["part"] != f"part-{index:02d}" or audit["manifest_sha256"] != digest(manifest_path):
            raise RuntimeError(f"manifest changed after audit: part-{index:02d}")
        database_path = part / "dedup.sqlite"
        if audit["dedup_db_sha256"] != digest(database_path):
            raise RuntimeError(f"dedup database changed after audit: part-{index:02d}")
        train_sources.extend(verify_shard(record) for record in manifest["train_shards"])
        val_sources.extend(verify_shard(record) for record in manifest["validation_shards"])
        manifests.append({"path": str(manifest_path), "sha256": digest(manifest_path)})
    available = sum(int(np.load(path, mmap_mode="r", allow_pickle=False).size) for path in train_sources)
    train_tokens = available if args.use_all_train_tokens else args.train_tokens
    if train_tokens is None:
        raise AssertionError("a training-token target was not resolved")
    if available < train_tokens:
        raise RuntimeError(f"only {available} verified train tokens are available")

    train_dir = args.output_dir / "train-npy"
    val_dir = args.output_dir / "val-npy"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_records = []
    remaining = train_tokens
    for source in train_sources:
        if remaining == 0:
            break
        source_tokens = int(np.load(source, mmap_mode="r", allow_pickle=False).size)
        take = min(source_tokens, remaining)
        take -= take % BLOCK_SIZE
        output = train_dir / f"train-{len(train_records):05d}.npy"
        train_records.append(write_shard(source, output, None if take == source_tokens else take))
        remaining -= take
    if remaining:
        raise RuntimeError(f"failed to materialize {remaining} train tokens")
    val_records = []
    for source in val_sources:
        output = val_dir / f"val-{len(val_records):05d}.npy"
        val_records.append(write_shard(source, output))
    validation_tokens = sum(record["tokens"] for record in val_records)
    manifest = {
        "status": "FROZEN_VERIFIED_PACK",
        "source": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "block_size": BLOCK_SIZE,
        "train_tokens": train_tokens,
        "unique_prediction_tokens": train_tokens // BLOCK_SIZE * (BLOCK_SIZE - 1),
        "validation_tokens": validation_tokens,
        "document_exact_dedup": True,
        "cross_part_dedup_receipt": str(args.dedup_receipt),
        "cross_part_dedup_receipt_sha256": digest(args.dedup_receipt),
        "input_manifests": manifests,
        "train_shards": train_records,
        "validation_shards": val_records,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
