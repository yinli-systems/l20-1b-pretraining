#!/usr/bin/env python3
"""Filter, decontaminate, deduplicate, tokenize, and shard one source."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import re
import signal
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from source_docs import CODE_LANGUAGE_MIN_SCORES, REVISIONS, iter_source


WORD_RE = re.compile(r"[a-z0-9]+")
WORKER_CONTAMINATION_HASHES: set[bytes] = set()
WORKER_TOKENIZER: Tokenizer | None = None
WORKER_MAX_DOC_TOKENS = 0


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def load_contamination_hashes(path: Path | None) -> set[bytes]:
    if path is None:
        return set()
    connection = sqlite3.connect(path)
    try:
        return {row[0] for row in connection.execute("SELECT hash FROM ngrams")}
    finally:
        connection.close()


def load_document_hashes(path: Path | None) -> set[bytes]:
    if path is None:
        return set()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {row[0] for row in connection.execute("SELECT hash FROM documents")}
    finally:
        connection.close()


def is_contaminated(text: str, hashes: set[bytes], ngram_size: int = 13) -> bool:
    if not hashes:
        return False
    words = WORD_RE.findall(text.lower())
    if len(words) < ngram_size:
        return False
    for index in range(len(words) - ngram_size + 1):
        ngram = " ".join(words[index : index + ngram_size]).encode()
        if hashlib.blake2b(ngram, digest_size=8).digest() in hashes:
            return True
    return False


def worker_initialize(tokenizer_path: str, max_doc_tokens: int) -> None:
    global WORKER_TOKENIZER, WORKER_MAX_DOC_TOKENS
    # The parent owns durable checkpointing. Let an in-flight map finish so it
    # can flush all accepted token buffers before the process exits.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    WORKER_MAX_DOC_TOKENS = max_doc_tokens


def worker_prepare(text: str) -> tuple[str, bytes | None]:
    contaminated = is_contaminated(text, WORKER_CONTAMINATION_HASHES)
    if contaminated:
        return "contaminated", None
    if WORKER_TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    ids = WORKER_TOKENIZER.encode(text).ids
    if len(ids) > WORKER_MAX_DOC_TOKENS:
        return "too_long", None
    # Compact uint16 IPC avoids pickling millions of Python integers per batch.
    return "clean", np.asarray(ids, dtype=np.uint16).tobytes()


def batches(iterator, size: int):
    while batch := list(itertools.islice(iterator, size)):
        yield batch


class ShardWriter:
    def __init__(
        self,
        directory: Path,
        prefix: str,
        target_tokens: int,
        shard_tokens: int,
        block_size: int,
        allow_existing_excess: bool = False,
    ) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.target_tokens = target_tokens // block_size * block_size
        self.shard_tokens = shard_tokens // block_size * block_size
        self.block_size = block_size
        self.allow_existing_excess = allow_existing_excess
        self.buffer = np.empty(self.shard_tokens, dtype=np.uint16)
        self.buffered = 0
        self.written = 0
        self.shard_index = 0
        self.records: list[dict] = []
        self._load_existing_shards()

    def _load_existing_shards(self) -> None:
        paths = sorted(self.directory.glob(f"{self.prefix}-*.npy"))
        expected_indices = list(range(len(paths)))
        actual_indices: list[int] = []
        for path in paths:
            try:
                actual_indices.append(int(path.stem.rsplit("-", 1)[1]))
            except (IndexError, ValueError) as error:
                raise RuntimeError(f"invalid shard filename: {path}") from error
        if actual_indices != expected_indices:
            raise RuntimeError(
                f"non-contiguous {self.prefix} shards: expected {expected_indices}, got {actual_indices}"
            )

        for path in paths:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.uint16 or array.ndim != 1:
                raise RuntimeError(f"invalid shard format: {path} has dtype={array.dtype}, ndim={array.ndim}")
            tokens = int(array.size)
            if tokens == 0 or tokens % self.block_size:
                raise RuntimeError(f"invalid shard token count: {path} has {tokens}")
            sidecar = path.with_suffix(path.suffix + ".sha256.json")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if sidecar.is_file():
                receipt = json.loads(sidecar.read_text())
                if receipt.get("sha256") != digest or int(receipt.get("tokens", -1)) != tokens:
                    raise RuntimeError(f"shard receipt mismatch: {path}")
            else:
                _atomic_write_json(sidecar, {"path": str(path), "tokens": tokens, "sha256": digest})
            self.records.append({"path": str(path), "tokens": tokens, "sha256": digest})
            self.written += tokens

        if self.written > self.target_tokens and not self.allow_existing_excess:
            raise RuntimeError(
                f"existing {self.prefix} shards contain {self.written} tokens, above target {self.target_tokens}"
            )
        self.shard_index = len(paths)

    @property
    def complete(self) -> bool:
        return self.written + self.buffered >= self.target_tokens

    def append(self, tokens: list[int]) -> None:
        offset = 0
        while offset < len(tokens) and not self.complete:
            remaining_target = self.target_tokens - self.written - self.buffered
            count = min(len(tokens) - offset, self.shard_tokens - self.buffered, remaining_target)
            self.buffer[self.buffered : self.buffered + count] = tokens[offset : offset + count]
            self.buffered += count
            offset += count
            if self.buffered == self.shard_tokens or self.complete:
                self.flush()

    def flush(self) -> None:
        usable = self.buffered // self.block_size * self.block_size
        if usable == 0:
            return
        array = self.buffer[:usable].copy()
        path = self.directory / f"{self.prefix}-{self.shard_index:05d}.npy"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = {"path": str(path), "tokens": int(usable), "sha256": digest}
        _atomic_write_json(path.with_suffix(path.suffix + ".sha256.json"), record)
        self.records.append(record)
        self.written += usable
        self.shard_index += 1
        remainder = self.buffered - usable
        if remainder:
            self.buffer[:remainder] = self.buffer[usable : self.buffered]
        self.buffered = remainder


def open_dedup(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS documents (hash BLOB PRIMARY KEY, source TEXT NOT NULL)")
    return connection


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("web", "dclm", "synthetic", "math", "code"), required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dedup-db", type=Path, required=True)
    parser.add_argument("--decontam-db", type=Path)
    parser.add_argument("--exclude-hash-db", type=Path)
    parser.add_argument("--source-state-db", type=Path)
    parser.add_argument("--delete-completed-raw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--validation-tokens", type=int)
    parser.add_argument("--shard-tokens", type=int, default=2049 * 8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-doc-tokens", type=int, default=262_144)
    parser.add_argument("--holdout-modulus", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--batch-docs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--allow-existing-excess", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-source-exhaustion", action="store_true")
    parser.add_argument("--partition-label")
    args = parser.parse_args()
    if args.batch_docs < 1:
        raise SystemExit("--batch-docs must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=False)
    if vocab_size > np.iinfo(np.uint16).max:
        raise SystemExit(f"vocabulary {vocab_size} does not fit uint16")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer has no <|endoftext|> token")

    validation_tokens = args.validation_tokens
    if validation_tokens is None:
        validation_tokens = min(10_000_000, max(2049, args.target_tokens // 1000))
    source_dir = args.output_dir / args.source
    train_writer = ShardWriter(
        source_dir / "train-npy",
        "train",
        args.target_tokens,
        args.shard_tokens,
        2049,
        allow_existing_excess=args.allow_existing_excess,
    )
    val_writer = ShardWriter(
        source_dir / "val-npy",
        "val",
        validation_tokens,
        args.shard_tokens,
        2049,
        allow_existing_excess=args.allow_existing_excess,
    )
    contamination_hashes = load_contamination_hashes(args.decontam_db)
    excluded_hashes = load_document_hashes(args.exclude_hash_db)
    global WORKER_CONTAMINATION_HASHES
    WORKER_CONTAMINATION_HASHES = contamination_hashes
    process_pool = None
    if contamination_hashes and args.workers > 1:
        process_pool = mp.get_context("fork").Pool(
            args.workers,
            initializer=worker_initialize,
            initargs=(str(args.tokenizer_dir / "tokenizer.json"), args.max_doc_tokens),
        )
    dedup = open_dedup(args.dedup_db)
    # Per-document SQLite point reads are extremely slow on networked SSDs.
    # Keep the durable DB as the receipt/checkpoint and use an in-memory mirror
    # for membership checks during this process.
    seen_hashes = {row[0] for row in dedup.execute("SELECT hash FROM documents")}

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    resumed_tokens = {"train": train_writer.written, "validation": val_writer.written}
    counters = {"seen": 0, "accepted": 0, "duplicate": 0, "cross_partition_duplicate": 0,
                "contaminated": 0, "too_long": 0}
    pending_hashes: list[tuple[bytes, str]] = []
    pending_set: set[bytes] = set()
    try:
        source_rows = (
            ()
            if train_writer.complete and val_writer.complete
            else iter_source(
                args.source,
                seed=args.seed,
                state_db=args.source_state_db,
                delete_completed_raw=args.delete_completed_raw,
            )
        )
        for rows in batches(source_rows, args.batch_docs):
            prepared: list[tuple[str, bytes, int] | None] = []
            candidates: list[tuple[str, bytes]] = []
            for row in rows:
                text = row["text"]
                normalized = normalized_text(text)
                digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                if digest in excluded_hashes:
                    prepared.append("excluded")
                    continue
                if digest in seen_hashes or digest in pending_set:
                    prepared.append(None)
                    continue
                prepared.append((text, digest, len(candidates)))
                candidates.append((text, digest))

            texts = [text for text, _ in candidates]
            if process_pool is not None:
                chunk_size = max(1, len(texts) // (args.workers * 4))
                results = process_pool.map(worker_prepare, texts, chunksize=chunk_size)
            else:
                results = []
                for text in texts:
                    contaminated_flag = is_contaminated(text, contamination_hashes)
                    if contaminated_flag:
                        results.append(("contaminated", None))
                        continue
                    ids = tokenizer.encode(text).ids
                    results.append(("too_long", None) if len(ids) > args.max_doc_tokens else
                                   ("clean", np.asarray(ids,dtype=np.uint16).tobytes()))
            contaminated = [result[0] == "contaminated" for result in results]

            processed_rows = 0
            for prepared_row in prepared:
                counters["seen"] += 1
                processed_rows += 1
                if prepared_row == "excluded":
                    counters["cross_partition_duplicate"] += 1
                    continue
                if prepared_row is None:
                    counters["duplicate"] += 1
                    continue
                text, digest, candidate_index = prepared_row
                contaminated_flag = contaminated[candidate_index]
                # Preserve serial semantics for duplicates inside the current batch.
                if digest in pending_set:
                    counters["duplicate"] += 1
                    continue
                if contaminated_flag:
                    counters["contaminated"] += 1
                    continue

                status,payload = results[candidate_index]
                if status == "too_long":
                    counters["too_long"] += 1
                    continue
                if status != "clean" or payload is None:
                    raise RuntimeError(f"unexpected worker result: {status}")
                ids = np.frombuffer(payload,dtype=np.uint16)
                holdout = int.from_bytes(digest[:8], "big") % args.holdout_modulus == 0
                if holdout and not val_writer.complete:
                    val_writer.append(ids)
                    val_writer.append([eos_id])
                elif not train_writer.complete:
                    train_writer.append(ids)
                    train_writer.append([eos_id])
                pending_hashes.append((digest, args.source))
                pending_set.add(digest)
                seen_hashes.add(digest)
                counters["accepted"] += 1

                if len(pending_hashes) >= 1000:
                    dedup.executemany("INSERT OR IGNORE INTO documents(hash, source) VALUES (?, ?)", pending_hashes)
                    dedup.commit()
                    pending_hashes.clear()
                    pending_set.clear()
                if train_writer.complete and val_writer.complete:
                    break

            if counters["seen"] // args.progress_every != (counters["seen"] - processed_rows) // args.progress_every:
                progress = {
                    "source": args.source,
                    **counters,
                    "resumed_tokens": resumed_tokens,
                    "train_tokens": train_writer.written + train_writer.buffered,
                    "val_tokens": val_writer.written + val_writer.buffered,
                    "train_target": train_writer.target_tokens,
                    "validation_target": val_writer.target_tokens,
                }
                _atomic_write_json(source_dir / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if train_writer.complete and val_writer.complete:
                break
            if stop_requested:
                break
    finally:
        train_writer.flush()
        val_writer.flush()
        if pending_hashes:
            dedup.executemany("INSERT OR IGNORE INTO documents(hash, source) VALUES (?, ?)", pending_hashes)
            dedup.commit()
        dedup.close()
        if process_pool is not None:
            process_pool.close()
            process_pool.join()

    if stop_requested:
        progress = {
            "source": args.source,
            **counters,
            "resumed_tokens": resumed_tokens,
            "train_tokens": train_writer.written,
            "val_tokens": val_writer.written,
            "train_target": train_writer.target_tokens,
            "validation_target": val_writer.target_tokens,
            "checkpointed_stop": True,
        }
        _atomic_write_json(source_dir / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        raise SystemExit(130)

    targets_complete = train_writer.complete and val_writer.complete
    if not targets_complete and not args.allow_source_exhaustion:
        raise SystemExit(
            f"source exhausted before targets: train={train_writer.written}/{train_writer.target_tokens}, "
            f"val={val_writer.written}/{val_writer.target_tokens}"
        )
    manifest = {
        "source": args.source,
        "dataset_revisions": REVISIONS,
        "seed": args.seed,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "block_size": 2049,
        "counters": counters,
        "resumed_tokens": resumed_tokens,
        "train_tokens": train_writer.written,
        "validation_tokens": val_writer.written,
        "train_shards": train_writer.records,
        "validation_shards": val_writer.records,
        "decontamination_enabled": bool(contamination_hashes),
        "cross_partition_exclusion_enabled": bool(excluded_hashes),
        "cross_partition_exclusion_db": str(args.exclude_hash_db) if args.exclude_hash_db else None,
        "cross_partition_exclusion_db_sha256": (
            hashlib.sha256(args.exclude_hash_db.read_bytes()).hexdigest() if args.exclude_hash_db else None
        ),
        "holdout_modulus": args.holdout_modulus,
        "source_state_db": str(args.source_state_db) if args.source_state_db else None,
        "delete_completed_raw": args.delete_completed_raw,
        "batch_docs": args.batch_docs,
        "workers": args.workers,
        "partition_label": args.partition_label,
        "targets_complete": targets_complete,
        "source_exhausted": not targets_complete,
    }
    if args.source == "code":
        manifest["quality_filter"] = {
            "license_type": "permissive",
            "int_score_min_by_language": CODE_LANGUAGE_MIN_SCORES,
        }
    path = source_dir / "manifest.json"
    _atomic_write_json(path, manifest)
    print(
        json.dumps({"manifest": str(path), "train_tokens": train_writer.written, "val_tokens": val_writer.written}),
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
