#!/usr/bin/env python3
"""Build the frozen 150B-token MoE candidate pack, one verified input file at a time.

The build is deliberately single-writer.  Each input file is downloaded with its
frozen LFS identity, filtered, decontaminated, globally exact/near deduplicated,
tokenized, and committed with a content receipt.  Completed files are restartable;
an interrupted file is recomputed without advancing the durable state.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import ipaddress
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import signal
import sqlite3
import struct
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / "data" / "moe7b_150b_build_v1.json"
MASK64 = (1 << 64) - 1
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{16,}"
)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalized_document(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_match(text: str) -> str:
    return " ".join(WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold()))


def valid_ipv4_present(text: str) -> bool:
    for match in IPV4_RE.finditer(text):
        try:
            ipaddress.IPv4Address(match.group(0))
            return True
        except ipaddress.AddressValueError:
            pass
    return False


def redact_web_pii(text: str) -> tuple[str, bool]:
    changed = bool(EMAIL_RE.search(text))
    text = EMAIL_RE.sub("<EMAIL>", text)

    def replace_ip(match: re.Match[str]) -> str:
        nonlocal changed
        try:
            ipaddress.IPv4Address(match.group(0))
        except ipaddress.AddressValueError:
            return match.group(0)
        changed = True
        return "<IP_ADDRESS>"

    return IPV4_RE.sub(replace_ip, text), changed


def clean_text(value: Any, minimum_characters: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFC", value).replace("\x00", "").strip()
    if len(text) < minimum_characters:
        return None
    if text.count("\ufffd") > max(2, len(text) // 10_000):
        return None
    compact = "".join(text.split())
    if not compact:
        return None
    sample = compact[:100_000]
    if max(sample.count(character) for character in set(sample)) / len(sample) > 0.45:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 20 and len(set(lines)) / len(lines) < 0.2:
        return None
    return text


def minhash_constants(count: int = 16) -> tuple[tuple[int, int], ...]:
    result = []
    for index in range(count):
        digest = hashlib.sha256(f"moe7b-minhash-v1:{index}".encode()).digest()
        a = int.from_bytes(digest[:8], "big") | 1
        b = int.from_bytes(digest[8:16], "big")
        result.append((a, b))
    return tuple(result)


MINHASH_CONSTANTS = minhash_constants()
WORKER_CONTAMINATION: "ContaminationIndex | None" = None
WORKER_TOKENIZER: "Tokenizer | None" = None


def sampled_shingles(text: str, width: int = 5, maximum: int = 4096) -> Iterator[bytes]:
    words = WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    if len(words) >= width:
        total = len(words) - width + 1
        step = max(1, math.ceil(total / maximum))
        for index in range(0, total, step):
            yield " ".join(words[index : index + width]).encode("utf-8")
        return
    compact = "".join(words)
    if len(compact) < width:
        return
    total = len(compact) - width + 1
    step = max(1, math.ceil(total / maximum))
    for index in range(0, total, step):
        yield compact[index : index + width].encode("utf-8")


def minhash_signature(text: str) -> tuple[int, ...] | None:
    signature = [MASK64] * len(MINHASH_CONSTANTS)
    observed = 0
    for shingle in sampled_shingles(text):
        observed += 1
        value = int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        for index, (a, b) in enumerate(MINHASH_CONSTANTS):
            candidate = (a * value + b) & MASK64
            if candidate < signature[index]:
                signature[index] = candidate
    return tuple(signature) if observed else None


def pack_signature(signature: tuple[int, ...] | None) -> bytes:
    if signature is None:
        return b""
    return struct.pack(">16Q", *signature)


def unpack_signature(value: bytes) -> tuple[int, ...] | None:
    if not value:
        return None
    if len(value) != 16 * 8:
        raise ValueError(f"invalid minhash signature length: {len(value)}")
    return struct.unpack(">16Q", value)


def signature_bands(signature: tuple[int, ...] | None) -> list[bytes]:
    if signature is None:
        return []
    bands = []
    for index in range(4):
        payload = struct.pack(">4Q", *signature[index * 4 : (index + 1) * 4])
        bands.append(bytes([index]) + hashlib.blake2b(payload, digest_size=8).digest())
    return bands


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return sum(a == b for a, b in zip(left, right)) / len(left)


class ContaminationIndex:
    def __init__(self, path: Path, expected_sha256: str, width: int) -> None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(f"contamination index SHA-256 mismatch: {actual} != {expected_sha256}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("plaintext_prompts_in_index") is not False:
            raise ValueError("contamination index must not contain plaintext prompts")
        self.scope_id = payload["scope_id"]
        self.width = width
        self.exact = {bytes.fromhex(value) for value in payload["exact_prompt_hashes"]}
        self.ngrams = {bytes.fromhex(value) for value in payload["ngram_13_hashes"]}

    def contaminated(self, text: str) -> bool:
        normalized = normalized_match(text)
        if hashlib.sha256(normalized.encode()).digest() in self.exact:
            return True
        words = normalized.split()
        for index in range(len(words) - self.width + 1):
            value = " ".join(words[index : index + self.width]).encode()
            if hashlib.sha256(value).digest() in self.ngrams:
                return True
        return False


def analyze_text(text: str) -> tuple[bytes, bool, tuple[int, ...] | None]:
    if WORKER_CONTAMINATION is None:
        raise RuntimeError("worker contamination index is not initialized")
    return (
        hashlib.sha256(normalized_document(text).encode()).digest(),
        WORKER_CONTAMINATION.contaminated(text),
        minhash_signature(text),
    )


def analyze_and_tokenize_batch(
    texts: list[str],
) -> list[tuple[bytes, bool, tuple[int, ...] | None, list[int]]]:
    """Run the expensive stateless work in worker processes, preserving order.

    Ordered ``Pool.imap`` lets the single writer apply global exact/near-dedup
    decisions in the frozen component/file/document order while parsing the
    next input batches, MinHash/contamination analysis, and tokenization overlap.
    """
    if WORKER_TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    analyses = [analyze_text(text) for text in texts]
    encodings = WORKER_TOKENIZER.encode_batch(texts, add_special_tokens=False)
    return [
        (digest, contaminated, signature, encoding.ids)
        for (digest, contaminated, signature), encoding in zip(analyses, encodings)
    ]


def analyze_batch(texts: list[str]) -> list[tuple[bytes, bool, tuple[int, ...] | None]]:
    return [analyze_text(text) for text in texts]


def batches(iterator: Iterator[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for value in iterator:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def open_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            hash BLOB PRIMARY KEY,
            component TEXT NOT NULL,
            signature BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS near_bands (
            band BLOB NOT NULL,
            hash BLOB NOT NULL,
            PRIMARY KEY (band, hash),
            FOREIGN KEY (hash) REFERENCES documents(hash)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS near_bands_band ON near_bands(band);
        CREATE TABLE IF NOT EXISTS completed_files (
            component TEXT NOT NULL,
            repository_path TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            receipt_path TEXT NOT NULL,
            train_blocks INTEGER NOT NULL,
            validation_blocks INTEGER NOT NULL,
            PRIMARY KEY (component, repository_path)
        );
        """
    )
    connection.commit()
    return connection


def completed_record(connection: sqlite3.Connection, component: str, repository_path: str) -> tuple | None:
    return connection.execute(
        "SELECT input_sha256, receipt_path, train_blocks, validation_blocks "
        "FROM completed_files WHERE component = ? AND repository_path = ?",
        (component, repository_path),
    ).fetchone()


def exact_hash_exists(connection: sqlite3.Connection, digest: bytes) -> bool:
    return connection.execute("SELECT 1 FROM documents WHERE hash = ?", (digest,)).fetchone() is not None


def query_chunks(values: list[bytes], size: int = 400) -> Iterator[list[bytes]]:
    """Keep batched BLOB lookups below SQLite's conservative variable limit."""

    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def existing_exact_hashes(
    connection: sqlite3.Connection, digests: Iterable[bytes]
) -> set[bytes]:
    """Return exact document hashes with one indexed query per bounded chunk."""

    unique = list(dict.fromkeys(digests))
    existing: set[bytes] = set()
    for chunk in query_chunks(unique):
        placeholders = ",".join("?" for _ in chunk)
        existing.update(
            bytes(row[0])
            for row in connection.execute(
                f"SELECT hash FROM documents WHERE hash IN ({placeholders})", chunk
            )
        )
    return existing


def near_hash_exists(
    connection: sqlite3.Connection,
    signature: tuple[int, ...] | None,
    hot_bucket_limit: int = 256,
) -> tuple[bool, bool]:
    if signature is None:
        return False, False
    candidates: dict[bytes, bytes] = {}
    for band in signature_bands(signature):
        rows = connection.execute(
            "SELECT d.hash, d.signature FROM near_bands b JOIN documents d ON d.hash = b.hash "
            "WHERE b.band = ? LIMIT ?",
            (band, hot_bucket_limit),
        ).fetchall()
        if len(rows) >= hot_bucket_limit:
            return True, True
        candidates.update(rows)
    for packed in candidates.values():
        candidate = unpack_signature(packed)
        if candidate is not None and signature_similarity(signature, candidate) >= 0.75:
            return True, False
    return False, False


def near_hash_statuses(
    connection: sqlite3.Connection,
    entries: Iterable[tuple[bytes, tuple[int, ...] | None]],
    hot_bucket_limit: int = 256,
) -> dict[bytes, tuple[bool, bool]]:
    """Match scalar near-dedup semantics with batched indexed band lookups."""

    values = list(entries)
    result = {digest: (False, False) for digest, _ in values}
    band_owners: dict[bytes, list[tuple[bytes, tuple[int, ...]]]] = defaultdict(list)
    for digest, signature in values:
        if signature is None:
            continue
        for band in signature_bands(signature):
            band_owners[band].append((digest, signature))
    bands = list(band_owners)
    if not bands:
        return result

    hot_bands: set[bytes] = set()
    for chunk in query_chunks(bands):
        placeholders = ",".join("?" for _ in chunk)
        hot_bands.update(
            bytes(row[0])
            for row in connection.execute(
                f"SELECT band FROM near_bands WHERE band IN ({placeholders}) "
                "GROUP BY band HAVING COUNT(*) >= ?",
                [*chunk, hot_bucket_limit],
            )
        )
    for band in hot_bands:
        for digest, _ in band_owners[band]:
            result[digest] = (True, True)

    candidate_signatures: dict[bytes, dict[bytes, bytes]] = defaultdict(dict)
    regular_bands = [band for band in bands if band not in hot_bands]
    for chunk in query_chunks(regular_bands):
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            "SELECT b.band, d.hash, d.signature FROM near_bands b "
            "JOIN documents d ON d.hash = b.hash "
            f"WHERE b.band IN ({placeholders})",
            chunk,
        )
        for band_value, candidate_hash, packed in rows:
            band = bytes(band_value)
            for digest, _ in band_owners[band]:
                if not result[digest][1]:
                    candidate_signatures[digest][bytes(candidate_hash)] = bytes(packed)
    signatures = dict(values)
    for digest, candidates in candidate_signatures.items():
        signature = signatures[digest]
        if signature is None:
            continue
        if any(
            candidate is not None
            and signature_similarity(signature, candidate) >= 0.75
            for candidate in (unpack_signature(packed) for packed in candidates.values())
        ):
            result[digest] = (True, False)
    return result


def input_files(component: dict[str, Any], lock: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    prefix = component["path_prefix"]
    files = [
        record
        for record in lock["files"]
        if record["path"].startswith(prefix)
        and not any(part.casefold() in {"dev", "eval", "evaluation", "test", "valid", "validation"}
                    for part in PurePosixPath(record["path"]).parts)
    ]
    if not files:
        raise ValueError(f"component {component['component_id']} selected no frozen files")
    keyed = []
    for record in files:
        key = hashlib.sha256(f"{seed}:{component['component_id']}:{record['path']}".encode()).digest()
        keyed.append((key, record["path"], record))
    return [record for _, _, record in sorted(keyed)]


def safe_cache_path(root: Path, source_id: str, repository_path: str) -> Path:
    parts = PurePosixPath(repository_path).parts
    if repository_path.startswith("/") or ".." in parts:
        raise ValueError(f"unsafe repository path: {repository_path}")
    return root / source_id / PurePosixPath(repository_path)


def download_verified(
    endpoint: str,
    raw_root: Path,
    source_id: str,
    dataset_id: str,
    revision: str,
    record: dict[str, Any],
    offline: bool = False,
    source_wait_seconds: int = 0,
) -> Path:
    path = safe_cache_path(raw_root, source_id, record["path"])
    expected_size = int(record["size"])
    expected_hash = record["lfs_sha256"]
    marker = path.with_suffix(path.suffix + ".verified.json")
    def already_verified() -> bool:
        if path.is_file() and marker.is_file() and path.stat().st_size == expected_size:
            receipt = json.loads(marker.read_text())
            return receipt.get("sha256") == expected_hash and receipt.get("bytes") == expected_size
        return False

    if already_verified():
        return path
    if offline:
        deadline = time.monotonic() + source_wait_seconds
        while time.monotonic() < deadline:
            if already_verified():
                return path
            time.sleep(10)
        raise TimeoutError(f"verified source file did not arrive before offline wait deadline: {record['path']}")
    if path.exists() and path.stat().st_size == expected_size and sha256_file(path) == expected_hash:
        atomic_json(marker, {"bytes": expected_size, "sha256": expected_hash, "status": "VERIFIED_NOT_ADMITTED"})
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    url = f"{endpoint}/datasets/{dataset_id}/resolve/{revision}/{quote(record['path'], safe='/')}"
    subprocess.run(
        [
            "curl", "--fail", "--location", "--silent", "--show-error", "--continue-at", "-",
            "--retry", "20", "--retry-all-errors", "--retry-delay", "5", "--connect-timeout", "30",
            "--speed-limit", "1024", "--speed-time", "180", "--output", str(partial), url,
        ],
        check=True,
    )
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"download size mismatch for {record['path']}")
    actual = sha256_file(partial)
    if actual != expected_hash:
        raise RuntimeError(f"download SHA-256 mismatch for {record['path']}: {actual} != {expected_hash}")
    os.replace(partial, path)
    atomic_json(
        marker,
        {
            "schema": "moe7b-source-file-receipt-v1",
            "dataset_id": dataset_id,
            "revision": revision,
            "repository_path": record["path"],
            "bytes": expected_size,
            "sha256": actual,
            "status": "VERIFIED_NOT_ADMITTED",
        },
    )
    return path


def iter_rows(path: Path, format_name: str, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    if format_name in {"parquet", "stack-v3-parquet"}:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path, memory_map=True)
        try:
            available = set(parquet.schema_arrow.names)
            selected = [column for column in (columns or list(available)) if column in available]
            for batch in parquet.iter_batches(batch_size=256, columns=selected, use_threads=True):
                yield from batch.to_pylist()
        finally:
            parquet.close()
        return
    if format_name == "json.gz":
        context: Any = gzip.open(path, "rt", encoding="utf-8")
    elif format_name == "jsonl.zst":
        import zstandard

        raw = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        context = io.TextIOWrapper(reader, encoding="utf-8")
    else:
        raise ValueError(f"unsupported input format: {format_name}")
    with context as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            yield value


def source_documents(
    path: Path,
    component: dict[str, Any],
    allowlist: set[str],
    counters: Counter[str],
    minimum_characters: int,
    code_minimum_characters: int,
) -> Iterator[str]:
    format_name = component["format"]
    columns = ["text", "language", "language_score", "int_score", "repo_path", "commit_id", "files"]
    for row in iter_rows(path, format_name, columns):
        counters["rows_seen"] += 1
        if format_name == "stack-v3-parquet":
            files = row.get("files")
            if not isinstance(files, list):
                counters["malformed_nested_files"] += 1
                continue
            for record in files:
                counters["documents_seen"] += 1
                if not isinstance(record, dict):
                    counters["malformed_document"] += 1
                    continue
                licenses = record.get("detected_licenses")
                if record.get("is_vendor") is True:
                    counters["vendor_reject"] += 1
                    continue
                if (
                    record.get("license_type") != "permissive"
                    or not isinstance(licenses, list)
                    or not licenses
                    or any(not isinstance(item, str) or item not in allowlist for item in licenses)
                ):
                    counters["license_reject"] += 1
                    continue
                text = clean_text(record.get("content"), code_minimum_characters)
                if text is None:
                    counters["quality_reject"] += 1
                    continue
                if EMAIL_RE.search(text) or valid_ipv4_present(text) or SECRET_RE.search(text):
                    counters["code_privacy_or_secret_reject"] += 1
                    continue
                yield text
            continue

        counters["documents_seen"] += 1
        minimum_score = component.get("minimum_int_score")
        if minimum_score is not None and int(row.get("int_score") or 0) < int(minimum_score):
            counters["quality_score_reject"] += 1
            continue
        minimum_language_score = component.get("minimum_language_score")
        if minimum_language_score is not None and float(row.get("language_score") or 0.0) < float(minimum_language_score):
            counters["language_score_reject"] += 1
            continue
        text = clean_text(row.get("text"), minimum_characters)
        if text is None:
            counters["quality_reject"] += 1
            continue
        text, changed = redact_web_pii(text)
        if changed:
            counters["pii_redacted"] += 1
        yield text


def write_uint16(handle, ids: list[int]) -> None:
    if ids:
        np.asarray(ids, dtype="<u2").tofile(handle)


def trim_to_blocks(path: Path, block_tokens: int) -> tuple[int, int]:
    token_count = path.stat().st_size // 2
    usable = token_count // block_tokens * block_tokens
    discarded = token_count - usable
    with path.open("r+b") as handle:
        handle.truncate(usable * 2)
        handle.flush()
        os.fsync(handle.fileno())
    return usable // block_tokens, discarded


def read_dedup_records(path: Path) -> Iterator[tuple[bytes, bytes]]:
    record_size = 32 + 16 * 8
    with path.open("rb") as handle:
        while record := handle.read(record_size):
            if len(record) != record_size:
                raise ValueError(f"truncated dedup record file: {path}")
            yield record[:32], record[32:]


def dedup_record_batches(
    path: Path, batch_records: int = 4096
) -> Iterator[list[tuple[bytes, bytes]]]:
    """Yield bounded record groups for bulk SQLite insertion."""

    batch: list[tuple[bytes, bytes]] = []
    for record in read_dedup_records(path):
        batch.append(record)
        if len(batch) == batch_records:
            yield batch
            batch = []
    if batch:
        yield batch


def commit_receipt(connection: sqlite3.Connection, receipt_path: Path) -> None:
    receipt = json.loads(receipt_path.read_text())
    component = receipt["component_id"]
    repository_path = receipt["input"]["repository_path"]
    existing = completed_record(connection, component, repository_path)
    if existing is not None:
        if existing[0] != receipt["input"]["sha256"]:
            raise ValueError(f"completed input identity changed for {repository_path}")
        return
    for artifact in receipt["artifacts"].values():
        path = Path(artifact["path"])
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"artifact identity mismatch while reconciling: {path}")
    dedup_path = Path(receipt["artifacts"]["dedup_records"]["path"])
    connection.execute("BEGIN IMMEDIATE")
    try:
        for records in dedup_record_batches(dedup_path):
            document_rows = []
            band_rows = []
            for digest, packed in records:
                empty = packed == b"\x00" * 128
                document_rows.append(
                    (digest, component, b"" if empty else packed)
                )
                signature = None if empty else unpack_signature(packed)
                band_rows.extend(
                    (band, digest) for band in signature_bands(signature)
                )
            connection.executemany(
                "INSERT OR IGNORE INTO documents(hash, component, signature) VALUES (?, ?, ?)",
                document_rows,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO near_bands(band, hash) VALUES (?, ?)",
                band_rows,
            )
        connection.execute(
            "INSERT INTO completed_files(component, repository_path, input_sha256, receipt_path, train_blocks, validation_blocks) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                component,
                repository_path,
                receipt["input"]["sha256"],
                str(receipt_path),
                receipt["train_blocks"],
                receipt["validation_blocks"],
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def process_file(
    path: Path,
    record: dict[str, Any],
    component: dict[str, Any],
    output_root: Path,
    connection: sqlite3.Connection,
    tokenizer: Tokenizer,
    eos_id: int,
    contamination: ContaminationIndex,
    config: dict[str, Any],
    allowlist: set[str],
    ordinal: int,
    stop_requested: list[bool],
    workers: int,
    batch_documents: int,
    pipeline_tokenization: bool,
    pipeline_analysis: bool,
) -> Path:
    component_root = output_root / "packed" / component["component_id"]
    component_root.mkdir(parents=True, exist_ok=True)
    stem = f"part-{ordinal:05d}-{record['lfs_sha256'][:12]}"
    receipt_path = component_root / f"{stem}.receipt.json"
    if receipt_path.is_file():
        commit_receipt(connection, receipt_path)
        return receipt_path

    train_tmp = component_root / f".{stem}.train.bin.tmp-{os.getpid()}"
    val_tmp = component_root / f".{stem}.validation.bin.tmp-{os.getpid()}"
    dedup_tmp = component_root / f".{stem}.dedup.bin.tmp-{os.getpid()}"
    counters: Counter[str] = Counter()
    pending_hashes: set[bytes] = set()
    pending_signatures: list[tuple[bytes, tuple[int, ...] | None]] = []
    pending_bands: dict[bytes, list[tuple[int, ...]]] = defaultdict(list)
    block_tokens = int(config["block_tokens"])
    validation_modulus = int(config["validation_modulus"])
    maximum_document_tokens = int(config["maximum_document_tokens"])
    started = time.time()

    def locally_near(signature: tuple[int, ...] | None) -> bool:
        if signature is None:
            return False
        candidates: list[tuple[int, ...]] = []
        for band in signature_bands(signature):
            candidates.extend(pending_bands.get(band, ()))
        return any(signature_similarity(signature, other) >= 0.75 for other in candidates)

    global WORKER_TOKENIZER
    WORKER_TOKENIZER = tokenizer if pipeline_tokenization else None
    if pipeline_tokenization:
        # One tokenizer per worker process; nested Rayon pools would otherwise
        # oversubscribe the six CPUs assigned by ParaCloud per RTX 4090.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["RAYON_NUM_THREADS"] = "1"
    process_pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    try:
        with train_tmp.open("wb") as train_handle, val_tmp.open("wb") as val_handle, dedup_tmp.open("wb") as dedup_handle:
            documents = source_documents(
                path,
                component,
                allowlist,
                counters,
                int(config["minimum_document_characters"]),
                int(config["code_minimum_document_characters"]),
            )
            text_batches = batches(documents, batch_documents)
            if pipeline_tokenization:
                if process_pool is None:
                    analyzed_batches = map(analyze_and_tokenize_batch, text_batches)
                else:
                    analyzed_batches = process_pool.imap(
                        analyze_and_tokenize_batch, text_batches, chunksize=1
                    )
            else:
                analyzed_batches = None

            def ordered_analysis_pipeline():
                if process_pool is None:
                    for texts in text_batches:
                        yield texts, analyze_batch(texts)
                    return
                pending = deque()
                exhausted = False

                def refill() -> None:
                    nonlocal exhausted
                    while not exhausted and len(pending) < workers * 2:
                        try:
                            texts = next(text_batches)
                        except StopIteration:
                            exhausted = True
                            break
                        pending.append((texts, process_pool.apply_async(analyze_batch, (texts,))))

                refill()
                while pending:
                    texts, result = pending.popleft()
                    analyses = result.get()
                    refill()
                    yield texts, analyses

            analysis_batches = ordered_analysis_pipeline() if pipeline_analysis else None

            while True:
                if stop_requested[0]:
                    raise InterruptedError("stop requested before input file commit")
                if pipeline_tokenization:
                    assert analyzed_batches is not None
                    try:
                        analyzed = next(analyzed_batches)
                    except StopIteration:
                        break
                elif pipeline_analysis:
                    assert analysis_batches is not None
                    try:
                        texts, analyses = next(analysis_batches)
                    except StopIteration:
                        break
                    encodings = tokenizer.encode_batch(
                        texts, add_special_tokens=False
                    ) if texts else []
                    analyzed = [
                        (digest, contaminated, signature, encoding.ids)
                        for (digest, contaminated, signature), encoding in zip(analyses, encodings)
                    ]
                else:
                    try:
                        texts = next(text_batches)
                    except StopIteration:
                        break
                    if process_pool is None:
                        analyses = [analyze_text(text) for text in texts]
                    else:
                        analyses = process_pool.map(
                            analyze_text,
                            texts,
                            chunksize=max(1, len(texts) // (workers * 2)),
                        )
                    encodings = tokenizer.encode_batch(
                        texts, add_special_tokens=False
                    ) if texts else []
                    analyzed = [
                        (digest, contaminated, signature, encoding.ids)
                        for (digest, contaminated, signature), encoding in zip(analyses, encodings)
                    ]
                unresolved = []
                batch_hashes: set[bytes] = set()
                for digest, contaminated, signature, token_ids in analyzed:
                    if digest in pending_hashes or digest in batch_hashes:
                        counters["exact_duplicate_reject"] += 1
                        continue
                    batch_hashes.add(digest)
                    unresolved.append((digest, contaminated, signature, token_ids))
                exact_hashes = existing_exact_hashes(
                    connection, (digest for digest, _, _, _ in unresolved)
                )
                near_inputs = []
                for digest, contaminated, signature, _ in unresolved:
                    if digest in exact_hashes:
                        counters["exact_duplicate_reject"] += 1
                        continue
                    if contaminated:
                        counters["contamination_reject"] += 1
                        continue
                    near_inputs.append((digest, signature))
                near_status = near_hash_statuses(connection, near_inputs)
                candidates = []
                for digest, contaminated, signature, token_ids in unresolved:
                    if digest in exact_hashes or contaminated:
                        continue
                    near, hot = near_status[digest]
                    if near:
                        counters["near_hot_bucket_reject" if hot else "near_duplicate_reject"] += 1
                        continue
                    candidates.append((digest, signature, token_ids))
                for digest, signature, ids in candidates:
                    if len(ids) > maximum_document_tokens:
                        counters["too_long_reject"] += 1
                        continue
                    if locally_near(signature):
                        counters["near_duplicate_reject"] += 1
                        continue
                    ids.append(eos_id)
                    holdout = int.from_bytes(digest[:8], "big") % validation_modulus == 0
                    write_uint16(val_handle if holdout else train_handle, ids)
                    packed = pack_signature(signature)
                    dedup_handle.write(digest)
                    dedup_handle.write(packed if packed else b"\x00" * 128)
                    pending_hashes.add(digest)
                    pending_signatures.append((digest, signature))
                    for band in signature_bands(signature):
                        pending_bands[band].append(signature)  # type: ignore[arg-type]
                    counters["accepted_documents"] += 1
                    counters["accepted_token_ids"] += len(ids)
            for handle in (train_handle, val_handle, dedup_handle):
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        if process_pool is not None:
            process_pool.close()
            process_pool.join()

    train_blocks, train_discarded = trim_to_blocks(train_tmp, block_tokens)
    validation_blocks, validation_discarded = trim_to_blocks(val_tmp, block_tokens)
    train_path = component_root / f"{stem}.train.bin"
    validation_path = component_root / f"{stem}.validation.bin"
    dedup_path = component_root / f"{stem}.dedup.bin"
    os.replace(train_tmp, train_path)
    os.replace(val_tmp, validation_path)
    os.replace(dedup_tmp, dedup_path)
    artifacts = {}
    for name, artifact_path in (
        ("train", train_path), ("validation", validation_path), ("dedup_records", dedup_path)
    ):
        artifacts[name] = {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
        }
    receipt = {
        "schema": "moe7b-packed-input-file-receipt-v1",
        "status": "PACKED_CANDIDATE_NOT_ADMITTED",
        "component_id": component["component_id"],
        "aggregate_source": component["aggregate_source"],
        "input": {
            "repository_path": record["path"],
            "local_path": str(path),
            "bytes": record["size"],
            "sha256": record["lfs_sha256"],
        },
        "artifacts": artifacts,
        "train_blocks": train_blocks,
        "validation_blocks": validation_blocks,
        "processing_mode": (
            "ordered_pipeline_tokenization_v1"
            if pipeline_tokenization
            else "ordered_pipeline_analysis_v1"
            if pipeline_analysis
            else "serial_tokenization_v1"
        ),
        "dedup_lookup_mode": "batched_sqlite_exact_and_lsh_v1",
        "train_prediction_tokens": train_blocks * int(config["prediction_tokens_per_block"]),
        "validation_prediction_tokens": validation_blocks * int(config["prediction_tokens_per_block"]),
        "discarded_tail_token_ids": {"train": train_discarded, "validation": validation_discarded},
        "counters": dict(sorted(counters.items())),
        "started_unix": started,
        "finished_unix": time.time(),
        "claim_boundary": "Source identity and mechanical filters passed. This shard remains unadmitted pending every gate in the frozen mixture plan.",
    }
    atomic_json(receipt_path, receipt)
    commit_receipt(connection, receipt_path)
    return receipt_path


def component_totals(connection: sqlite3.Connection, component_id: str) -> tuple[int, int, int]:
    row = connection.execute(
        "SELECT COALESCE(SUM(train_blocks),0), COALESCE(SUM(validation_blocks),0), COUNT(*) "
        "FROM completed_files WHERE component = ?",
        (component_id,),
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2])


def write_progress(
    output_root: Path,
    config: dict[str, Any],
    connection: sqlite3.Connection,
    active_component: str | None,
    status: str,
) -> dict[str, Any]:
    prediction_tokens_per_block = int(config["prediction_tokens_per_block"])
    components = []
    for component in config["components"]:
        train_blocks, validation_blocks, files = component_totals(connection, component["component_id"])
        observed = train_blocks * prediction_tokens_per_block
        target = int(component["target_prediction_tokens"])
        components.append(
            {
                "component_id": component["component_id"],
                "aggregate_source": component["aggregate_source"],
                "target_prediction_tokens": target,
                "available_train_prediction_tokens": observed,
                "available_validation_prediction_tokens": validation_blocks * prediction_tokens_per_block,
                "completed_input_files": files,
                "target_reached": observed >= target,
            }
        )
    report = {
        "schema": "moe7b-full-data-build-progress-v1",
        "status": status,
        "active_component": active_component,
        "updated_unix": time.time(),
        "components": components,
        "target_prediction_tokens": sum(item["target_prediction_tokens"] for item in components),
        "available_train_prediction_tokens": sum(item["available_train_prediction_tokens"] for item in components),
        "all_targets_reached": all(item["target_reached"] for item in components),
        "training_admitted": False,
    }
    atomic_json(output_root / "progress.json", report)
    return report


def validate_config(config: dict[str, Any], plan: dict[str, Any]) -> None:
    if config.get("schema") != "moe7b-full-data-build-v1":
        raise ValueError("unsupported build schema")
    component_total = sum(int(item["target_prediction_tokens"]) for item in config["components"])
    if component_total != int(config["completion_contract"]["required_prediction_tokens"]):
        raise ValueError("component token targets do not match completion contract")
    if component_total != int(plan["budget"]["target_exposed_tokens"]):
        raise ValueError("build target does not match mixture plan")
    aggregate = Counter()
    for component in config["components"]:
        aggregate[component["aggregate_source"]] += int(component["target_prediction_tokens"])
    expected = {item["source_id"]: int(item["target_tokens"]) for item in plan["sources"]}
    if dict(aggregate) != expected:
        raise ValueError(f"aggregate source targets differ from mixture plan: {dict(aggregate)} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--contamination-index", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--offline", action="store_true", help="Wait for an external verified-download feeder")
    parser.add_argument("--source-wait-seconds", type=int, default=86_400)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--batch-documents", type=int, default=32)
    parser.add_argument(
        "--pipeline-tokenization",
        action="store_true",
        help="Overlap ordered parsing, analysis, and tokenization while retaining one writer",
    )
    parser.add_argument(
        "--pipeline-analysis",
        action="store_true",
        help="Overlap ordered parsing/analysis with parent tokenization while retaining one writer",
    )
    parser.add_argument("--component", action="append", help="Only process these component ids, in contract order")
    parser.add_argument("--preflight", action="store_true", help="Validate identities and runtime without downloading")
    args = parser.parse_args()

    config = json.loads(args.build.read_text())
    if args.workers < 1 or args.batch_documents < 1:
        raise ValueError("workers and batch-documents must be positive")
    if args.pipeline_tokenization and args.pipeline_analysis:
        raise ValueError("pipeline tokenization and pipeline analysis are mutually exclusive")
    plan_path = ROOT / config["mixture_plan"]
    plan = json.loads(plan_path.read_text())
    validate_config(config, plan)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=False) != plan["tokenizer"]["vocabulary_size"]:
        raise ValueError("tokenizer vocabulary does not match frozen plan")
    actual_tokenizer_hash = sha256_file(args.tokenizer_dir / "tokenizer.json")
    expected_tokenizer_hash = plan["tokenizer"]["files"]["tokenizer.json"]
    if actual_tokenizer_hash != expected_tokenizer_hash:
        raise ValueError(f"tokenizer identity mismatch: {actual_tokenizer_hash} != {expected_tokenizer_hash}")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id != plan["tokenizer"]["eos_token_id"]:
        raise ValueError(f"EOS id mismatch: {eos_id}")
    lock_root = ROOT / config["source_lock_directory"]
    locks = {}
    for source in plan["sources"]:
        path = lock_root / f"{source['source_id']}.json"
        lock = json.loads(path.read_text())
        if lock["requested_revision"] != source["revision"] or lock["dataset_id"] != source["dataset_id"]:
            raise ValueError(f"source lock mismatch for {source['source_id']}")
        locks[source["source_id"]] = lock
    allowlist_payload = json.loads((ROOT / "data" / "code_license_allowlist_v1.json").read_text())
    allowlist = set(allowlist_payload["allowed_spdx_identifiers"])
    contamination = ContaminationIndex(
        args.contamination_index,
        config["decontamination"]["index_sha256"],
        int(config["decontamination"]["ngram_width"]),
    )
    global WORKER_CONTAMINATION
    WORKER_CONTAMINATION = contamination
    selected = set(args.component or [item["component_id"] for item in config["components"]])
    unknown = selected - {item["component_id"] for item in config["components"]}
    if unknown:
        raise ValueError(f"unknown component ids: {sorted(unknown)}")
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "PASS_PREFLIGHT_NOT_STARTED_NOT_ADMITTED",
                    "build_id": config["build_id"],
                    "selected_components": sorted(selected),
                    "tokenizer_sha256": actual_tokenizer_hash,
                    "contamination_scope_id": contamination.scope_id,
                    "output_free_bytes": shutil.disk_usage(args.output_root.parent).free,
                    "raw_cache_free_bytes": shutil.disk_usage(args.raw_root.parent).free,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    state = open_state(args.output_root / "state" / "build.sqlite")
    stop_requested = [False]

    def request_stop(_signum, _frame) -> None:
        stop_requested[0] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        write_progress(args.output_root, config, state, None, "RUNNING_NOT_ADMITTED")
        for component in config["components"]:
            component_id = component["component_id"]
            if component_id not in selected:
                continue
            lock = locks[component["source_id"]]
            files = input_files(component, lock, int(config["selection_seed"]))
            target = int(component["target_prediction_tokens"])
            for ordinal, record in enumerate(files):
                train_blocks, _, _ = component_totals(state, component_id)
                available = train_blocks * int(config["prediction_tokens_per_block"])
                if available >= target:
                    break
                if stop_requested[0]:
                    write_progress(args.output_root, config, state, component_id, "CHECKPOINTED_STOP_NOT_ADMITTED")
                    return 130
                existing = completed_record(state, component_id, record["path"])
                if existing is not None:
                    if existing[0] != record["lfs_sha256"]:
                        raise ValueError(f"completed source identity changed: {record['path']}")
                    continue
                source_path = download_verified(
                    args.endpoint.rstrip("/"),
                    args.raw_root,
                    component["source_id"],
                    lock["dataset_id"],
                    lock["requested_revision"],
                    record,
                    offline=args.offline,
                    source_wait_seconds=args.source_wait_seconds,
                )
                receipt = process_file(
                    source_path,
                    record,
                    component,
                    args.output_root,
                    state,
                    tokenizer,
                    int(eos_id),
                    contamination,
                    config,
                    allowlist,
                    ordinal,
                    stop_requested,
                    args.workers,
                    args.batch_documents,
                    args.pipeline_tokenization,
                    args.pipeline_analysis,
                )
                progress = write_progress(args.output_root, config, state, component_id, "RUNNING_NOT_ADMITTED")
                print(
                    json.dumps(
                        {
                            "component": component_id,
                            "input": record["path"],
                            "receipt": str(receipt),
                            "available_train_prediction_tokens": next(
                                item["available_train_prediction_tokens"]
                                for item in progress["components"]
                                if item["component_id"] == component_id
                            ),
                            "target_prediction_tokens": target,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            train_blocks, _, _ = component_totals(state, component_id)
            available = train_blocks * int(config["prediction_tokens_per_block"])
            if available < target:
                write_progress(args.output_root, config, state, component_id, "BLOCKED_SOURCE_EXHAUSTED_NOT_ADMITTED")
                raise RuntimeError(f"component exhausted below target: {component_id} {available}/{target}")
        final = write_progress(args.output_root, config, state, None, "PACK_COMPLETE_NOT_ADMITTED")
        if selected == {item["component_id"] for item in config["components"]} and not final["all_targets_reached"]:
            raise RuntimeError("full build returned without reaching every component target")
        return 0
    except InterruptedError:
        write_progress(args.output_root, config, state, None, "CHECKPOINTED_STOP_NOT_ADMITTED")
        return 130
    finally:
        state.close()


if __name__ == "__main__":
    raise SystemExit(main())
