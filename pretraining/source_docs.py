#!/usr/bin/env python3
"""Pinned, quality-filtered document streams used by tokenizer and data builds."""

from __future__ import annotations

import gzip
import hashlib
import ipaddress
import itertools
import json
import os
import random
import re
import sqlite3
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests


HF_ENDPOINT = "https://hf-mirror.com"
HF_API_ENDPOINTS = (HF_ENDPOINT, "https://huggingface.co")
REVISIONS = {
    "smollm": "3ba9d605774198c5868892d7a8deda78031a781f",
    "finemath": "e92b25a616738fe95dc186b64dfb19f9c8525594",
    "stack_edu": "eeec5caac5cc3758a18f1d3ba4416837a9ba814c",
    "dclm": "817d6752765f6a41261085171dd546b104f60626",
}

CODE_LANGUAGE_WEIGHTS = {
    "Python": 40,
    "JavaScript": 20,
    "TypeScript": 10,
    "Cpp": 15,
    "Java": 15,
}
# Stack-Edu was curated at score >= 3 for every language except Java,
# whose validated release threshold is >= 2. Requiring >= 4 here exhausted
# the permissively licensed reservoir before the production target. Keep the
# upstream, language-specific quality gate rather than inventing a stricter
# threshold that the released corpus cannot sustain.
CODE_LANGUAGE_MIN_SCORES = {
    language: (2 if language == "Java" else 3) for language in CODE_LANGUAGE_WEIGHTS
}
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")


def _repo_parquet_files(repo: str, prefix: str, revision: str, seed: int) -> list[dict[str, Any]]:
    cache_root = Path(os.environ.get("HQ_METADATA_CACHE", "/home/hhai/pretrain/manifests/hub-filelists"))
    cache_path = cache_root / repo.replace("/", "--") / f"{revision}.json"
    metadata = None
    if cache_path.is_file():
        candidate = json.loads(cache_path.read_text())
        if candidate.get("sha") == revision:
            metadata = candidate
    if metadata is None:
        errors = []
        for attempt in range(3):
            for endpoint in HF_API_ENDPOINTS:
                api = f"{endpoint}/api/datasets/{repo}/revision/{revision}?blobs=true"
                try:
                    response = requests.get(api, timeout=(10, 30))
                    response.raise_for_status()
                    candidate = response.json()
                    if candidate.get("sha") != revision:
                        raise RuntimeError(
                            f"revision mismatch for {repo}: {candidate.get('sha')} != {revision}"
                        )
                    metadata = candidate
                    break
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    errors.append(f"{endpoint}: {type(error).__name__}: {error}")
            if metadata is not None:
                break
            time.sleep(2**attempt)
        if metadata is None:
            raise RuntimeError(f"unable to fetch pinned metadata for {repo}: {'; '.join(errors)}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("w") as stream:
            json.dump(metadata, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
    if metadata.get("sha") != revision:
        raise RuntimeError(f"revision mismatch for {repo}: {metadata.get('sha')} != {revision}")
    files = [
        item
        for item in metadata.get("siblings", [])
        if item.get("rfilename", "").startswith(prefix) and item.get("rfilename", "").endswith(".parquet")
    ]
    if not files:
        raise RuntimeError(f"no parquet files found for {repo}/{prefix} at {revision}")
    random.Random(seed).shuffle(files)
    return files


def _download_parquet(repo: str, revision: str, item: dict[str, Any]) -> Path:
    relative = item["rfilename"]
    expected_size = int(item.get("size") or item.get("lfs", {}).get("size") or 0)
    expected_sha = item.get("lfs", {}).get("sha256")
    cache_root = Path(os.environ.get("HQ_RAW_CACHE", "/home/hhai/pretrain/data/raw-cache"))
    path = cache_root / repo.replace("/", "--") / revision / relative
    marker = path.with_suffix(path.suffix + ".verified.json")
    if path.is_file() and marker.is_file() and (not expected_size or path.stat().st_size == expected_size):
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    url = f"{HF_ENDPOINT}/datasets/{repo}/resolve/{revision}/{relative}"
    command = [
        "aria2c",
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=16M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--max-tries=20",
        "--retry-wait=3",
        "--connect-timeout=30",
        "--timeout=60",
        "--lowest-speed-limit=1K",
        "--console-log-level=warn",
        "--summary-interval=30",
        f"--dir={part.parent}",
        f"--out={part.name}",
        url,
    ]
    # aria2 retries individual connections, but a long-lived Xet/CDN redirect
    # can still become unusable. Re-launching the resolve URL obtains a fresh
    # signed redirect and resumes the same verified .part file.
    last_error = None
    for attempt in range(6):
        try:
            subprocess.run(command, check=True)
            last_error = None
            break
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt == 5:
                break
            time.sleep(min(5 * 2**attempt, 60))
    if last_error is not None:
        raise last_error
    if expected_size and part.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {relative}: {part.stat().st_size} != {expected_size}")
    digest = None
    if expected_sha:
        hasher = hashlib.sha256()
        with part.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    if expected_sha and digest != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch for {relative}: {digest} != {expected_sha}")
    os.replace(part, path)
    marker.write_text(
        json.dumps({"repo": repo, "revision": revision, "path": relative, "size": path.stat().st_size, "sha256": digest})
        + "\n"
    )
    return path


def _source_state_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS completed_files ("
        "repo TEXT NOT NULL, revision TEXT NOT NULL, path TEXT NOT NULL, "
        "PRIMARY KEY(repo, revision, path))"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS file_progress ("
        "repo TEXT NOT NULL, revision TEXT NOT NULL, path TEXT NOT NULL, rows_consumed INTEGER NOT NULL, "
        "PRIMARY KEY(repo, revision, path))"
    )
    return connection


def _file_is_completed(state_db: Path | None, repo: str, revision: str, relative: str) -> bool:
    if state_db is None:
        return False
    connection = _source_state_connection(state_db)
    try:
        return connection.execute(
            "SELECT 1 FROM completed_files WHERE repo = ? AND revision = ? AND path = ?",
            (repo, revision, relative),
        ).fetchone() is not None
    finally:
        connection.close()


def _mark_file_completed(state_db: Path, repo: str, revision: str, relative: str) -> None:
    connection = _source_state_connection(state_db)
    try:
        connection.execute(
            "INSERT OR IGNORE INTO completed_files(repo, revision, path) VALUES (?, ?, ?)",
            (repo, revision, relative),
        )
        connection.execute(
            "DELETE FROM file_progress WHERE repo = ? AND revision = ? AND path = ?",
            (repo, revision, relative),
        )
        connection.commit()
    finally:
        connection.close()


def _file_progress(state_db: Path | None, repo: str, revision: str, relative: str) -> int:
    if state_db is None:
        return 0
    connection = _source_state_connection(state_db)
    try:
        row = connection.execute(
            "SELECT rows_consumed FROM file_progress WHERE repo = ? AND revision = ? AND path = ?",
            (repo, revision, relative),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        connection.close()


def _set_file_progress(
    state_db: Path, repo: str, revision: str, relative: str, rows_consumed: int
) -> None:
    connection = _source_state_connection(state_db)
    try:
        connection.execute(
            "INSERT INTO file_progress(repo, revision, path, rows_consumed) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(repo, revision, path) DO UPDATE SET rows_consumed = excluded.rows_consumed",
            (repo, revision, relative, rows_consumed),
        )
        connection.commit()
    finally:
        connection.close()


def _parquet_rows(
    repo: str,
    prefix: str,
    revision: str,
    seed: int,
    state_db: Path | None = None,
    delete_completed_raw: bool = False,
) -> Iterator[dict[str, Any]]:
    items = [
        item
        for item in _repo_parquet_files(repo, prefix, revision, seed)
        if not _file_is_completed(state_db, repo, revision, item["rfilename"])
    ]
    if not items:
        return

    # Download the next file while the current file is filtered and tokenized.
    # Keeping a single future bounds disk usage and preserves deterministic order.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_download_parquet, repo, revision, items[0])
        for item_index, item in enumerate(items):
            path = future.result()
            if item_index + 1 < len(items):
                future = executor.submit(_download_parquet, repo, revision, items[item_index + 1])
            relative = item["rfilename"]
            parquet = pq.ParquetFile(path, memory_map=True)
            resume_rows = _file_progress(state_db, repo, revision, relative)
            rows_consumed = 0
            exhausted = False
            try:
                for row_group in range(parquet.metadata.num_row_groups):
                    group_rows = parquet.metadata.row_group(row_group).num_rows
                    if rows_consumed + group_rows <= resume_rows:
                        rows_consumed += group_rows
                        continue
                    for batch in parquet.iter_batches(batch_size=1024, row_groups=[row_group]):
                        rows = batch.to_pylist()
                        batch_start = rows_consumed
                        for index, row in enumerate(rows):
                            absolute_row = batch_start + index
                            if absolute_row < resume_rows:
                                continue
                            yield row
                            rows_consumed = absolute_row + 1
                            if state_db is not None and rows_consumed % 1024 == 0:
                                _set_file_progress(state_db, repo, revision, relative, rows_consumed)
                        rows_consumed = batch_start + len(rows)
                exhausted = True
            finally:
                parquet.close()
            if exhausted and state_db is not None:
                _mark_file_completed(state_db, repo, revision, relative)
                if delete_completed_raw:
                    path.unlink(missing_ok=True)
                    path.with_suffix(path.suffix + ".verified.json").unlink(missing_ok=True)
                    path.with_suffix(path.suffix + ".aria2").unlink(missing_ok=True)


def _clean_text(text: Any, minimum_chars: int = 200) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.replace("\x00", "").strip()
    if len(text) < minimum_chars:
        return None
    if text.count("\ufffd") > max(2, len(text) // 10_000):
        return None
    compact = "".join(text.split())
    if not compact:
        return None
    sample = compact[:100_000]
    most_common_fraction = max(sample.count(c) for c in set(sample)) / len(sample)
    if most_common_fraction > 0.45:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 20 and len(set(lines)) / len(lines) < 0.2:
        return None
    return text


def _redact_email_and_ipv4(text: str) -> str:
    text = EMAIL_RE.sub("<EMAIL>", text)

    def replace_ip(match) -> str:
        candidate = match.group(0)
        try:
            ipaddress.IPv4Address(candidate)
        except ipaddress.AddressValueError:
            return candidate
        return "<IP_ADDRESS>"

    return IPV4_RE.sub(replace_ip, text)


def iter_web(
    seed: int = 42, state_db: Path | None = None, delete_completed_raw: bool = False
) -> Iterator[dict[str, str]]:
    rows = _parquet_rows(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup/",
        REVISIONS["smollm"],
        seed,
        state_db,
        delete_completed_raw,
    )
    for row in rows:
        metadata = row.get("metadata") or {}
        if metadata.get("language_score", 1.0) < 0.90:
            continue
        if metadata.get("int_score", 4) < 4:
            continue
        text = _clean_text(row.get("text"))
        if text:
            yield {"text": text, "id": str(row.get("id", "")), "source": "web"}


def iter_dclm(
    seed: int = 42, state_db: Path | None = None, delete_completed_raw: bool = False
) -> Iterator[dict[str, str]]:
    rows = _parquet_rows(
        "mlfoundations/dclm-baseline-1.0-parquet",
        "filtered/OH_eli5_vs_rw_v2_bigram_200k_train/"
        "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/",
        REVISIONS["dclm"],
        seed,
        state_db,
        delete_completed_raw,
    )
    for row in rows:
        if row.get("language", "en") != "en" or row.get("language_score", 1.0) < 0.90:
            continue
        text = _clean_text(row.get("text"))
        if text:
            yield {
                "text": _redact_email_and_ipv4(text),
                "id": str(row.get("id", row.get("url", ""))),
                "source": "dclm",
            }


def iter_synthetic(
    seed: int = 42, state_db: Path | None = None, delete_completed_raw: bool = False
) -> Iterator[dict[str, str]]:
    rows = _parquet_rows(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2/",
        REVISIONS["smollm"],
        seed,
        state_db,
        delete_completed_raw,
    )
    allowed_seeds = {"stanford", "openstax", "khanacademy"}
    allowed_formats = {"textbook_narrative", "textbook_academic", "scientific_article", "e-learning_module"}
    for row in rows:
        if row.get("seed_data") not in allowed_seeds or row.get("format") not in allowed_formats:
            continue
        text = _clean_text(row.get("text"))
        if text:
            yield {"text": text, "id": str(row.get("prompt", "")), "source": "synthetic"}


def iter_math(
    seed: int = 42, state_db: Path | None = None, delete_completed_raw: bool = False
) -> Iterator[dict[str, str]]:
    rows = _parquet_rows(
        "HuggingFaceTB/finemath",
        "finemath-4plus/",
        REVISIONS["finemath"],
        seed,
        state_db,
        delete_completed_raw,
    )
    for row in rows:
        if row.get("int_score", 4) < 4 or row.get("language_score", 1.0) < 0.90:
            continue
        text = _clean_text(row.get("text"))
        if text:
            yield {"text": text, "id": str(row.get("url", "")), "source": "math"}


def _weighted_round_robin(iterators: dict[str, Iterator], weights: dict[str, int]) -> Iterator[tuple[str, Any]]:
    schedule = list(itertools.chain.from_iterable(itertools.repeat(name, weight) for name, weight in weights.items()))
    cursor = 0
    while schedule:
        cursor %= len(schedule)
        name = schedule[cursor]
        try:
            yield name, next(iterators[name])
            cursor += 1
        except StopIteration:
            schedule = [entry for entry in schedule if entry != name]
            iterators.pop(name, None)


def _code_row_is_eligible(language: str, row: dict[str, Any]) -> bool:
    return (
        row.get("license_type") == "permissive"
        and row.get("int_score", 0) >= CODE_LANGUAGE_MIN_SCORES[language]
    )


def _rolling_unordered_map(
    executor: ThreadPoolExecutor,
    function,
    iterator: Iterator,
    max_in_flight: int,
) -> Iterator[Any]:
    """Map with a bounded rolling window and yield work as it finishes.

    ``Executor.map`` preserves input order, so one slow request can leave every
    other worker idle at the end of each finite batch.  This scheduler replaces
    each completed future before yielding it, keeps the window bounded, and
    avoids making source checkpoint look-ahead larger than the old four-wave
    batch.
    """
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")

    pending: dict[Future, int] = {}
    next_sequence = 0
    exhausted = False

    def fill_window() -> None:
        nonlocal exhausted, next_sequence
        while not exhausted and len(pending) < max_in_flight:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[executor.submit(function, item)] = next_sequence
            next_sequence += 1

    fill_window()
    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        ordered = sorted(completed, key=pending.__getitem__)
        for future in ordered:
            pending.pop(future)
            # Replace exactly this future before yielding. Other completed
            # futures still count against the window until they are emitted,
            # so source look-ahead never grows beyond max_in_flight.
            fill_window()
            yield future.result()


def iter_code(
    seed: int = 42, state_db: Path | None = None, delete_completed_raw: bool = False
) -> Iterator[dict[str, str]]:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    # A same-object SWH benchmark on the target L20 host found 48 workers
    # matched 64-worker throughput while materially outperforming 32 workers.
    workers = int(os.environ.get("HQ_CODE_DOWNLOAD_WORKERS", "48"))
    if not 1 <= workers <= 64:
        raise ValueError(f"HQ_CODE_DOWNLOAD_WORKERS must be in [1, 64], got {workers}")
    s3 = boto3.client(
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
    streams = {
        language: iter(
            _parquet_rows(
                "HuggingFaceTB/stack-edu",
                f"{language}/",
                REVISIONS["stack_edu"],
                seed + index,
                state_db,
                delete_completed_raw,
            )
        )
        for index, language in enumerate(CODE_LANGUAGE_WEIGHTS)
    }

    def candidates():
        for language, row in _weighted_round_robin(streams, CODE_LANGUAGE_WEIGHTS):
            if _code_row_is_eligible(language, row):
                yield language, row

    def download(candidate: tuple[str, dict[str, Any]]) -> dict[str, str] | None:
        language, row = candidate
        blob_id = row.get("blob_id")
        if not blob_id:
            return None
        try:
            obj = s3.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
            with gzip.GzipFile(fileobj=obj["Body"]) as stream:
                raw = stream.read()
        except (BotoCoreError, ClientError):
            return None
        encoding = row.get("src_encoding") or "utf-8"
        try:
            decoded = raw.decode(encoding, errors="ignore")
        except LookupError:
            decoded = raw.decode("utf-8", errors="ignore")
        text = _clean_text(decoded, minimum_chars=80)
        if text:
            identifier = f"{row.get('repo_name', '')}:{row.get('path', '')}:{blob_id}"
            return {"text": text, "id": identifier, "source": f"code:{language}"}
        return None

    iterator = iter(candidates())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Keep the same bounded four-wave look-ahead as the previous batched
        # implementation, but refill it continuously and do not let one slow
        # object impose a barrier on all of the fast objects behind it.
        for result in _rolling_unordered_map(executor, download, iterator, workers * 4):
            if result:
                yield result


def iter_source(
    source: str,
    seed: int = 42,
    state_db: Path | None = None,
    delete_completed_raw: bool = False,
) -> Iterator[dict[str, str]]:
    functions = {
        "web": iter_web,
        "dclm": iter_dclm,
        "synthetic": iter_synthetic,
        "math": iter_math,
        "code": iter_code,
    }
    try:
        return functions[source](seed, state_db=state_db, delete_completed_raw=delete_completed_raw)
    except KeyError as exc:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(functions)}") from exc
