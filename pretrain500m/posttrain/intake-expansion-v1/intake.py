"""Revision-pinned, bounded raw intake for the frozen F2/F3 expansion plan.

Outputs are raw and explicitly NOT_ADMITTED. Completed segments are idempotent;
failed attempts are archived before a retry so evidence is retained.
"""
from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import threading
import warnings

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import http_ranges


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
SOURCE = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/intake-expansion-v1"
MIN_FREE = 18 * 1024**3
OUTER_WORKERS = 10
CODE_WORKERS = 64
CACHE = Path("/dev/shm/p529m-intake-expansion-v1")
http_ranges.CAP = 2 * 1024**3
LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "CC0-1.0", "Unlicense"}
_thread_local = threading.local()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def session() -> requests.Session:
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        retry = Retry(
            total=4,
            connect=3,
            read=2,
            status=3,
            backoff_factor=0.25,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        value.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2))
        _thread_local.session = value
    return value


def code_text(row: dict) -> tuple[dict | None, str, int]:
    licenses = set(row.get("detected_licenses") or [])
    if row.get("license_type") != "permissive" or not licenses or not licenses <= LICENSES or row.get("int_score", 0) < 3:
        return None, "LICENSE_OR_SCORE_EXCLUDED", 0
    blob = row.get("blob_id", "")
    if len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob):
        return None, "INVALID_BLOB_ID", 0
    downloaded = 0
    try:
        url = "https://softwareheritage.s3.amazonaws.com/content/" + blob
        with session().get(url, stream=True, timeout=(5, 30)) as response:
            response.raise_for_status()
            compressed = response.raw.read(1024**2 + 1)
        downloaded = len(compressed)
        if downloaded > 1024**2:
            return None, "SIZE_EXCLUDED", downloaded
        content = gzip.GzipFile(fileobj=io.BytesIO(compressed)).read(1024**2 + 1)
        if len(content) > 1024**2:
            return None, "SIZE_EXCLUDED", downloaded
        if hashlib.sha1(content).hexdigest() != blob:
            return None, "CONTENT_HASH_MISMATCH", downloaded
        text = content.decode("utf-8", errors="strict")
        if row.get("language") == "Python":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse(text)
            except (SyntaxError, ValueError):
                return None, "PYTHON_SYNTAX_EXCLUDED", downloaded
        value = dict(row, text=text, content_sha256=hashlib.sha256(content).hexdigest())
        return value, "CONTENT_VERIFIED", downloaded
    except Exception as exc:  # receipt records class only; never URLs or payloads
        return None, type(exc).__name__, downloaded


def verify_history(segment: dict) -> set[int]:
    groups: set[int] = set()
    for ref in segment["history"]:
        path = Path(ref["path"])
        if not path.is_file() or sha256(path) != ref["sha256"]:
            raise ValueError("history receipt changed: " + path.name)
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "RAW_SEGMENT_READY":
            raise ValueError("history receipt is not complete: " + path.name)
        for key in ("repo_id", "revision", "path"):
            if receipt.get(key) != segment[key]:
                raise ValueError("history source identity mismatch: " + path.name)
        output = path.with_name(path.name.replace(".receipt.json", ".jsonl.gz"))
        if not output.is_file() or sha256(output) != receipt.get("output_sha256"):
            raise ValueError("history output changed: " + output.name)
        groups.update(int(group["row_group"]) for group in receipt.get("groups", []))
    if groups != set(segment["exclude_row_groups"]):
        raise ValueError("history row-group union mismatch")
    return groups


def archive_incomplete(receipt_path: Path, part: Path) -> None:
    if not receipt_path.exists() and not part.exists():
        return
    archive = OUTPUT / "attempts"
    archive.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if receipt_path.exists():
        os.replace(receipt_path, archive / f"{receipt_path.name}.{stamp}")
    if part.exists():
        os.replace(part, archive / f"{part.name}.{stamp}")


def acquire(segment: dict) -> dict:
    segment_id = segment["segment_id"]
    logical_id = segment["logical_source_id"]
    final = OUTPUT / f"{segment_id}.jsonl.gz"
    part = Path(str(final) + ".part")
    receipt_path = OUTPUT / f"{segment_id}.receipt.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get("status") == "RAW_SEGMENT_READY" and final.is_file() and sha256(final) == old.get("output_sha256"):
            return old
        if final.exists():
            raise RuntimeError("final output exists without a valid ready receipt: " + segment_id)
        archive_incomplete(receipt_path, part)
    elif final.exists():
        raise RuntimeError("final output exists without a receipt: " + segment_id)
    elif part.exists():
        archive_incomplete(receipt_path, part)

    receipt = dict(
        segment,
        status="READING",
        started_utc=utc(),
        training_admitted=False,
        full_shard_sha256_verified=False,
        verification="revision-pinned HTTPS, exact size, Parquet metadata, range hashes, and prior output hashes; LFS digest is expected metadata",
        rows_written=0,
        uncompressed_output_bytes=0,
        parquet_network_bytes=0,
        code_payload_bytes=0,
        code_rehydration_counts={},
        groups=[],
    )
    atomic_json(receipt_path, receipt)
    remote = None
    rows_written = bytes_written = code_attempts = code_payload_bytes = 0
    try:
        excluded = verify_history(segment)
        cache = CACHE / f"{segment_id}.parquet"
        cache_receipt_path = CACHE / f"{segment_id}.receipt.json"
        if logical_id.startswith("code_") and cache.is_file() and cache_receipt_path.is_file():
            cache_receipt = json.loads(cache_receipt_path.read_text())
            if cache_receipt.get("status") != "VERIFIED_CACHE_READY":
                raise ValueError("code cache is not complete")
            if any(cache_receipt.get(key) != segment[key] for key in ("repo_id", "revision", "path", "bytes")):
                raise ValueError("code cache identity mismatch")
            cache_digest = sha256(cache)
            if cache.stat().st_size != segment["bytes"] or cache_digest != segment["lfs_sha256"] or cache_receipt.get("sha256") != cache_digest:
                raise ValueError("code cache full-file identity mismatch")
            receipt.update(full_shard_sha256_verified=True, cache_path=str(cache), cache_sha256=cache_digest,
                           cache_receipt_path=str(cache_receipt_path), cache_receipt_sha256=sha256(cache_receipt_path))
            parquet = pq.ParquetFile(cache)
        else:
            url = f'https://hf-mirror.com/datasets/{segment["repo_id"]}/resolve/{segment["revision"]}/{segment["path"]}?download=true'
            remote = http_ranges.BoundedHTTPFile(url)
            if remote.total != segment["bytes"]:
                raise ValueError("pinned file size mismatch")
            parquet = pq.ParquetFile(remote)
        if parquet.num_row_groups != segment["parquet_row_groups"]:
            raise ValueError("Parquet row-group count changed")
        if segment["additional_row_groups"] > parquet.num_row_groups - len(excluded):
            raise ValueError("insufficient unused row groups")
        columns = [name for name in parquet.schema_arrow.names if name in {
            "text", "content", "url", "id", "language", "language_score", "score", "int_score",
            "blob_id", "repo_name", "path", "detected_licenses", "license_type", "prompt",
            "seed_data", "parent_document_id", "seed_document_url", "metadata", "audience",
            "format", "dump", "date",
        }]
        receipt["schema"] = parquet.schema_arrow.names
        available = [i for i in range(parquet.num_row_groups) if i not in excluded]
        groups = sorted(available, key=lambda i: hashlib.sha256(f"{segment_id}|{i}|p529m-expansion-v1".encode()).hexdigest())
        with contextlib.ExitStack() as stack:
            writer = stack.enter_context(gzip.open(part, "wt", encoding="utf-8", compresslevel=3))
            code_pool = stack.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=CODE_WORKERS)) if logical_id.startswith("code_") else None
            for group in groups[: segment["additional_row_groups"]]:
                if shutil.disk_usage(OUTPUT).free < MIN_FREE:
                    receipt["stop_reason"] = "DISK_HEADROOM"
                    break
                metadata = parquet.metadata.row_group(group)
                compressed = sum(
                    metadata.column(i).total_compressed_size
                    for i in range(metadata.num_columns)
                    if metadata.column(i).path_in_schema.split(".")[0] in columns
                )
                if remote is not None and remote.transferred + compressed + 65536 > segment["max_parquet_network_bytes"]:
                    receipt["stop_reason"] = "PARQUET_NETWORK_CAP"
                    break
                table = parquet.read_row_group(group, columns=columns, use_threads=False)
                row_offset = sum(parquet.metadata.row_group(i).num_rows for i in range(group))
                decorated = []
                for i, item in enumerate(table.to_pylist()):
                    row = dict(item)
                    row["_provenance"] = {
                        "source_id": logical_id,
                        "segment_id": segment_id,
                        "repo_id": segment["repo_id"],
                        "revision": segment["revision"],
                        "shard": segment["path"],
                        "row_group": group,
                        "row_in_group": i,
                        "physical_row": row_offset + i,
                    }
                    decorated.append(row)
                if logical_id.startswith("code_"):
                    eligible = [
                        row for row in decorated
                        if row.get("license_type") == "permissive"
                        and set(row.get("detected_licenses") or [])
                        and set(row["detected_licenses"]) <= LICENSES
                        and row.get("int_score", 0) >= 3
                    ]
                    remaining = max(0, segment["max_code_attempts"] - code_attempts)
                    eligible = eligible[:remaining]
                    code_attempts += len(eligible)
                    outputs = list(code_pool.map(code_text, eligible))
                    accepted = []
                    for row, status, downloaded in outputs:
                        code_payload_bytes += downloaded
                        receipt["code_rehydration_counts"][status] = receipt["code_rehydration_counts"].get(status, 0) + 1
                        if row is not None:
                            accepted.append(row)
                    if code_payload_bytes > segment["max_code_payload_bytes"]:
                        receipt["stop_reason"] = "CODE_PAYLOAD_CAP"
                else:
                    accepted = decorated
                receipt["groups"].append({"row_group": group, "physical_row_offset": row_offset, "rows_read": len(decorated)})
                for row in accepted:
                    text = row.get("text") or row.get("content")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    payload = json.dumps(row, ensure_ascii=False, default=str) + "\n"
                    size = len(payload.encode())
                    if rows_written >= segment["max_rows"] or bytes_written + size > segment["max_text_bytes"]:
                        receipt["stop_reason"] = "OUTPUT_CAP"
                        break
                    writer.write(payload)
                    rows_written += 1
                    bytes_written += size
                writer.flush()
                receipt.update(
                    rows_written=rows_written,
                    uncompressed_output_bytes=bytes_written,
                    parquet_network_bytes=remote.transferred if remote is not None else 0,
                    code_payload_bytes=code_payload_bytes,
                    code_attempts=code_attempts,
                    updated_utc=utc(),
                )
                atomic_json(receipt_path, receipt)
                print(json.dumps({"segment": segment_id, "groups": len(receipt["groups"]), "rows": rows_written,
                                  "code_attempts": code_attempts, "code_payload_bytes": code_payload_bytes}), flush=True)
                if receipt.get("stop_reason") or (
                    logical_id.startswith("code_") and code_attempts >= segment["max_code_attempts"]
                ):
                    break
        if rows_written == 0:
            raise ValueError("no complete usable text rows acquired within bounds")
        if len(receipt["groups"]) != segment["additional_row_groups"]:
            raise ValueError("planned row-group target was not completed")
        os.replace(part, final)
        receipt.update(
            status="RAW_SEGMENT_READY",
            rows_written=rows_written,
            uncompressed_output_bytes=bytes_written,
            output_sha256=sha256(final),
            output_bytes=final.stat().st_size,
            completed_utc=utc(),
        )
    except Exception as exc:
        receipt.update(status="BLOCKED", error_type=type(exc).__name__, error=str(exc).split("?")[0][:240], updated_utc=utc())
    finally:
        if remote is not None:
            receipt.update(parquet_network_bytes=remote.transferred, range_receipts=remote.ranges)
        atomic_json(receipt_path, receipt)
    print(json.dumps({key: receipt.get(key) for key in ("segment_id", "logical_source_id", "status", "rows_written", "output_bytes", "error_type", "stop_reason")}), flush=True)
    return receipt


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    lock_path = OUTPUT / "writer.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another intake writer holds the lock") from exc
        lock.write(str(os.getpid()))
        lock.flush()
        plan_path = SOURCE / "plan.json"
        plan = json.loads(plan_path.read_text())
        if plan.get("status") != "FROZEN_UNIQUE_DATA_EXPANSION_PLAN_NOT_ADMITTED":
            raise ValueError("unexpected plan status")
        segments = sorted(plan["segments"], key=lambda x: (not x["logical_source_id"].startswith("code_"), -x["additional_row_groups"]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=OUTER_WORKERS) as pool:
            receipts = list(pool.map(acquire, segments))
        summary = {
            "schema_version": 1,
            "status": "RAW_INTAKE_FINISHED_NOT_ADMITTED",
            "checked_utc": utc(),
            "plan_sha256": sha256(plan_path),
            "sources_ready": sum(x["status"] == "RAW_SEGMENT_READY" for x in receipts),
            "sources_blocked": sum(x["status"] != "RAW_SEGMENT_READY" for x in receipts),
            "groups": sum(len(x.get("groups", [])) for x in receipts),
            "rows": sum(x.get("rows_written", 0) for x in receipts),
            "training_admitted": False,
        }
        atomic_json(OUTPUT / "intake-summary.json", summary)
        if summary["sources_blocked"]:
            raise RuntimeError("one or more intake segments are blocked")


if __name__ == "__main__":
    main()
