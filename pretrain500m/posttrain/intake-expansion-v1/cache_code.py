"""Cache the five pinned Stack-Edu Parquet indexes in verified tmpfs files."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SOURCE = Path(__file__).resolve().parent
CACHE = Path("/dev/shm/p529m-intake-expansion-v1")
MIN_FREE = 8 * 1024**3


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def download(segment: dict) -> dict:
    final = CACHE / (segment["segment_id"] + ".parquet")
    part = Path(str(final) + ".part")
    receipt_path = CACHE / (segment["segment_id"] + ".receipt.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "VERIFIED_CACHE_READY" and final.stat().st_size == segment["bytes"] and sha256(final) == segment["lfs_sha256"]:
            return receipt
        raise RuntimeError("existing cache receipt requires inspection: " + segment["segment_id"])
    if final.exists() or part.exists():
        raise RuntimeError("unreceipted cache file exists: " + segment["segment_id"])
    if shutil.disk_usage(CACHE).free < MIN_FREE + segment["bytes"]:
        raise RuntimeError("tmpfs headroom floor")
    retry = Retry(total=5, connect=4, read=3, status=4, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))
    client = requests.Session()
    client.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1))
    url = f'https://hf-mirror.com/datasets/{segment["repo_id"]}/resolve/{segment["revision"]}/{segment["path"]}?download=true'
    size = 0
    h = hashlib.sha256()
    with client.get(url, stream=True, timeout=(5, 60), headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        with part.open("xb") as handle:
            for block in response.iter_content(8 * 1024**2):
                if not block:
                    continue
                size += len(block)
                if size > segment["bytes"]:
                    raise ValueError("download exceeded pinned size")
                h.update(block)
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    if size != segment["bytes"] or h.hexdigest() != segment["lfs_sha256"]:
        raise ValueError("full-file size or LFS SHA-256 mismatch")
    os.replace(part, final)
    receipt = {
        "schema_version": 1,
        "status": "VERIFIED_CACHE_READY",
        "completed_utc": utc(),
        "segment_id": segment["segment_id"],
        "repo_id": segment["repo_id"],
        "revision": segment["revision"],
        "path": segment["path"],
        "bytes": size,
        "sha256": h.hexdigest(),
        "cache_path": str(final),
        "training_admitted": False,
    }
    temporary = Path(str(receipt_path) + ".next")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    print(json.dumps({"segment_id": segment["segment_id"], "status": receipt["status"], "bytes": size}), flush=True)
    return receipt


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    plan = json.loads((SOURCE / "plan.json").read_text())
    segments = [x for x in plan["segments"] if x["logical_source_id"].startswith("code_")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        receipts = list(pool.map(download, segments))
    summary = {
        "schema_version": 1,
        "status": "VERIFIED_CODE_CACHE_COMPLETE",
        "completed_utc": utc(),
        "files": len(receipts),
        "bytes": sum(x["bytes"] for x in receipts),
        "receipts": {x["segment_id"]: sha256(CACHE / (x["segment_id"] + ".receipt.json")) for x in receipts},
        "training_admitted": False,
    }
    path = CACHE / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
