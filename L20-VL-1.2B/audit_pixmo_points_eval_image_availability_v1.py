#!/usr/bin/env python3
"""Preflight public PixMo-Points-Eval image availability without persistence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image
import pyarrow.parquet as pq
import requests


STATUS = "authorized_pixmo_points_eval_image_availability_preflight_only_v1"
USER_AGENT = "L20-VL-1.2B-benchmark-integrity-preflight/1.0"


def deterministic_sample(parquet: Path, sample_size: int) -> list[dict[str, str]]:
    rows = pq.read_table(parquet, columns=["image_url", "image_sha256"]).to_pylist()
    by_hash: dict[str, str] = {}
    for row in rows:
        image_hash = str(row["image_sha256"])
        image_url = str(row["image_url"])
        previous = by_hash.setdefault(image_hash, image_url)
        if previous != image_url:
            raise RuntimeError(f"one image hash maps to multiple URLs: {image_hash}")
    if sample_size < 1 or sample_size > len(by_hash):
        raise ValueError("sample_size must be in [1, unique_images]")
    # SHA-256 digests are uniformly distributed, so their fixed lexical order is
    # a deterministic model-blind sample that does not depend on labels or points.
    return [
        {"image_sha256": image_hash, "image_url": by_hash[image_hash]}
        for image_hash in sorted(by_hash)[:sample_size]
    ]


def verify_image_bytes(payload: bytes, expected_sha256: str, maximum_bytes: int) -> dict[str, Any]:
    if not payload:
        return {"status": "empty_payload", "bytes": 0}
    if len(payload) > maximum_bytes:
        return {"status": "size_limit_exceeded", "bytes": len(payload)}
    observed = sha256(payload).hexdigest()
    if observed != expected_sha256:
        return {
            "status": "sha256_mismatch",
            "bytes": len(payload),
            "observed_sha256": observed,
        }
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            image_format = image.format
    except Exception as error:  # Pillow exposes multiple decode exception types.
        return {
            "status": "decode_failure",
            "bytes": len(payload),
            "error_type": type(error).__name__,
        }
    if width < 1 or height < 1:
        return {"status": "invalid_dimensions", "bytes": len(payload)}
    return {
        "status": "verified",
        "bytes": len(payload),
        "width": width,
        "height": height,
        "format": image_format,
    }


def fetch_one(
    item: dict[str, str],
    connect_timeout_seconds: int,
    read_timeout_seconds: int,
    maximum_bytes: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_sha256": item["image_sha256"],
        "image_url": item["image_url"],
    }
    try:
        with requests.get(
            item["image_url"],
            headers={"User-Agent": USER_AGENT},
            stream=True,
            timeout=(connect_timeout_seconds, read_timeout_seconds),
        ) as response:
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > maximum_bytes:
                return {**result, "status": "declared_size_limit_exceeded", "declared_bytes": int(declared)}
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                received += len(chunk)
                if received > maximum_bytes:
                    return {**result, "status": "stream_size_limit_exceeded", "bytes": received}
                chunks.append(chunk)
        return {**result, **verify_image_bytes(b"".join(chunks), item["image_sha256"], maximum_bytes)}
    except requests.RequestException as error:
        return {
            **result,
            "status": "request_failure",
            "error_type": type(error).__name__,
        }
    except (TypeError, ValueError) as error:
        return {
            **result,
            "status": "response_metadata_failure",
            "error_type": type(error).__name__,
        }


def run_preflight(protocol: dict[str, Any]) -> dict[str, Any]:
    sample = deterministic_sample(
        Path(protocol["prerequisites"]["evaluation_parquet"]["path"]),
        int(protocol["sample"]["unique_images"]),
    )
    settings = protocol["network"]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(settings["workers"])) as pool:
        futures = {
            pool.submit(
                fetch_one,
                item,
                int(settings["connect_timeout_seconds"]),
                int(settings["read_timeout_seconds"]),
                int(settings["maximum_image_bytes"]),
            ): item
            for item in sample
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["image_sha256"])
    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    verified = counts.get("verified", 0)
    hash_mismatches = counts.get("sha256_mismatch", 0)
    gate = protocol["gate"]
    passed = (
        verified >= int(gate["minimum_verified_images"])
        and hash_mismatches <= int(gate["maximum_sha256_mismatches"])
    )
    return {
        "sample": sample,
        "results": results,
        "status_counts": dict(sorted(counts.items())),
        "verified_images": verified,
        "verified_fraction": verified / len(sample),
        "sha256_mismatches": hash_mismatches,
        "gate_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()

    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("PixMo points image-availability preflight is not authorized")
    for field in ("image_persistence_authorized", "training_authorized", "model_evaluation_authorized"):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    root = Path(__file__).resolve().parent
    for path, key in (
        (Path(__file__), "auditor_sha256"),
        (root / "test_audit_pixmo_points_eval_image_availability_v1.py", "test_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise RuntimeError(f"source hash mismatch: {path.name}")
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"prerequisite hash mismatch: {name}")
    receipt_path = Path(protocol["output_receipt"])
    if receipt_path.exists():
        raise FileExistsError(receipt_path)

    preflight = run_preflight(protocol)
    decision = (
        "pass_direct_urls_sufficient_for_full_acquisition_attempt"
        if preflight["gate_passed"]
        else "reject_direct_urls_as_incomplete_or_integrity_unsafe"
    )
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_pixmo_points_eval_image_availability_preflight",
        "completed_at": utc_now(),
        "decision": decision,
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "auditor_sha256": sha256_file(Path(__file__)),
        "network_settings": protocol["network"],
        "gate": protocol["gate"],
        **preflight,
        "downloaded_images_persisted": 0,
        "model_under_test_accessed": False,
        "training_started": False,
        "model_evaluation_started": False,
        "next_allowed_step": (
            "Freeze a complete 431-image acquisition protocol before any benchmark scoring."
            if preflight["gate_passed"]
            else "Do not score an available-only subset; seek an integrity-verifiable archival source or use a different benchmark."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "decision": decision,
        "verified_images": preflight["verified_images"],
        "sample_images": len(preflight["sample"]),
        "status_counts": preflight["status_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
