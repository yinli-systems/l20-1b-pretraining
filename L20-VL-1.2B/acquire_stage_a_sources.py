#!/usr/bin/env python3
"""Acquire only the frozen Stage-A source artifacts and safely unpack CLEVR."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
ADMISSION = ROOT / "stage_a_data_admission.json"
RECEIPT = ROOT / "evidence" / "stage-a-source-acquisition.json"
MAX_EXTRACTED_BYTES = 30_000_000_000
MAX_ARCHIVE_MEMBERS = 250_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(receipt: dict) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, RECEIPT)


def download(url: str, destination: Path, expected_bytes: int, receipt: dict) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise RuntimeError(f"existing artifact has wrong size: {destination}")
        return {
            "path": str(destination),
            "bytes": expected_bytes,
            "sha256": sha256_file(destination),
            "download_status": "reused_complete",
        }
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise RuntimeError(f"partial artifact exceeds expected size: {partial}")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    print(f"DOWNLOAD_START path={destination.name} resume_bytes={offset}", flush=True)
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(30, 300),
        allow_redirects=True,
    ) as response:
        if offset and response.status_code != 206:
            offset = 0
            partial.unlink(missing_ok=True)
            return download(url, destination, expected_bytes, receipt)
        response.raise_for_status()
        mode = "ab" if offset else "wb"
        downloaded = offset
        next_report = ((downloaded // (512 * 1024 * 1024)) + 1) * 512 * 1024 * 1024
        with partial.open(mode) as handle:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not block:
                    continue
                downloaded += len(block)
                if downloaded > expected_bytes:
                    raise RuntimeError(f"server exceeded frozen byte count for {url}")
                handle.write(block)
                if downloaded >= next_report:
                    receipt["active"] = {
                        "path": str(destination),
                        "downloaded_bytes": downloaded,
                        "expected_bytes": expected_bytes,
                    }
                    save(receipt)
                    print(
                        f"DOWNLOAD_PROGRESS path={destination.name} bytes={downloaded}/{expected_bytes}",
                        flush=True,
                    )
                    next_report += 512 * 1024 * 1024
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"frozen size mismatch for {url}: {partial.stat().st_size} != {expected_bytes}"
        )
    os.replace(partial, destination)
    result = {
        "path": str(destination),
        "bytes": expected_bytes,
        "sha256": sha256_file(destination),
        "download_status": "complete",
    }
    print(f"DOWNLOAD_DONE path={destination.name} sha256={result['sha256']}", flush=True)
    return result


def safe_extract_clevr(archive: Path, destination: Path) -> dict:
    marker = destination / ".extraction-receipt.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior.get("archive_sha256") != sha256_file(archive):
            raise RuntimeError("CLEVR archive changed after extraction")
        return prior
    if destination.exists():
        raise RuntimeError("unreceipted CLEVR extraction destination already exists")
    temporary = destination.with_name(destination.name + ".extracting")
    if temporary.exists():
        raise RuntimeError("incomplete extraction exists; inspect it before retrying")
    temporary.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise RuntimeError("CLEVR archive member cap exceeded")
        extracted_bytes = sum(member.file_size for member in members)
        if extracted_bytes > MAX_EXTRACTED_BYTES:
            raise RuntimeError("CLEVR extracted-byte cap exceeded")
        root = temporary.resolve()
        for member in members:
            target = (temporary / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"archive path traversal: {member.filename}")
        handle.extractall(temporary)
    nested = temporary / "CLEVR_v1.0"
    for required in (
        nested / "LICENSE.txt",
        nested / "images" / "train",
        nested / "images" / "val",
        nested / "questions" / "CLEVR_train_questions.json",
        nested / "questions" / "CLEVR_val_questions.json",
        nested / "scenes" / "CLEVR_train_scenes.json",
        nested / "scenes" / "CLEVR_val_scenes.json",
    ):
        if not required.exists():
            raise RuntimeError(f"CLEVR extraction missing required artifact: {required}")
    os.replace(nested, destination)
    temporary.rmdir()
    record = {
        "archive_sha256": sha256_file(archive),
        "archive_members": len(members),
        "declared_extracted_bytes": extracted_bytes,
        "train_images": len(list((destination / "images" / "train").glob("*.png"))),
        "validation_images": len(list((destination / "images" / "val").glob("*.png"))),
        "test_directory_present_but_prohibited": (destination / "images" / "test").exists(),
    }
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    if not admission.get("download_authorized") or admission.get("automatic_training_start"):
        raise SystemExit("Stage-A admission is not safe for acquisition")
    destination = Path(admission["limits"]["destination"])
    free_before = shutil.disk_usage(destination.parent if destination.parent.exists() else ROOT).free
    if free_before < admission["limits"]["minimum_free_bytes_before_download"]:
        raise SystemExit(f"insufficient free disk before acquisition: {free_before}")
    destination.mkdir(parents=True, exist_ok=True)
    source_root = destination / "sources"
    receipt = {
        "schema_version": "2026-09-13-v1",
        "started_at": utc_now(),
        "status": "running",
        "admission_sha256": sha256_file(ADMISSION),
        "free_bytes_before": free_before,
        "download_byte_cap": admission["limits"]["download_bytes_max"],
        "formal_training": False,
        "training_prediction_tokens": 0,
        "artifacts": {},
    }
    save(receipt)
    try:
        clevr = admission["sources"]["clevr_v1_0"]
        clevr_zip = source_root / "CLEVR_v1.0.zip"
        receipt["artifacts"]["clevr_archive"] = download(
            clevr["artifact_url"], clevr_zip, clevr["expected_bytes"], receipt
        )
        save(receipt)
        receipt["clevr_extraction"] = safe_extract_clevr(
            clevr_zip, source_root / "CLEVR_v1.0"
        )
        save(receipt)
        narratives = admission["sources"]["open_images_localized_narratives"]
        receipt["artifacts"]["localized_narratives_captions"] = download(
            narratives["captions_url"],
            source_root / "open_images_train_v6_captions.jsonl",
            narratives["captions_expected_bytes"],
            receipt,
        )
        save(receipt)
        receipt["artifacts"]["open_images_metadata"] = download(
            narratives["image_metadata_url"],
            source_root / "train-images-boxable-with-rotation.csv",
            narratives["image_metadata_expected_bytes"],
            receipt,
        )
        receipt["status"] = "complete_pending_content_audit"
        receipt["completed_at"] = utc_now()
        receipt["free_bytes_after"] = shutil.disk_usage(destination).free
        receipt["active"] = None
        save(receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        receipt["status"] = "failed"
        receipt["failed_at"] = utc_now()
        receipt["error"] = f"{type(error).__name__}: {error}"
        save(receipt)
        raise


if __name__ == "__main__":
    main()
