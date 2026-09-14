#!/usr/bin/env python3
"""Select and acquire the exact bounded PixMo-Cap image-audit sample."""
from __future__ import annotations

import hashlib
import heapq
import io
import json
import os
import re
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests
from PIL import Image, ImageDraw, ImageFont

from audit_pixmo_metadata import (
    EMAIL_RE,
    PHONE_RE,
    REFUSAL_MARKERS,
    ascii_fraction,
    benchmark_markers,
    normalized_words,
)


ROOT = Path(__file__).resolve().parent
METADATA_ADMISSION = ROOT / "pixmo_cap_metadata_admission.json"
IMAGE_ADMISSION = ROOT / "pixmo_image_audit_admission.json"
METADATA_AUDIT = ROOT / "evidence" / "pixmo-cap-metadata-audit.json"
HOSTED_PROFILE = ROOT / "evidence" / "pixmo-cap-hosted-profile.json"
RECEIPT = ROOT / "evidence" / "pixmo-cap-image-audit-acquisition.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return value


def conservative_candidate(url: str, caption: str, transcripts: list[str]) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        return False, "non_https"
    if benchmark_markers(url):
        return False, "benchmark_url_marker"
    words = normalized_words(caption)
    transcript_text = " ".join(item for item in transcripts if isinstance(item, str))
    if not 60 <= len(words) <= 350:
        return False, "caption_length"
    if not normalized_words(transcript_text):
        return False, "empty_transcript"
    if ascii_fraction(caption) < 0.985:
        return False, "caption_non_ascii"
    combined = caption + "\n" + transcript_text
    if EMAIL_RE.search(combined):
        return False, "email_regex"
    if PHONE_RE.search(combined):
        return False, "phone_regex"
    if any(marker in combined.lower() for marker in REFUSAL_MARKERS):
        return False, "refusal_marker"
    return True, "pass"


def select_rows(metadata: dict, admission: dict) -> list[dict]:
    host = admission["host_allowlist"][0]
    quotas = admission["path_families"]
    heaps: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    root = Path(metadata["destination"])
    row_id = 0
    for relative in metadata["files"]:
        parquet = pq.ParquetFile(root / relative)
        for batch in parquet.iter_batches(
            batch_size=8192, columns=["image_url", "caption", "transcripts"]
        ):
            columns = batch.to_pydict()
            for url, caption, transcripts in zip(
                columns["image_url"], columns["caption"], columns["transcripts"], strict=True
            ):
                current_id = row_id
                row_id += 1
                parsed = urlparse(url)
                if (parsed.hostname or "").lower() != host:
                    continue
                parts = [part for part in parsed.path.split("/") if part]
                family = parts[0].lower() if parts else ""
                if family not in quotas:
                    continue
                accepted, _ = conservative_candidate(url, caption, transcripts)
                if not accepted:
                    continue
                score_hex = hashlib.sha256(url.encode("utf-8")).hexdigest()
                score = int(score_hex, 16)
                record = {
                    "row_id": current_id,
                    "family": family,
                    "url": url,
                    "caption": caption,
                    "transcripts": transcripts,
                    "selection_sha256": score_hex,
                }
                heap = heaps[family]
                item = (-score, url, record)
                if len(heap) < quotas[family]:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)

    selected = []
    for family, quota in quotas.items():
        records = [item[2] for item in heaps[family]]
        records.sort(key=lambda item: item["selection_sha256"])
        if len(records) != quota:
            raise RuntimeError(f"insufficient candidates for {family}: {len(records)} != {quota}")
        selected.extend(records)
    return selected


def fetch_one(record: dict, admission: dict, image_root: Path) -> dict:
    parsed = urlparse(record["url"])
    family = record["family"]
    if (parsed.hostname or "").lower() not in admission["host_allowlist"]:
        raise RuntimeError("host escaped allowlist")
    if not parsed.path.startswith(f"/{family}/"):
        raise RuntimeError("path escaped selected family")
    name = record["selection_sha256"] + ".img"
    destination = image_root / family / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    total = 0
    digest = hashlib.sha256()
    content_type = ""
    with requests.get(record["url"], stream=True, timeout=(20, 90), allow_redirects=True) as response:
        response.raise_for_status()
        final_host = (urlparse(response.url).hostname or "").lower()
        if final_host not in admission["host_allowlist"]:
            raise RuntimeError(f"redirect escaped host allowlist: {final_host}")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if not (
            content_type.startswith("image/")
            or content_type in {"application/octet-stream", "binary/octet-stream"}
        ):
            raise RuntimeError(f"unexpected content type: {content_type}")
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > admission["max_single_file_bytes"]:
                    raise RuntimeError("single-file byte cap exceeded")
                digest.update(chunk)
                handle.write(chunk)
    data = partial.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        image_dhash = dhash(image)
        if width < admission["minimum_width"] or height < admission["minimum_height"]:
            raise RuntimeError(f"image too small: {width}x{height}")
    os.replace(partial, destination)
    return {
        "row_id": record["row_id"],
        "family": family,
        "url_sha256": hashlib.sha256(record["url"].encode("utf-8")).hexdigest(),
        "selection_sha256": record["selection_sha256"],
        "image_path": str(destination),
        "bytes": total,
        "image_sha256": digest.hexdigest(),
        "dhash64": f"{image_dhash:016x}",
        "width": width,
        "height": height,
        "mode": mode,
        "content_type": content_type,
        "caption_words": len(normalized_words(record["caption"])),
    }


def contact_sheet(family: str, records: list[dict], results: dict[int, dict], output: Path) -> None:
    width, cell_width, cell_height = 1600, 400, 310
    rows = (len(records) + 3) // 4
    sheet = Image.new("RGB", (width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, record in enumerate(records):
        x = (index % 4) * cell_width
        y = (index // 4) * cell_height
        result = results.get(record["row_id"])
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#999999")
        if result:
            with Image.open(result["image_path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((380, 220), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + (cell_width - image.width) // 2, y + 5))
            caption = re.sub(r"\s+", " ", record["caption"]).strip()
            caption_lines = textwrap.wrap(caption, width=44)[:3]
            text = (
                f"row {record['row_id']} | {result['width']}x{result['height']}\n"
                + "\n".join(caption_lines)
            )
        else:
            text = f"row {record['row_id']} | DOWNLOAD FAILED"
        draw.multiline_text((x + 8, y + 232), text, fill="black", font=font, spacing=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def main() -> None:
    metadata = json.loads(METADATA_ADMISSION.read_text())
    admission = json.loads(IMAGE_ADMISSION.read_text())
    if sha256_file(METADATA_AUDIT) != admission["required_metadata_audit_sha256"]:
        raise SystemExit("metadata audit receipt hash mismatch")
    if sha256_file(HOSTED_PROFILE) != admission["required_hosted_profile_sha256"]:
        raise SystemExit("hosted profile receipt hash mismatch")
    if admission.get("image_audit_acquisition_authorized") is not True:
        raise SystemExit("image audit acquisition is not authorized")
    if admission.get("formal_training_authorized") is not False:
        raise SystemExit("formal training must remain blocked")

    destination = Path(admission["destination"])
    image_root = destination / "images"
    manifest = destination / "selected-manifest.jsonl"
    selected = select_rows(metadata, admission)
    if len(selected) != admission["max_files"]:
        raise SystemExit("selected sample exceeds or misses the file cap")
    destination.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))

    receipt = {
        "schema_version": "2026-09-13-v1",
        "started_at": utc_now(),
        "repo": admission["repo"],
        "revision": admission["revision"],
        "selection": admission["selection"],
        "selected_rows": len(selected),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "status": "running",
        "formal_training": False,
        "training_prediction_tokens": 0,
        "results": [],
        "failures": [],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    successes: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch_one, row, admission, image_root): row for row in selected}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                successes[row["row_id"]] = result
                receipt["results"].append(result)
            except Exception as exc:
                receipt["failures"].append(
                    {
                        "row_id": row["row_id"],
                        "family": row["family"],
                        "url_sha256": hashlib.sha256(row["url"].encode("utf-8")).hexdigest(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    total_bytes = sum(item["bytes"] for item in receipt["results"])
    if total_bytes > admission["max_total_bytes"]:
        raise SystemExit("total byte cap exceeded")
    image_hashes = Counter(item["image_sha256"] for item in receipt["results"])
    exact_duplicate_items = sum(count - 1 for count in image_hashes.values() if count > 1)
    dhashes = [int(item["dhash64"], 16) for item in receipt["results"]]
    near_pairs = sum(
        1
        for i, left in enumerate(dhashes)
        for right in dhashes[i + 1 :]
        if 0 < (left ^ right).bit_count() <= 2
    )
    by_family = Counter(item["family"] for item in receipt["results"])

    sheets = []
    for family in admission["path_families"]:
        family_records = [row for row in selected if row["family"] == family]
        sheet = ROOT / "evidence" / "human-audit" / f"pixmo-{family}-sample-sheet.png"
        contact_sheet(family, family_records, successes, sheet)
        sheets.append({"family": family, "path": str(sheet), "sha256": sha256_file(sheet)})

    receipt.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "downloaded_images": len(receipt["results"]),
            "failed_images": len(receipt["failures"]),
            "downloaded_bytes": total_bytes,
            "per_family_downloaded": dict(by_family),
            "exact_duplicate_image_items": exact_duplicate_items,
            "near_dhash_unique_pairs_hamming_1_or_2": near_pairs,
            "contact_sheets": sheets,
            "human_audit_status": "pending_manual_review",
            "training_admission": "blocked_pending_manual_image_caption_review_rights_and_benchmark_pixel_dedup",
        }
    )
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
