#!/usr/bin/env python3
"""Build a bounded, auditable CLEVR + Open Images Stage-A manifest."""
from __future__ import annotations

import base64
import csv
import hashlib
import heapq
import io
import json
import math
import os
import re
import shutil
import textwrap
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parent
ADMISSION = ROOT / "stage_a_data_admission.json"
ACQUISITION = ROOT / "evidence" / "stage-a-source-acquisition.json"
RECEIPT = ROOT / "evidence" / "stage-a-content-audit.json"
SEED = "20260913-stage-a-v1"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
URL_RE = re.compile(r"(?:https?://|www\.)", re.I)
PERSON_TERMS = {
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "person", "persons", "people", "child", "children", "baby", "babies",
    "gentleman", "gentlemen", "lady", "ladies", "crowd", "family",
    "human", "humans", "kid", "kids", "infant", "infants", "toddler",
    "toddlers", "teen", "teens", "teenager", "teenagers", "adult",
    "adults", "worker", "workers", "pedestrian", "pedestrians", "tourist",
    "tourists", "athlete", "athletes", "rider", "riders", "couple",
    "couples", "bride", "groom",
}
HIGH_RISK_TERMS = {
    "nude", "naked", "porn", "sexual", "sex", "weapon", "gun", "rifle",
    "blood", "corpse", "dead", "injury", "hospital", "patient", "passport",
    "license plate", "credit card", "address", "phone number", "email address",
}
HTTP_STATE = threading.local()


def http_session() -> requests.Session:
    session = getattr(HTTP_STATE, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=2)
        session.mount("https://", adapter)
        HTTP_STATE.session = session
    return session


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def selection_score(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}|{namespace}|{value}".encode()).hexdigest()


def dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    result = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            result = (result << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return result


def hamming_distance(left: int, right: int) -> int:
    """Python 3.9-compatible 64-bit Hamming distance."""
    return bin(left ^ right).count("1")


def caption_gate(caption: str) -> tuple[bool, str]:
    words = WORD_RE.findall(caption)
    lowered = caption.lower()
    # Normalize possessives so "person's" cannot bypass the person filter.
    normalized_words = {word.lower().removesuffix("'s") for word in words}
    if not 12 <= len(words) <= 100:
        return False, "word_count"
    if sum(character.isascii() for character in caption) / max(1, len(caption)) < 0.98:
        return False, "non_ascii"
    if EMAIL_RE.search(caption) or PHONE_RE.search(caption) or URL_RE.search(caption):
        return False, "pii_or_url_pattern"
    if PERSON_TERMS & normalized_words:
        return False, "person_term"
    if any(term in lowered for term in HIGH_RISK_TERMS):
        return False, "high_risk_term"
    if len(set(normalized_words)) < 8:
        return False, "low_lexical_diversity"
    return True, "pass"


def smallest_records(records: Iterable[dict], count: int) -> list[dict]:
    heap: list[tuple[int, str, dict]] = []
    for record in records:
        score = int(record["selection_sha256"], 16)
        item = (-score, record["selection_sha256"], record)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda item: item[1])]


def select_caption_candidates(path: Path, limit: int) -> tuple[list[dict], dict]:
    counts = Counter()
    seen_images: set[str] = set()
    seen_text: set[str] = set()

    def records() -> Iterable[dict]:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                counts["rows"] += 1
                row = json.loads(line)
                image_id = str(row.get("image_id", "")).lower()
                caption = str(row.get("caption", "")).strip()
                if not re.fullmatch(r"[0-9a-f]{16}", image_id):
                    counts["invalid_image_id"] += 1
                    continue
                accepted, reason = caption_gate(caption)
                if not accepted:
                    counts[reason] += 1
                    continue
                text_key = normalized(caption)
                if image_id in seen_images:
                    counts["duplicate_image_id"] += 1
                    continue
                if text_key in seen_text:
                    counts["duplicate_normalized_caption"] += 1
                    continue
                seen_images.add(image_id)
                seen_text.add(text_key)
                counts["caption_gate_pass"] += 1
                yield {
                    "image_id": image_id,
                    "caption": caption,
                    "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
                    "annotator_id": row.get("annotator_id"),
                    "source_line": line_number,
                    "selection_sha256": selection_score("open-images", image_id),
                }

    selected = smallest_records(records(), limit)
    if len(selected) != limit:
        raise RuntimeError(f"insufficient caption candidates: {len(selected)} != {limit}")
    counts["deterministically_selected_candidates"] = len(selected)
    return selected, dict(sorted(counts.items()))


def attach_image_metadata(path: Path, candidates: list[dict], required_license: str) -> tuple[list[dict], dict]:
    by_id = {record["image_id"]: record for record in candidates}
    counts = Counter()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("ImageID", "").lower()
            if image_id not in by_id:
                continue
            counts["candidate_metadata_rows"] += 1
            if row.get("License") != required_license:
                counts["wrong_license"] += 1
                continue
            if not row.get("Author") or not row.get("OriginalLandingURL"):
                counts["missing_attribution"] += 1
                continue
            record = by_id[image_id]
            record["attribution"] = {
                "license": row["License"],
                "author": row["Author"],
                "author_profile_url": row.get("AuthorProfileURL", ""),
                "title": row.get("Title", ""),
                "original_url": row.get("OriginalURL", ""),
                "original_landing_url": row["OriginalLandingURL"],
                "original_md5_base64": row.get("OriginalMD5", ""),
            }
            counts["license_and_attribution_pass"] += 1
    eligible = [record for record in candidates if "attribution" in record]
    eligible.sort(key=lambda record: record["selection_sha256"])
    return eligible, dict(sorted(counts.items()))


def fetch_image(record: dict, image_root: Path, admission: dict) -> dict:
    image_id = record["image_id"]
    url = admission["image_url_template"].format(image_id=image_id)
    destination = image_root / f"{image_id}.jpg"
    partial = destination.with_suffix(".partial")
    if destination.exists():
        data = destination.read_bytes()
    else:
        error: Exception | None = None
        for attempt in range(3):
            try:
                with http_session().get(url, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    total = 0
                    digest = hashlib.sha256()
                    with partial.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            if not block:
                                continue
                            total += len(block)
                            if total > admission["maximum_single_image_bytes"]:
                                raise RuntimeError("single image byte cap exceeded")
                            digest.update(block)
                            handle.write(block)
                os.replace(partial, destination)
                data = destination.read_bytes()
                break
            except Exception as current:
                error = current
                partial.unlink(missing_ok=True)
                time.sleep(1 + attempt)
        else:
            raise RuntimeError(f"download failed after retries: {error}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        if width < 224 or height < 224:
            raise RuntimeError(f"image below 224px: {width}x{height}")
        if max(width / height, height / width) > 4.0:
            raise RuntimeError(f"extreme aspect ratio: {width}x{height}")
        rgb = image.convert("RGB")
        stats = ImageStat.Stat(rgb.resize((64, 64), Image.Resampling.BILINEAR))
        channel_stddev = sum(stats.stddev) / 3
        if channel_stddev < 8.0:
            raise RuntimeError(f"near-blank image stddev={channel_stddev:.3f}")
        image_dhash = dhash(rgb)
    return {
        **record,
        "image_path": str(destination),
        "image_url": url,
        "image_bytes": len(data),
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "dhash64": f"{image_dhash:016x}",
        "width": width,
        "height": height,
        "channel_stddev": channel_stddev,
    }


def deduplicate_images(records: list[dict], threshold: int = 4) -> tuple[list[dict], list[dict]]:
    records = sorted(records, key=lambda record: record["selection_sha256"])
    exact_seen: dict[str, str] = {}
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept: list[dict] = []
    rejected: list[dict] = []
    widths = (13, 13, 13, 13, 12)
    shifts = []
    remaining = 64
    for width in widths:
        remaining -= width
        shifts.append((remaining, (1 << width) - 1))
    for record in records:
        prior = exact_seen.get(record["image_sha256"])
        if prior is not None:
            rejected.append({"image_id": record["image_id"], "reason": "exact", "canonical": prior})
            continue
        value = int(record["dhash64"], 16)
        candidates: set[int] = set()
        for index, (shift, mask) in enumerate(shifts):
            candidates.update(buckets[(index, (value >> shift) & mask)])
        near = next(
            (
                kept[index]
                for index in sorted(candidates)
                if hamming_distance(value, int(kept[index]["dhash64"], 16)) <= threshold
            ),
            None,
        )
        if near is not None:
            rejected.append({"image_id": record["image_id"], "reason": "dhash_lte_4", "canonical": near["image_id"]})
            continue
        exact_seen[record["image_sha256"]] = record["image_id"]
        kept_index = len(kept)
        kept.append(record)
        for index, (shift, mask) in enumerate(shifts):
            buckets[(index, (value >> shift) & mask)].append(kept_index)
    return kept, rejected


def write_jsonl(path: Path, records: Iterable[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return sha256_file(path)


def clevr_records(root: Path, split: str, count: int) -> list[dict]:
    questions_path = root / "questions" / f"CLEVR_{split}_questions.json"
    data = json.loads(questions_path.read_text())
    by_image: dict[int, dict] = {}
    seen_question: set[str] = set()
    for row in data["questions"]:
        question = str(row["question"]).strip()
        text_key = normalized(question)
        if text_key in seen_question:
            continue
        seen_question.add(text_key)
        image_index = int(row["image_index"])
        score = selection_score(f"clevr-{split}", f"{image_index}|{question}")
        record = {
            "source": "clevr_v1_0",
            "source_split": split,
            "image_id": f"CLEVR_{split}_{image_index:06d}",
            "image_path": str(root / "images" / split / row["image_filename"]),
            "question": question,
            "answer": str(row["answer"]),
            "question_family_index": row.get("question_family_index"),
            "functional_program": row.get("program"),
            "selection_sha256": score,
            "license": "CC-BY-4.0",
        }
        previous = by_image.get(image_index)
        if previous is None or score < previous["selection_sha256"]:
            by_image[image_index] = record
    selected = smallest_records(by_image.values(), count)
    if len(selected) != count:
        raise RuntimeError(f"insufficient CLEVR {split} records")
    for record in selected:
        path = Path(record["image_path"])
        with Image.open(path) as image:
            image.load()
            record["image_sha256"] = sha256_file(path)
            record["width"], record["height"] = image.size
    return selected


def make_contact_sheet(records: list[dict], output: Path, source: str) -> None:
    sample = smallest_records(
        [
            {**record, "selection_sha256": selection_score(f"audit-{source}", record["image_id"])}
            for record in records
        ],
        100,
    )
    columns, cell_width, cell_height = 5, 360, 300
    rows = math.ceil(len(sample) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for index, record in enumerate(sample):
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#999")
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((340, 205), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (cell_width - image.width) // 2, y + 5))
        text = f"Prompt: {record['prompt']} Response: {record['response']}"
        lines = textwrap.wrap(re.sub(r"\s+", " ", text), 45)[:4]
        draw.multiline_text((x + 8, y + 215), f"{record['image_id']}\n" + "\n".join(lines), fill="black", font=font, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    acquisition = json.loads(ACQUISITION.read_text())
    if acquisition.get("status") != "complete_pending_content_audit":
        raise SystemExit("source acquisition is not complete")
    expected_source_admission = admission.get(
        "source_acquisition_admission_sha256", sha256_file(ADMISSION)
    )
    if acquisition.get("admission_sha256") != expected_source_admission:
        raise SystemExit("source acquisition used an unrecognized admission file")
    destination = Path(admission["limits"]["destination"])
    source_root = destination / "sources"
    prepared = destination / "prepared"
    receipt = {
        "schema_version": "2026-09-13-v1",
        "started_at": utc_now(),
        "status": "running",
        "admission_sha256": sha256_file(ADMISSION),
        "source_acquisition_sha256": sha256_file(ACQUISITION),
        "formal_training": False,
        "training_prediction_tokens": 0,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    narratives = admission["sources"]["open_images_localized_narratives"]
    candidates, caption_counts = select_caption_candidates(
        source_root / "open_images_train_v6_captions.jsonl", 30_000
    )
    eligible, metadata_counts = attach_image_metadata(
        source_root / "train-images-boxable-with-rotation.csv",
        candidates,
        narratives["required_image_license"],
    )
    if len(eligible) < 27_500:
        raise RuntimeError(f"insufficient licensed Open Images candidates: {len(eligible)}")
    download_candidates = eligible[:27_500]
    image_root = prepared / "open-images" / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failures: list[dict] = []
    rejected_files = 0
    rejected_file_bytes = 0
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {
            pool.submit(fetch_image, record, image_root, narratives): record
            for record in download_candidates
        }
        for completed, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"image_id": record["image_id"], "error": f"{type(error).__name__}: {error}"})
                rejected_path = image_root / f"{record['image_id']}.jpg"
                if rejected_path.exists():
                    rejected_file_bytes += rejected_path.stat().st_size
                    rejected_path.unlink()
                    rejected_files += 1
            if completed % 1000 == 0:
                print(f"OPEN_IMAGES_PROGRESS completed={completed}/27500 success={len(results)}", flush=True)
    downloaded_bytes = sum(record["image_bytes"] for record in results)
    if downloaded_bytes > narratives["maximum_image_bytes"]:
        raise RuntimeError("Open Images byte cap exceeded")
    deduplicated, duplicate_rejections = deduplicate_images(results)
    target = narratives["target_images"]
    if len(deduplicated) < target:
        raise RuntimeError(f"not enough images after deduplication: {len(deduplicated)} < {target}")
    selected = deduplicated[:target]
    selected_ids = {record["image_id"] for record in selected}
    removed_files = 0
    removed_bytes = 0
    # Prune from the directory rather than only this run's result list. This also
    # removes artifacts left by a superseded deterministic selection.
    for path in image_root.glob("*.jpg"):
        if path.stem not in selected_ids:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    selected.sort(key=lambda record: record["selection_sha256"])
    for index, record in enumerate(selected):
        record["source"] = "open_images_localized_narratives"
        record["split"] = "development" if index % 10 == 0 else "train"
        record["prompt"] = "Describe the image in detail."
        record["response"] = record.pop("caption")
    open_images_manifest = prepared / "open-images-manifest.jsonl"
    open_images_sha = write_jsonl(open_images_manifest, selected)

    clevr_root = source_root / "CLEVR_v1.0"
    clevr_train = clevr_records(clevr_root, "train", 25_000)
    clevr_dev = clevr_records(clevr_root, "val", 2_500)
    for record in clevr_train:
        record["split"] = "train"
        record["prompt"] = record.pop("question")
        record["response"] = record.pop("answer")
    for record in clevr_dev:
        record["split"] = "development"
        record["prompt"] = record.pop("question")
        record["response"] = record.pop("answer")
    clevr_manifest = prepared / "clevr-manifest.jsonl"
    clevr_sha = write_jsonl(clevr_manifest, clevr_train + clevr_dev)

    diagnostic_hashes: set[str] = set()
    diagnostic_manifest = ROOT / "data" / "diagnostic" / "counterfactual-v1" / "manifest.jsonl"
    if not diagnostic_manifest.exists():
        diagnostic_manifest = Path("/home/hhai/l20-vl-1.2b/data/diagnostic/counterfactual-v1/manifest.jsonl")
    if diagnostic_manifest.exists():
        for line in diagnostic_manifest.read_text().splitlines():
            row = json.loads(line)
            diagnostic_hashes.update(
                row[field]
                for field in ("base_image_sha256", "edited_image_sha256", "invariant_image_sha256")
            )
    selected_hashes = {record["image_sha256"] for record in selected + clevr_train + clevr_dev}
    overlap = selected_hashes & diagnostic_hashes
    if overlap:
        raise RuntimeError(f"controlled diagnostic exact-image overlap: {len(overlap)}")

    audit_root = ROOT / "evidence" / "human-audit"
    make_contact_sheet(selected, audit_root / "stage-a-open-images-sheet.png", "open-images")
    make_contact_sheet(clevr_train + clevr_dev, audit_root / "stage-a-clevr-sheet.png", "clevr")

    receipt.update(
        {
            "status": "complete_pending_human_audit",
            "completed_at": utc_now(),
            "caption_filter_counts": caption_counts,
            "metadata_filter_counts": metadata_counts,
            "open_images": {
                "download_candidates": len(download_candidates),
                "download_successes": len(results),
                "download_failures": failures,
                "removed_rejected_files": rejected_files,
                "removed_rejected_file_bytes": rejected_file_bytes,
                "downloaded_bytes_before_pruning": downloaded_bytes,
                "exact_or_near_duplicate_rejections": duplicate_rejections,
                "selected_images": len(selected),
                "train_images": sum(record["split"] == "train" for record in selected),
                "development_images": sum(record["split"] == "development" for record in selected),
                "removed_unselected_files": removed_files,
                "removed_unselected_bytes": removed_bytes,
                "manifest": str(open_images_manifest),
                "manifest_sha256": open_images_sha,
            },
            "clevr": {
                "train_examples": len(clevr_train),
                "development_examples": len(clevr_dev),
                "manifest": str(clevr_manifest),
                "manifest_sha256": clevr_sha,
            },
            "controlled_diagnostic_exact_image_overlap": len(overlap),
            "contact_sheets": {
                "open_images": {
                    "path": str(audit_root / "stage-a-open-images-sheet.png"),
                    "sha256": sha256_file(audit_root / "stage-a-open-images-sheet.png"),
                    "samples": 100,
                },
                "clevr": {
                    "path": str(audit_root / "stage-a-clevr-sheet.png"),
                    "sha256": sha256_file(audit_root / "stage-a-clevr-sheet.png"),
                    "samples": 100,
                },
            },
            "free_bytes_after_preparation": shutil.disk_usage(destination).free,
            "formal_training": False,
            "training_prediction_tokens": 0,
            "claim_boundary": "Automated checks and deterministic selection do not replace the required human audit or authorize model training.",
        }
    )
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if RECEIPT.exists():
            failed = json.loads(RECEIPT.read_text())
            failed["status"] = "failed"
            failed["failed_at"] = utc_now()
            failed["error"] = f"{type(error).__name__}: {error}"
            RECEIPT.write_text(json.dumps(failed, indent=2, sort_keys=True) + "\n")
        raise
