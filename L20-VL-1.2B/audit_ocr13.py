#!/usr/bin/env python3
"""Audit acquired synthetic OCR components without extracting their tar archives."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
import random
import statistics
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/hhai/l20-vl-1.2b")
DATA = ROOT / "data/audit/nvidia-ocr13"
ACQUISITION = ROOT / "evidence/ocr13-acquisition.json"
OUTPUT = ROOT / "evidence/ocr13-integrity-audit.json"
SHEETS = ROOT / "evidence/human-audit"
COMPONENTS = ("ocr_1", "ocr_3")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def near_pair_count(values: set[int]) -> int:
    count = 0
    for value in values:
        for first in range(64):
            other = value ^ (1 << first)
            if other > value and other in values:
                count += 1
        for first in range(64):
            once = value ^ (1 << first)
            for second in range(first + 1, 64):
                other = once ^ (1 << second)
                if other > value and other in values:
                    count += 1
    return count


def render_sheet(component: str, samples: list[tuple[str, str, Image.Image]]) -> Path:
    cols, cell_w, cell_h = 3, 340, 260
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (name, label, image) in enumerate(samples):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        thumbnail = image.copy()
        thumbnail.thumbnail((cell_w - 12, cell_h - 58), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, (x + (cell_w - thumbnail.width) // 2, y + 4))
        text = f"{name} | target: {label[:72]}"
        draw.rectangle((x, y + cell_h - 50, x + cell_w - 1, y + cell_h - 1), fill="white")
        draw.text((x + 4, y + cell_h - 46), text, fill="black", font=font)
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#888888")
    SHEETS.mkdir(parents=True, exist_ok=True)
    path = SHEETS / f"{component}-sample-sheet.png"
    sheet.save(path, optimize=True)
    return path


def audit_component(component: str) -> tuple[dict, dict[str, str], dict[str, int]]:
    jsonl = DATA / f"{component}.jsonl"
    records = []
    ids = set()
    image_refs = set()
    schema_errors = []
    prompt_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    label_lengths = []
    non_ascii_labels = 0
    with jsonl.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                schema_errors.append(f"line {line_number}: {error}")
                continue
            if set(record) != {"id", "image", "conversations"}:
                schema_errors.append(f"line {line_number}: unexpected top-level schema")
                continue
            conversations = record["conversations"]
            if (
                not isinstance(conversations, list)
                or len(conversations) != 2
                or conversations[0].get("from") != "human"
                or conversations[1].get("from") != "gpt"
                or "<image>" not in conversations[0].get("value", "")
            ):
                schema_errors.append(f"line {line_number}: invalid conversation schema")
                continue
            if record["id"] in ids:
                schema_errors.append(f"line {line_number}: duplicate id")
            ids.add(record["id"])
            if record["image"] in image_refs:
                schema_errors.append(f"line {line_number}: duplicate image reference")
            image_refs.add(record["image"])
            prompt = conversations[0]["value"].replace("<image>\n", "")
            label = conversations[1]["value"]
            prompt_counter[prompt] += 1
            label_counter[label] += 1
            label_lengths.append(len(label))
            non_ascii_labels += int(any(ord(character) > 127 for character in label))
            records.append((record["image"], label))

    rng = random.Random(20260913 + int(component.split("_")[1]))
    selected = set(rng.sample([image for image, _ in records], min(12, len(records))))
    label_by_image = dict(records)
    tar_names = set()
    unsafe_members = []
    decode_failures = []
    exact_hashes: dict[str, str] = {}
    perceptual_hashes: dict[str, int] = {}
    selected_images: list[tuple[str, str, Image.Image]] = []
    image_modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    for archive in sorted((DATA / f"{component}_images").glob("*.tar")):
        with tarfile.open(archive, "r:") as tar:
            for member in tar:
                pure = PurePosixPath(member.name)
                if member.name.startswith("/") or ".." in pure.parts or not member.isfile():
                    unsafe_members.append(member.name)
                    continue
                if member.name in tar_names:
                    unsafe_members.append(f"duplicate-member:{member.name}")
                    continue
                tar_names.add(member.name)
                extracted = tar.extractfile(member)
                if extracted is None:
                    decode_failures.append(f"{member.name}: no stream")
                    continue
                raw = extracted.read()
                exact_hashes[member.name] = hashlib.sha256(raw).hexdigest()
                try:
                    with Image.open(io.BytesIO(raw)) as probe:
                        probe.verify()
                    with Image.open(io.BytesIO(raw)) as image:
                        converted = image.convert("RGB")
                        perceptual_hashes[member.name] = dhash(converted)
                        image_modes[image.mode] += 1
                        dimensions[f"{image.width}x{image.height}"] += 1
                        if member.name in selected:
                            selected_images.append(
                                (member.name, label_by_image[member.name], converted.copy())
                            )
                except Exception as error:
                    decode_failures.append(f"{member.name}: {error!r}")

    exact_counts = Counter(exact_hashes.values())
    dhash_counts = Counter(perceptual_hashes.values())
    exact_duplicate_items = sum(count - 1 for count in exact_counts.values() if count > 1)
    perceptual_collision_items = sum(count - 1 for count in dhash_counts.values() if count > 1)
    sheet = render_sheet(component, sorted(selected_images))
    result = {
        "records": len(records),
        "unique_ids": len(ids),
        "unique_image_references": len(image_refs),
        "schema_errors": schema_errors,
        "tar_members": len(tar_names),
        "unsafe_tar_members": unsafe_members,
        "missing_image_references": sorted(image_refs - tar_names)[:100],
        "unreferenced_tar_members": sorted(tar_names - image_refs)[:100],
        "decode_failures": decode_failures[:100],
        "decoded_images": len(perceptual_hashes),
        "image_modes": dict(image_modes),
        "top_dimensions": dict(dimensions.most_common(20)),
        "label_length": {
            "min": min(label_lengths),
            "p50": percentile(label_lengths, 0.50),
            "p95": percentile(label_lengths, 0.95),
            "p99": percentile(label_lengths, 0.99),
            "max": max(label_lengths),
            "mean": statistics.mean(label_lengths),
        },
        "labels_with_non_ascii_fraction": non_ascii_labels / len(records),
        "duplicate_label_items": sum(count - 1 for count in label_counter.values() if count > 1),
        "prompt_distribution": dict(prompt_counter),
        "exact_duplicate_image_items": exact_duplicate_items,
        "exact_dhash_collision_items": perceptual_collision_items,
        "near_dhash_unique_pairs_hamming_1_or_2": near_pair_count(set(dhash_counts)),
        "human_audit_sheet": str(sheet),
        "human_audit_status": "pending_manual_review",
    }
    return result, exact_hashes, perceptual_hashes


def main() -> None:
    acquisition = json.loads(ACQUISITION.read_text())
    if acquisition.get("status") != "complete" or acquisition.get("errors"):
        raise SystemExit("FAIL: acquisition receipt is not complete and clean")
    results = {}
    exact_by_component = {}
    perceptual_by_component = {}
    for component in COMPONENTS:
        result, exact, perceptual = audit_component(component)
        results[component] = result
        exact_by_component[component] = exact
        perceptual_by_component[component] = perceptual

    cross_exact = set(exact_by_component["ocr_1"].values()) & set(
        exact_by_component["ocr_3"].values()
    )
    cross_dhash = set(perceptual_by_component["ocr_1"].values()) & set(
        perceptual_by_component["ocr_3"].values()
    )
    blocking = []
    for component, result in results.items():
        for key in ("schema_errors", "unsafe_tar_members", "missing_image_references", "decode_failures"):
            if result[key]:
                blocking.append(f"{component}:{key}")
        if result["records"] != result["tar_members"]:
            blocking.append(f"{component}:record_tar_count_mismatch")

    payload = {
        "schema_version": "2026-09-13-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": acquisition["source_revision"],
        "acquisition_receipt_status": acquisition["status"],
        "archive_extracted": False,
        "components": results,
        "cross_component_exact_duplicate_hashes": len(cross_exact),
        "cross_component_exact_dhash_collisions": len(cross_dhash),
        "automated_integrity_status": "pass" if not blocking else "fail",
        "blocking_findings": blocking,
        "formal_training": False,
        "training_prediction_tokens": 0,
        "training_admission": "blocked_pending_manual_sample_review_and_source_mix_completion",
        "claim_boundary": "Decode and hash checks establish file/schema integrity, not OCR-label correctness or general multimodal quality.",
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if blocking:
        raise SystemExit("FAIL: automated integrity audit found blocking issues")


if __name__ == "__main__":
    main()
