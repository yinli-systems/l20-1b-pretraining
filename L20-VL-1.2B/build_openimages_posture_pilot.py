#!/usr/bin/env python3
"""Build a caption-corroborated, balanced Open Images posture audit pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from prepare_stage_a_manifest import deduplicate_images, fetch_image, sha256_file


ROOT = Path(__file__).resolve().parent
CLASS_PATTERNS = {
    "Boy": r"\b(boy|boys|child|children|kid|kids)\b",
    "Girl": r"\b(girl|girls|child|children|kid|kids)\b",
    "Man": r"\b(man|men|male|gentleman|person|people)\b",
    "Woman": r"\b(woman|women|female|lady|ladies|person|people)\b",
}
ATTRIBUTE_PATTERNS = {
    "Sit": r"\b(sit|sits|sitting|seated)\b",
    "Stand": r"\b(stands|standing)\b",
}
SECONDARY_MEDIA_PATTERN = (
    r"\b(drawings?|illustrations?|cartoons?|statues?|sculptures?|dolls?|toys?|"
    r"mannequins?|posters?|advertisements?|magazines?|newspapers?|books?|screens?|"
    r"televisions?|paintings?|artworks?|figurines?|photos?|photographs?)\b"
)


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return sha256_file(path)


def validate_protocol(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if "base_protocol" in raw:
        base_record = raw["base_protocol"]
        base_path = Path(base_record["path"])
        if sha256_file(base_path) != base_record["sha256"]:
            raise RuntimeError("base posture pilot protocol hash mismatch")
        protocol = json.loads(base_path.read_text())
        protocol.update(raw)
    else:
        protocol = raw
    if protocol.get("status") != "authorized_posture_data_pilot_only":
        raise RuntimeError("posture pilot protocol is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("posture pilot must not authorize training")
    if sha256_file(Path(__file__)) != protocol["source_code"]["builder_sha256"]:
        raise RuntimeError("posture pilot builder hash mismatch")
    if sha256_file(ROOT / "test_build_openimages_posture_pilot.py") != protocol["source_code"]["test_sha256"]:
        raise RuntimeError("posture pilot test hash mismatch")
    negative = protocol["prerequisite_negative_evidence"]
    if sha256_file(Path(negative["path"])) != negative["sha256"]:
        raise RuntimeError("superseded dataset negative evidence hash mismatch")
    if json.loads(Path(negative["path"]).read_text()).get("decision") != negative["decision"]:
        raise RuntimeError("superseded dataset was not rejected")
    if "attempt_001_negative_evidence" in protocol:
        attempt = protocol["attempt_001_negative_evidence"]
        if sha256_file(Path(attempt["path"])) != attempt["sha256"]:
            raise RuntimeError("attempt-001 negative evidence hash mismatch")
    for name, record in protocol["sources"].items():
        source = Path(record["path"])
        if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"posture pilot source mismatch: {name}")
    return protocol


def load_captions(path: Path) -> dict[str, dict[str, Any]]:
    captions: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        image_id = row.get("image_id")
        caption = str(row.get("caption", "")).strip()
        if not image_id or not caption:
            raise RuntimeError(f"caption line {line_number}: missing image or caption")
        current = captions.setdefault(image_id, {"captions": [], "annotator_ids": []})
        current["captions"].append(caption)
        current["annotator_ids"].append(row.get("annotator_id"))
    return captions


def caption_supports(caption: str, class_name: str, attribute_name: str) -> bool:
    if re.search(SECONDARY_MEDIA_PATTERN, caption, re.IGNORECASE):
        return False
    return bool(
        re.search(CLASS_PATTERNS[class_name], caption, re.IGNORECASE)
        and re.search(ATTRIBUTE_PATTERNS[attribute_name], caption, re.IGNORECASE)
    )


def scan_vrd(protocol: dict[str, Any], captions: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    class_names = {
        row[0]: row[1]
        for row in csv.reader(open(protocol["sources"]["class_names"]["path"], newline=""))
    }
    attribute_names = {
        row[0]: row[1]
        for row in csv.reader(open(protocol["sources"]["attribute_names"]["path"], newline=""))
    }
    filters = protocol["quality_filters"]
    counts: Counter[str] = Counter()
    raw: list[dict[str, Any]] = []
    object_attributes: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    with open(protocol["sources"]["visual_relationships"]["path"], newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts["vrd_rows"] += 1
            if row["RelationshipLabel"] != "is":
                continue
            image_id = row["ImageID"]
            caption_record = captions.get(image_id)
            if caption_record is None:
                continue
            class_name = class_names.get(row["LabelName1"])
            attribute_name = attribute_names.get(row["LabelName2"])
            if class_name not in CLASS_PATTERNS or attribute_name not in ATTRIBUTE_PATTERNS:
                continue
            combined_caption = " ".join(caption_record["captions"])
            if not caption_supports(combined_caption, class_name, attribute_name):
                counts["caption_corroboration_rejection"] += 1
                continue
            box = tuple(float(row[name]) for name in ("XMin1", "XMax1", "YMin1", "YMax1"))
            second = tuple(float(row[name]) for name in ("XMin2", "XMax2", "YMin2", "YMax2"))
            if box != second:
                counts["object_attribute_box_mismatch"] += 1
                continue
            xmin, xmax, ymin, ymax = box
            if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                counts["invalid_box"] += 1
                continue
            width, height = xmax - xmin, ymax - ymin
            if width < filters["min_edge"] or height < filters["min_edge"] or width * height < filters["min_area"]:
                counts["small_box"] += 1
                continue
            box_key = tuple(f"{value:.6f}" for value in box)
            object_key = (image_id, row["LabelName1"], box_key)
            object_attributes[object_key].add(attribute_name)
            raw.append({
                "image_id": image_id,
                "class_id": row["LabelName1"],
                "class_name": class_name,
                "attribute_id": row["LabelName2"],
                "attribute_name": attribute_name,
                "answer": "sitting" if attribute_name == "Sit" else "standing",
                "bbox": list(box),
                "bbox_key": box_key,
                "bbox_area": width * height,
                "caption": combined_caption,
                "caption_annotator_ids": caption_record["annotator_ids"],
            })
    conflicts = {key for key, values in object_attributes.items() if len(values) > 1}
    counts["same_box_sit_stand_conflicts"] = len(conflicts)
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in raw:
        object_key = (row["image_id"], row["class_id"], row["bbox_key"])
        if object_key in conflicts:
            counts["conflict_annotations"] += 1
            continue
        key = (row["image_id"], row["class_name"], row["attribute_name"])
        current = best.get(key)
        if current is None or (-row["bbox_area"], row["bbox_key"]) < (-current["bbox_area"], current["bbox_key"]):
            best[key] = row
    counts["caption_corroborated_box_candidates"] = len(best)
    return list(best.values()), dict(sorted(counts.items()))


def join_metadata(protocol: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_id[row["image_id"]].append(row)
    counts: Counter[str] = Counter()
    with open(protocol["sources"]["image_metadata"]["path"], newline="") as handle:
        for row in csv.DictReader(handle):
            values = by_id.get(row["ImageID"])
            if not values:
                continue
            counts["candidate_metadata_rows"] += 1
            if row["Subset"] != "train":
                counts["wrong_subset"] += 1
                continue
            if row["License"] != protocol["required_image_license"]:
                counts["license_rejection"] += 1
                continue
            if not row.get("Author") or not row.get("OriginalLandingURL"):
                counts["missing_attribution"] += 1
                continue
            try:
                rotation = float(row.get("Rotation") or 0)
            except ValueError:
                counts["invalid_rotation"] += 1
                continue
            if rotation != 0:
                counts["nonzero_rotation"] += 1
                continue
            title = row.get("Title", "")
            if re.search(SECONDARY_MEDIA_PATTERN, title, re.IGNORECASE):
                counts["secondary_media_title"] += 1
                continue
            attribution = {
                "license": row["License"],
                "author": row["Author"],
                "author_profile_url": row.get("AuthorProfileURL", ""),
                "title": title,
                "original_url": row.get("OriginalURL", ""),
                "original_landing_url": row["OriginalLandingURL"],
                "original_md5_base64": row.get("OriginalMD5", ""),
            }
            for value in values:
                value["attribution"] = attribution
                counts["metadata_gate_pass_annotations"] += 1
    eligible = [row for row in candidates if "attribution" in row]
    counts["metadata_gate_pass_images"] = len({row["image_id"] for row in eligible})
    return eligible, dict(sorted(counts.items()))


def select_download_candidates(protocol: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["class_name"], row["attribute_name"])].append(row)
    expected = {(class_name, attribute) for class_name in CLASS_PATTERNS for attribute in ATTRIBUTE_PATTERNS}
    if set(grouped) != expected:
        raise RuntimeError(f"missing posture strata: {sorted(expected - set(grouped))}")
    selected = []
    count = protocol["sampling"]["download_candidates_per_stratum"]
    for key in sorted(grouped):
        values = grouped[key]
        values.sort(key=lambda row: stable_hash(protocol["seed"], "download", *key, row["image_id"], row["bbox_key"]))
        if len(values) < count:
            raise RuntimeError(f"insufficient candidates for {key}: {len(values)} < {count}")
        selected.extend([
            {
                **row,
                "selection_sha256": stable_hash(
                    protocol["seed"], "download", *key, row["image_id"], row["bbox_key"]
                ),
            }
            for row in values[:count]
        ])
    return selected


def download_images(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(protocol["outputs"]["image_directory"])
    root.mkdir(parents=True, exist_ok=True)
    admission = {
        "image_url_template": protocol["image_url_template"],
        "maximum_single_image_bytes": protocol["limits"]["maximum_single_image_bytes"],
    }
    fetched, failures = [], []
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["image_id"], row)
    with ThreadPoolExecutor(max_workers=protocol["limits"]["download_workers"]) as executor:
        futures = {executor.submit(fetch_image, row, root, admission): row for row in unique.values()}
        for future in as_completed(futures):
            row = futures[future]
            try:
                fetched.append(future.result())
            except Exception as error:
                failures.append({"image_id": row["image_id"], "error": f"{type(error).__name__}: {error}"})
    total_bytes = sum(row["image_bytes"] for row in fetched)
    if total_bytes > protocol["limits"]["maximum_downloaded_image_bytes"]:
        raise RuntimeError("posture pilot image byte cap exceeded")
    fetched_by_id = {row["image_id"]: row for row in fetched}
    image_fields = (
        "image_path", "image_url", "image_bytes", "image_sha256", "dhash64",
        "width", "height", "channel_stddev",
    )
    successes = [
        {**row, **{field: fetched_by_id[row["image_id"]][field] for field in image_fields}}
        for row in rows
        if row["image_id"] in fetched_by_id
    ]
    return successes, failures


def choose_final(protocol: dict[str, Any], downloaded: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduplicated, duplicate_rejections = deduplicate_images(downloaded)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated:
        grouped[(row["class_name"], row["attribute_name"])].append(row)
    count = protocol["sampling"]["selected_images_per_stratum"]
    selected = []
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda row: stable_hash(protocol["seed"], "final", *key, row["image_id"]))
        if len(values) < count:
            raise RuntimeError(f"post-download stratum gate failed for {key}: {len(values)} < {count}")
        selected.extend(values[:count])
    return selected, duplicate_rejections


def make_pairs(protocol: dict[str, Any], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(row["class_name"], row["attribute_name"])].append(row)
    pairs = []
    for class_name in sorted(CLASS_PATTERNS):
        sitting = sorted(grouped[(class_name, "Sit")], key=lambda row: stable_hash(protocol["seed"], "pair_sit", row["image_id"]))
        standing = sorted(grouped[(class_name, "Stand")], key=lambda row: stable_hash(protocol["seed"], "pair_stand", row["image_id"]))
        if len(sitting) != len(standing):
            raise RuntimeError(f"unbalanced final stratum: {class_name}")
        for left, right in zip(sitting, standing):
            pair_id = stable_hash(protocol["seed"], "pair", class_name, left["image_id"], right["image_id"])[:24]
            pairs.append({
                "schema_version": "2026-09-14-v1",
                "pair_id": pair_id,
                "class_name": class_name,
                "candidate_answers": ["sitting", "standing"],
                "image_a": left,
                "image_b": right,
            })
    return sorted(pairs, key=lambda row: row["pair_id"])


def draw_box(image: Image.Image, bbox: list[float]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    xmin, xmax, ymin, ymax = bbox
    coordinates = (
        round(xmin * output.width), round(ymin * output.height),
        round(xmax * output.width), round(ymax * output.height),
    )
    width = max(3, round(min(output.size) / 100))
    for offset in range(width):
        draw.rectangle((coordinates[0] - offset, coordinates[1] - offset, coordinates[2] + offset, coordinates[3] + offset), outline=(255, 32, 32))
    return output


def render_pair(pair: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    half = size[0] // 2
    for index, key in enumerate(("image_a", "image_b")):
        row = pair[key]
        with Image.open(row["image_path"]) as image:
            boxed = draw_box(image, row["bbox"])
            fitted = ImageOps.contain(boxed, (half - 12, size[1] - 48))
        x = index * half + (half - fitted.width) // 2
        tile.paste(fitted, (x, 24 + (size[1] - 48 - fitted.height) // 2))
        draw.text((index * half + 6, 5), f"{key[-1].upper()}: {row['class_name']} / {row['answer']}", fill="black", font=font)
    draw.text((6, size[1] - 18), pair["pair_id"], fill="black", font=font)
    return tile


def render_sheets(protocol: dict[str, Any], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    directory = Path(protocol["outputs"]["audit_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    per_sheet = protocol["human_audit"]["pairs_per_sheet"]
    tile_size = (720, 320)
    columns = 2
    records = []
    for page, start in enumerate(range(0, len(pairs), per_sheet), 1):
        page_pairs = pairs[start : start + per_sheet]
        height = ((len(page_pairs) + columns - 1) // columns) * tile_size[1]
        sheet = Image.new("RGB", (columns * tile_size[0], height), (236, 239, 244))
        for index, pair in enumerate(page_pairs):
            sheet.paste(render_pair(pair, tile_size), ((index % columns) * tile_size[0], (index // columns) * tile_size[1]))
        path = directory / f"openimages-posture-pilot-sheet-{page:02d}.png"
        sheet.save(path, optimize=True)
        records.append({"path": str(path), "sha256": sha256_file(path), "pairs": len(page_pairs)})
    return records


def build(protocol_path: Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    destination = Path(protocol["outputs"]["root"])
    free_before = shutil.disk_usage(destination.parent if destination.parent.exists() else ROOT).free
    if free_before < protocol["limits"]["minimum_free_bytes_before"]:
        raise RuntimeError(f"insufficient free disk: {free_before}")
    captions = load_captions(Path(protocol["sources"]["localized_narratives"]["path"]))
    candidates, scan_counts = scan_vrd(protocol, captions)
    eligible, metadata_counts = join_metadata(protocol, candidates)
    downloads = select_download_candidates(protocol, eligible)
    downloaded, failures = download_images(protocol, downloads)
    selected, duplicate_rejections = choose_final(protocol, downloaded)
    selected_ids = {row["image_id"] for row in selected}
    removed_files, removed_bytes = 0, 0
    image_directory = Path(protocol["outputs"]["image_directory"])
    for path in image_directory.glob("*.jpg"):
        if path.stem not in selected_ids:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    pairs = make_pairs(protocol, selected)
    image_manifest = Path(protocol["outputs"]["image_manifest"])
    image_sha = write_jsonl_atomic(image_manifest, sorted(selected, key=lambda row: row["image_id"]))
    pair_manifest = Path(protocol["outputs"]["pair_manifest"])
    pair_sha = write_jsonl_atomic(pair_manifest, pairs)
    sheets = render_sheets(protocol, pairs)
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_pending_full_human_audit",
        "training_authorized": False,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(destination).free,
        "scan_counts": scan_counts,
        "metadata_counts": metadata_counts,
        "download_candidates": len(downloads),
        "download_successes": len(downloaded),
        "download_failures": failures,
        "near_or_exact_duplicate_rejections": duplicate_rejections,
        "removed_unselected_files": removed_files,
        "removed_unselected_bytes": removed_bytes,
        "selected_images": len(selected),
        "selected_by_stratum": [
            {"class_name": key[0], "attribute_name": key[1], "images": value}
            for key, value in sorted(Counter((row["class_name"], row["attribute_name"]) for row in selected).items())
        ],
        "image_manifest": {"path": str(image_manifest), "sha256": image_sha, "rows": len(selected)},
        "pair_manifest": {"path": str(pair_manifest), "sha256": pair_sha, "rows": len(pairs)},
        "audit_sheets": sheets,
        "claim_boundary": (
            "A balanced caption-corroborated posture pilot was acquired and mechanically checked. "
            "Every pair still requires human visual review; no training or model claim is authorized."
        ),
    }
    write_json_atomic(Path(protocol["outputs"]["receipt"]), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.protocol)
    print(json.dumps({
        "status": receipt["status"],
        "selected_images": receipt["selected_images"],
        "pairs": receipt["pair_manifest"]["rows"],
        "audit_sheets": len(receipt["audit_sheets"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
