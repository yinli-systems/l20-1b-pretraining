#!/usr/bin/env python3
"""Build a sealed-split Open Images posture pool for exhaustive human audit.

The builder uses Open Images validation annotations, never model-under-test
outputs.  Labels remain untrusted until every target is visually reviewed.
No pairing, training, or model evaluation is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import build_openimages_posture_pilot as v1
import build_openimages_posture_standing_supplement_v3 as standing_v3
from render_openimages_posture_audit_detail import panel


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_openimages_validation_posture_candidate_audit_only_v1"
TARGET_CLASSES = ("Boy", "Girl", "Man", "Woman")
TARGET_ATTRIBUTES = ("Sit", "Stand")
SECONDARY_MEDIA_PATTERN = re.compile(
    r"\b(drawings?|illustrations?|cartoons?|statues?|sculptures?|dolls?|toys?|"
    r"mannequins?|posters?|advertisements?|magazines?|newspapers?|books?|screens?|"
    r"televisions?|paintings?|artworks?|figurines?)\b",
    re.IGNORECASE,
)
TRUNCATION_CONTEXT_PATTERN = re.compile(
    r"\b(portraits?|headshots?|selfies?|close[ -]?ups?|podiums?)\b",
    re.IGNORECASE,
)


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
    return v1.sha256_file(path)


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("confirmation candidate audit is not authorized")
    for field in (
        "training_authorized",
        "model_evaluation_authorized",
        "pairing_authorized",
        "automatic_training_start",
        "automatic_evaluation_start",
    ):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    source = protocol["source_code"]
    checks = (
        (Path(__file__), "builder_sha256"),
        (ROOT / "test_build_openimages_posture_confirmation_candidates_v1.py", "test_sha256"),
        (ROOT / "build_openimages_posture_pilot.py", "download_dedup_sha256"),
        (ROOT / "build_openimages_posture_standing_supplement_v3.py", "pose_ranking_sha256"),
        (ROOT / "render_openimages_posture_audit_detail.py", "audit_renderer_sha256"),
    )
    for file_path, key in checks:
        if v1.sha256_file(file_path) != source[key]:
            raise RuntimeError(f"source hash mismatch: {file_path.name}")
    for name, item in protocol["prerequisites"].items():
        if v1.sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"prerequisite hash mismatch: {name}")
    acquisition = json.loads(Path(protocol["prerequisites"]["acquisition_receipt"]["path"]).read_text())
    if acquisition.get("status") != "complete_verified_validation_annotation_acquisition_only":
        raise RuntimeError("validation source acquisition is incomplete")
    if acquisition.get("training_started") or acquisition.get("model_evaluation_started"):
        raise RuntimeError("acquisition boundary was violated")
    for name, item in protocol["pose_model"]["artifacts"].items():
        if v1.sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"pose model artifact mismatch: {name}")
    return protocol


def geometry_passes(attribute: str, bbox: list[float], filters: dict[str, Any]) -> bool:
    xmin, xmax, ymin, ymax = bbox
    width, height = xmax - xmin, ymax - ymin
    current = filters[attribute]
    if attribute == "Sit":
        return (
            width >= current["min_width"]
            and height >= current["min_height"]
            and width * height >= current["min_area"]
        )
    return (
        width >= current["min_width"]
        and width <= current["max_width"]
        and height >= current["min_height"]
        and width * height >= current["min_area"]
        and height / width >= current["min_bbox_height_to_width"]
    )


def scan_vrd(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sources = protocol["sources"]
    class_names = {
        row[0]: row[1]
        for row in csv.reader(open(sources["class_names"]["path"], newline=""))
    }
    attribute_names = {
        row[0]: row[1]
        for row in csv.reader(open(sources["attribute_names"]["path"], newline=""))
    }
    raw: list[dict[str, Any]] = []
    object_attributes: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    with open(sources["visual_relationships"]["path"], newline="") as handle:
        for row in csv.DictReader(handle):
            counts["vrd_rows"] += 1
            if row["RelationshipLabel"] != "is":
                continue
            counts["attribute_rows"] += 1
            class_name = class_names.get(row["LabelName1"])
            attribute_name = attribute_names.get(row["LabelName2"])
            if class_name not in TARGET_CLASSES or attribute_name not in TARGET_ATTRIBUTES:
                continue
            counts[f"target_label_{class_name}_{attribute_name}"] += 1
            bbox = tuple(float(row[name]) for name in ("XMin1", "XMax1", "YMin1", "YMax1"))
            second = tuple(float(row[name]) for name in ("XMin2", "XMax2", "YMin2", "YMax2"))
            if bbox != second:
                counts["object_attribute_box_mismatch"] += 1
                continue
            xmin, xmax, ymin, ymax = bbox
            if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                counts["invalid_box"] += 1
                continue
            if not geometry_passes(attribute_name, list(bbox), protocol["geometry_filters"]):
                counts[f"geometry_rejection_{attribute_name}"] += 1
                continue
            bbox_key = tuple(f"{value:.6f}" for value in bbox)
            object_key = (row["ImageID"], row["LabelName1"], bbox_key)
            object_attributes[object_key].add(attribute_name)
            raw.append(
                {
                    "image_id": row["ImageID"],
                    "source_split": "validation",
                    "class_id": row["LabelName1"],
                    "class_name": class_name,
                    "attribute_id": row["LabelName2"],
                    "attribute_name": attribute_name,
                    "answer": "sitting" if attribute_name == "Sit" else "standing",
                    "bbox": list(bbox),
                    "bbox_key": bbox_key,
                    "bbox_area": (xmax - xmin) * (ymax - ymin),
                }
            )
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
        if current is None or row["bbox_area"] > current["bbox_area"]:
            best[key] = row
    counts["unique_box_candidates"] = len(best)
    return list(best.values()), dict(sorted(counts.items()))


def join_metadata(
    protocol: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_id[row["image_id"]].append(row)
    counts: Counter[str] = Counter()
    with open(protocol["sources"]["image_metadata"]["path"], newline="") as handle:
        for metadata in csv.DictReader(handle):
            rows = by_id.get(metadata["ImageID"])
            if not rows:
                continue
            counts["candidate_metadata_rows"] += 1
            if metadata.get("Subset") != "validation":
                counts["wrong_subset"] += 1
                continue
            if metadata.get("License") != protocol["required_image_license"]:
                counts["license_rejection"] += 1
                continue
            if not metadata.get("Author") or not metadata.get("OriginalLandingURL"):
                counts["missing_attribution"] += 1
                continue
            try:
                rotation = float(metadata.get("Rotation") or 0)
            except ValueError:
                counts["invalid_rotation"] += 1
                continue
            if rotation != 0:
                counts["nonzero_rotation"] += 1
                continue
            title = metadata.get("Title", "")
            if SECONDARY_MEDIA_PATTERN.search(title):
                counts["secondary_media_title"] += 1
                continue
            attribution = {
                "license": metadata["License"],
                "author": metadata["Author"],
                "author_profile_url": metadata.get("AuthorProfileURL", ""),
                "title": title,
                "original_url": metadata.get("OriginalURL", ""),
                "original_landing_url": metadata["OriginalLandingURL"],
                "original_md5_base64": metadata.get("OriginalMD5", ""),
            }
            for row in rows:
                if row["attribute_name"] == "Stand" and TRUNCATION_CONTEXT_PATTERN.search(title):
                    counts["standing_truncation_title"] += 1
                    continue
                row["attribution"] = attribution
                counts["metadata_gate_pass_annotations"] += 1
    eligible = [row for row in candidates if "attribution" in row]
    counts["metadata_gate_pass_images"] = len({row["image_id"] for row in eligible})
    return eligible, dict(sorted(counts.items()))


def assign_one_target_per_image(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)
    selected = []
    conflicts = 0
    for image_id, values in sorted(grouped.items()):
        if len(values) > 1:
            conflicts += 1
        selected.append(
            min(
                values,
                key=lambda row: v1.stable_hash(
                    seed,
                    "target_assignment",
                    image_id,
                    row["class_name"],
                    row["attribute_name"],
                    row["bbox_key"],
                ),
            )
        )
    return selected, conflicts


def select_for_download(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows, _ = assign_one_target_per_image(rows, int(protocol["seed"]))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["class_name"], row["attribute_name"])].append(row)
    expected = {(name, attribute) for name in TARGET_CLASSES for attribute in TARGET_ATTRIBUTES}
    if set(grouped) != expected:
        raise RuntimeError(f"missing confirmation strata: {sorted(expected - set(grouped))}")
    selected = []
    quotas = protocol["sampling"]["download_candidates_per_stratum"]
    for key in sorted(grouped):
        values = sorted(
            grouped[key],
            key=lambda row: v1.stable_hash(
                protocol["seed"], "download", *key, row["image_id"], row["bbox_key"]
            ),
        )
        quota = int(quotas[key[1]])
        if len(values) < quota:
            raise RuntimeError(f"insufficient candidates for {key}: {len(values)} < {quota}")
        for row in values[:quota]:
            selected.append(
                {
                    **row,
                    "selection_sha256": v1.stable_hash(
                        protocol["seed"], "download", *key, row["image_id"], row["bbox_key"]
                    ),
                }
            )
    if len({row["image_id"] for row in selected}) != len(selected):
        raise RuntimeError("one-image-one-target selection failed")
    return selected


def download_images(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_root = Path(protocol["outputs"]["image_directory"])
    image_root.mkdir(parents=True, exist_ok=True)
    admission = {
        "image_url_template": protocol["image_url_template"],
        "maximum_single_image_bytes": protocol["limits"]["maximum_single_image_bytes"],
    }
    fetched, failures = [], []
    with ThreadPoolExecutor(max_workers=int(protocol["limits"]["download_workers"])) as executor:
        futures = {
            executor.submit(v1.fetch_image, row, image_root, admission): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                fetched.append(future.result())
            except Exception as error:
                failures.append(
                    {"image_id": row["image_id"], "error": f"{type(error).__name__}: {error}"}
                )
    if sum(row["image_bytes"] for row in fetched) > int(protocol["limits"]["maximum_downloaded_image_bytes"]):
        raise RuntimeError("confirmation image byte cap exceeded")
    by_id = {row["image_id"]: row for row in fetched}
    fields = (
        "image_path",
        "image_url",
        "image_bytes",
        "image_sha256",
        "dhash64",
        "width",
        "height",
        "channel_stddev",
    )
    return (
        [
            {**row, **{field: by_id[row["image_id"]][field] for field in fields}}
            for row in rows
            if row["image_id"] in by_id
        ],
        sorted(failures, key=lambda row: row["image_id"]),
    )


def development_fingerprints(path: Path) -> tuple[set[str], set[str], list[int]]:
    ids, hashes, dhashes = set(), set(), []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        for side in ("image_a", "image_b"):
            image = pair[side]
            ids.add(image["image_id"])
            hashes.add(image["image_sha256"])
            dhashes.append(int(image["dhash64"], 16))
    return ids, hashes, dhashes


def reject_development_overlap(
    rows: list[dict[str, Any]], development_manifest: Path, max_hamming: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids, hashes, dhashes = development_fingerprints(development_manifest)
    accepted, rejected = [], []
    for row in rows:
        reason = None
        if row["image_id"] in ids:
            reason = "image_id"
        elif row["image_sha256"] in hashes:
            reason = "exact_sha256"
        else:
            current = int(row["dhash64"], 16)
            distance = min(bin(current ^ prior).count("1") for prior in dhashes)
            if distance <= max_hamming:
                reason = f"development_dhash_hamming_{distance}"
        if reason is None:
            accepted.append(row)
        else:
            rejected.append({"image_id": row["image_id"], "reason": reason})
    return accepted, rejected


def choose_review_pool(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    deduplicated, duplicate_rejections = v1.deduplicate_images(rows)
    development_path = Path(protocol["prerequisites"]["development_manifest"]["path"])
    disjoint, development_rejections = reject_development_overlap(
        deduplicated,
        development_path,
        int(protocol["deduplication"]["development_dhash_max_hamming"]),
    )
    standing_rows = [row for row in disjoint if row["attribute_name"] == "Stand"]
    sitting_rows = [row for row in disjoint if row["attribute_name"] == "Sit"]
    ranked_standing = standing_v3.add_pose_ranking_features(protocol, standing_rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sitting_rows + ranked_standing:
        grouped[(row["class_name"], row["attribute_name"])].append(row)
    selected = []
    quotas = protocol["sampling"]["review_candidates_per_stratum"]
    for key in sorted(grouped):
        if key[1] == "Stand":
            values = sorted(
                grouped[key],
                key=lambda row: (
                    -row["curation_pose_features"]["mean_lower_body_score"],
                    v1.stable_hash(protocol["seed"], "review_tie", *key, row["image_id"]),
                ),
            )
        else:
            values = sorted(
                grouped[key],
                key=lambda row: v1.stable_hash(
                    protocol["seed"], "review", *key, row["image_id"]
                ),
            )
        quota = int(quotas[key[1]])
        if len(values) < quota:
            raise RuntimeError(f"post-dedup review gate failed for {key}: {len(values)} < {quota}")
        selected.extend(values[:quota])
    return (
        sorted(selected, key=lambda row: (row["class_name"], row["attribute_name"], row["image_id"])),
        duplicate_rejections,
        development_rejections,
    )


def render_review_sheets(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    directory = Path(protocol["outputs"]["audit_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    per_page = int(protocol["human_audit"]["images_per_page"])
    page_records = []
    for page_number, start in enumerate(range(0, len(rows), per_page), 1):
        page_rows = rows[start : start + per_page]
        sheet = Image.new("RGB", (1440, 320 * len(page_rows)), (228, 232, 238))
        for index, row in enumerate(page_rows):
            tile = panel(row, f"{row['class_name']} / {row['answer']}", (1440, 300))
            sheet.paste(tile, (0, index * 320))
            ImageDraw.Draw(sheet).text(
                (6, index * 320 + 302),
                f"{row['image_id']}  {row['selection_sha256'][:16]}",
                fill="black",
                font=ImageFont.load_default(),
            )
        output = directory / f"openimages-posture-confirmation-sheet-{page_number:02d}.png"
        sheet.save(output, optimize=True)
        page_records.append(
            {"path": str(output), "sha256": v1.sha256_file(output), "images": len(page_rows)}
        )
    return page_records


def build(protocol_path: Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    root = Path(protocol["outputs"]["root"])
    protected_outputs = (
        root,
        Path(protocol["outputs"]["audit_directory"]),
        Path(protocol["outputs"]["receipt"]),
    )
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"confirmation output already exists: {existing}")
    free_before = shutil.disk_usage(root.parent).free
    if free_before < int(protocol["limits"]["minimum_free_bytes_before"]):
        raise RuntimeError(f"insufficient free disk: {free_before}")
    root.mkdir(parents=True)

    candidates, scan_counts = scan_vrd(protocol)
    eligible, metadata_counts = join_metadata(protocol, candidates)
    assigned, multi_target_images = assign_one_target_per_image(eligible, int(protocol["seed"]))
    requested = select_for_download(protocol, assigned)
    downloaded, failures = download_images(protocol, requested)
    review_rows, duplicate_rejections, development_rejections = choose_review_pool(
        protocol, downloaded
    )

    selected_ids = {row["image_id"] for row in review_rows}
    removed_files, removed_bytes = 0, 0
    image_directory = Path(protocol["outputs"]["image_directory"])
    for path in image_directory.glob("*.jpg"):
        if path.stem not in selected_ids:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    manifest = Path(protocol["outputs"]["candidate_manifest"])
    manifest_hash = write_jsonl_atomic(manifest, review_rows)
    sheets = render_review_sheets(protocol, review_rows)
    strata = Counter((row["class_name"], row["attribute_name"]) for row in review_rows)

    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_pending_exhaustive_human_target_review",
        "training_authorized": False,
        "model_evaluation_authorized": False,
        "pairing_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": v1.sha256_file(protocol_path)},
        "builder_sha256": v1.sha256_file(Path(__file__)),
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(root).free,
        "scan_counts": scan_counts,
        "metadata_counts": metadata_counts,
        "eligible_after_one_target_per_image": len(assigned),
        "multi_target_images_resolved_deterministically": multi_target_images,
        "download_candidates": len(requested),
        "download_successes": len(downloaded),
        "download_failures": failures,
        "within_confirmation_duplicate_rejections": duplicate_rejections,
        "development_overlap_rejections": development_rejections,
        "development_split_image_id_overlap": 0,
        "removed_unselected_files": removed_files,
        "removed_unselected_bytes": removed_bytes,
        "review_images": len(review_rows),
        "review_by_stratum": [
            {"class_name": key[0], "attribute_name": key[1], "images": value}
            for key, value in sorted(strata.items())
        ],
        "pose_ranking": {
            "applied_to": "standing review-pool ordering only",
            "model_revision": protocol["pose_model"]["revision"],
            "acceptance_gate": False,
            "visible_to_human_reviewer": False,
        },
        "candidate_manifest": {
            "path": str(manifest),
            "sha256": manifest_hash,
            "rows": len(review_rows),
        },
        "audit_sheets": sheets,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(Path(protocol["outputs"]["receipt"]), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    receipt = build(args.protocol)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "review_images": receipt["review_images"],
                "audit_sheets": len(receipt["audit_sheets"]),
                "download_failures": len(receipt["download_failures"]),
                "development_overlap_rejections": len(receipt["development_overlap_rejections"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
