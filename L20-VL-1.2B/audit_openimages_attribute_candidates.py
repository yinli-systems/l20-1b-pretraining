#!/usr/bin/env python3
"""Audit real-image Open Images attribute candidates without creating training data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_MANIFEST_SHA256 = "33b88611299e095eaefebe0fe0a0f95f5ff9ab6d241c255d6747a240bc9ab264"
FAMILIES = {
    "material": {"Plastic", "(made of)Textile", "(made of)Leather", "Wooden"},
    "posture": {"Stand", "Sit", "Lay"},
    "locomotion": {"Walk", "Run", "Jump"},
    "expression": {"Smile", "Cry"},
    "vocal_action": {"Sing", "Talk"},
}
THRESHOLDS = (
    (0.06, 0.010),
    (0.08, 0.015),
    (0.10, 0.020),
    (0.12, 0.030),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_two_column_map(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    mapping = {row[0]: row[1] for row in rows if len(row) == 2}
    if len(mapping) != len(rows):
        raise RuntimeError(f"invalid or duplicate two-column rows in {path}")
    return mapping


def normalized_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    values = tuple(float(row[name]) for name in ("XMin1", "XMax1", "YMin1", "YMax1"))
    xmin, xmax, ymin, ymax = values
    if not (0.0 <= xmin < xmax <= 1.0 and 0.0 <= ymin < ymax <= 1.0):
        raise ValueError("invalid normalized box")
    return values


def box_key(box: Iterable[float]) -> tuple[str, ...]:
    return tuple(f"{value:.6f}" for value in box)


def family_for_attribute(name: str) -> str | None:
    matches = [family for family, values in FAMILIES.items() if name in values]
    if len(matches) > 1:
        raise RuntimeError(f"attribute appears in multiple families: {name}")
    return matches[0] if matches else None


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Stage-A Open Images manifest hash mismatch")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        image_id = row.get("image_id")
        if not image_id or image_id in rows:
            raise RuntimeError(f"manifest line {line_number}: duplicate/missing image_id")
        if row.get("split") not in {"train", "development"}:
            raise RuntimeError(f"manifest line {line_number}: invalid split")
        rows[image_id] = row
    if len(rows) != 25_000:
        raise RuntimeError(f"expected 25,000 selected images, got {len(rows)}")
    return rows


def threshold_summary(records: list[dict[str, Any]], min_edge: float, min_area: float) -> dict[str, Any]:
    selected = [
        row
        for row in records
        if row["box_width"] >= min_edge
        and row["box_height"] >= min_edge
        and row["box_area"] >= min_area
        and not row["family_conflict"]
    ]
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        cells[(row["split"], row["family"], row["class_name"])].append(row)
    pairable_cells = {
        key
        for key, values in cells.items()
        if len({row["attribute_name"] for row in values}) >= 2
    }
    same_attribute_counts = Counter(
        (row["split"], row["family"], row["class_name"], row["attribute_name"])
        for row in selected
    )
    pairable = [row for row in selected if (row["split"], row["family"], row["class_name"]) in pairable_cells]
    no_op_ready = [
        row
        for row in pairable
        if same_attribute_counts[(row["split"], row["family"], row["class_name"], row["attribute_name"])] >= 2
    ]
    cross_pairs = 0
    for values in cells.values():
        counts = Counter(row["attribute_name"] for row in values)
        names = sorted(counts)
        cross_pairs += sum(counts[a] * counts[b] for index, a in enumerate(names) for b in names[index + 1 :])
    pairable_cell_details = []
    for key in sorted(pairable_cells):
        values = cells[key]
        split, family, class_name = key
        pairable_cell_details.append({
            "split": split,
            "family": family,
            "class_name": class_name,
            "attribute_counts": dict(sorted(Counter(row["attribute_name"] for row in values).items())),
            "annotations": len(values),
            "images": len({row["image_id"] for row in values}),
        })
    return {
        "min_edge": min_edge,
        "min_area": min_area,
        "accepted_annotations": len(selected),
        "accepted_images": len({row["image_id"] for row in selected}),
        "family_conflict_rejections": sum(row["family_conflict"] for row in records),
        "pairable_annotations": len(pairable),
        "pairable_images": len({row["image_id"] for row in pairable}),
        "same_attribute_no_op_ready_annotations": len(no_op_ready),
        "cross_attribute_pair_count": cross_pairs,
        "pairable_class_family_cells": len(pairable_cells),
        "pairable_class_family_details": pairable_cell_details,
        "by_split": dict(sorted(Counter(row["split"] for row in pairable).items())),
        "by_family": dict(sorted(Counter(row["family"] for row in pairable).items())),
        "by_attribute": dict(sorted(Counter(row["attribute_name"] for row in pairable).items())),
        "top_pairable_classes": Counter(row["class_name"] for row in pairable).most_common(30),
    }


def audit(
    acquisition_receipt: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    receipt = json.loads(acquisition_receipt.read_text())
    if receipt.get("status") != "complete_verified_acquisition_only":
        raise RuntimeError("Open Images VRD acquisition is not verified")
    if receipt.get("training_started") is not False:
        raise RuntimeError("acquisition receipt unexpectedly reports training")
    sources = receipt["source_records"]
    for name, record in sources.items():
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"acquisition source mismatch: {name}")

    manifest = load_manifest(manifest_path)
    attributes = load_two_column_map(Path(sources["attribute_names"]["path"]))
    classes = load_two_column_map(Path(sources["boxable_class_names"]["path"]))
    expected_attributes = set().union(*FAMILIES.values())
    if set(attributes.values()) != expected_attributes | {"Transparent"}:
        raise RuntimeError("unexpected Open Images attribute vocabulary")

    raw: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    vrd_path = Path(sources["visual_relationships"]["path"])
    with vrd_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = {
            "ImageID", "LabelName1", "LabelName2", "XMin1", "XMax1", "YMin1", "YMax1",
            "XMin2", "XMax2", "YMin2", "YMax2", "RelationshipLabel",
        }
        if set(reader.fieldnames or ()) != expected_header:
            raise RuntimeError(f"unexpected VRD header: {reader.fieldnames}")
        for row in reader:
            counts["vrd_rows"] += 1
            image_id = row["ImageID"]
            if image_id not in manifest:
                continue
            counts["selected_image_vrd_rows"] += 1
            if row["RelationshipLabel"] != "is":
                continue
            counts["selected_image_is_rows"] += 1
            attribute_name = attributes.get(row["LabelName2"])
            family = None if attribute_name is None else family_for_attribute(attribute_name)
            if family is None:
                counts["excluded_attribute_rows"] += 1
                continue
            class_name = classes.get(row["LabelName1"])
            if class_name is None:
                counts["unknown_class_rows"] += 1
                continue
            try:
                box = normalized_bbox(row)
            except ValueError:
                counts["invalid_box_rows"] += 1
                continue
            object_box = box_key(box)
            attribute_box = tuple(row[name] for name in ("XMin2", "XMax2", "YMin2", "YMax2"))
            if object_box != tuple(f"{float(value):.6f}" for value in attribute_box):
                counts["object_attribute_box_mismatch_rows"] += 1
                continue
            xmin, xmax, ymin, ymax = box
            raw.append({
                "image_id": image_id,
                "split": manifest[image_id]["split"],
                "class_id": row["LabelName1"],
                "class_name": class_name,
                "attribute_id": row["LabelName2"],
                "attribute_name": attribute_name,
                "family": family,
                "bbox": list(box),
                "box_key": object_box,
                "box_width": xmax - xmin,
                "box_height": ymax - ymin,
                "box_area": (xmax - xmin) * (ymax - ymin),
            })

    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in raw:
        key = (row["image_id"], row["class_id"], row["attribute_id"], row["box_key"])
        dedup[key] = row
    counts["exact_duplicate_rows"] = len(raw) - len(dedup)
    records = list(dedup.values())
    conflict_keys: set[tuple[Any, ...]] = set()
    attributes_by_object: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in records:
        key = (row["image_id"], row["class_id"], row["box_key"], row["family"])
        attributes_by_object[key].add(row["attribute_id"])
    conflict_keys = {key for key, values in attributes_by_object.items() if len(values) > 1}
    for row in records:
        key = (row["image_id"], row["class_id"], row["box_key"], row["family"])
        row["family_conflict"] = key in conflict_keys

    result = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_candidate_audit_only",
        "training_authorized": False,
        "claim_boundary": (
            "Candidate statistics only. No examples were admitted for training; no natural-image "
            "transfer, causal mechanism, data quality, novelty, or model superiority is established."
        ),
        "acquisition_receipt": str(acquisition_receipt),
        "acquisition_receipt_sha256": sha256_file(acquisition_receipt),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": {name: record["sha256"] for name, record in sorted(sources.items())},
        "families": {name: sorted(values) for name, values in sorted(FAMILIES.items())},
        "transparent_excluded_from_contrastive_families": True,
        "raw_counts": dict(sorted(counts.items())),
        "deduplicated_candidate_annotations": len(records),
        "family_conflict_object_boxes": len(conflict_keys),
        "threshold_summaries": [threshold_summary(records, edge, area) for edge, area in THRESHOLDS],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.acquisition_receipt, args.manifest, args.output)
    print(json.dumps({
        "status": result["status"],
        "deduplicated_candidate_annotations": result["deduplicated_candidate_annotations"],
        "family_conflict_object_boxes": result["family_conflict_object_boxes"],
        "threshold_summaries": result["threshold_summaries"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
