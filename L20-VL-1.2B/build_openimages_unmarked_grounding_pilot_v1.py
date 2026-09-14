#!/usr/bin/env python3
"""Build an image-disjoint unmarked class-conditioned grounding pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_openimages_evidence_router_pilot_v1 import center_patch, expanded_xyxy
from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_openimages_unmarked_grounding_generation_only_v1"


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def bbox_tuple(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(round(float(row[key]), 6) for key in ("XMin1", "XMax1", "YMin1", "YMax1"))


def unique_class_boxes(vrd_path: Path, keys: set[tuple[str, str]]) -> dict[tuple[str, str], set[tuple[float, ...]]]:
    boxes: dict[tuple[str, str], set[tuple[float, ...]]] = defaultdict(set)
    with vrd_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["ImageID"], row["LabelName1"])
            if key in keys:
                boxes[key].add(bbox_tuple(row))
    return boxes


def build_rows(point_rows: list[dict[str, Any]], boxes, crop_expansion: float, grid: int):
    result = []
    for source in point_rows:
        key = (source["source_image_id"], source["class_id"])
        if len(boxes.get(key, set())) != 1:
            continue
        bbox = list(next(iter(boxes[key])))
        expected = [round(float(value), 6) for value in source["source_bbox_xmin_xmax_ymin_ymax"]]
        if bbox != expected:
            raise RuntimeError(f"selected bbox differs from sole official box: {key}")
        family_id = f"oi-ug-{source['split']}-{source['source_image_id']}"
        common = {
            "schema_version": "2026-09-14-v1",
            "family_id": family_id,
            "split": source["split"],
            "source": "open_images_v6_unique_subject_box_plus_localized_narratives",
            "source_image_id": source["source_image_id"],
            "class_id": source["class_id"],
            "class_name": source["class_name"],
            "source_bbox_xmin_xmax_ymin_ymax": bbox,
            "image_path": source["source_image_path"],
            "image_sha256": source["source_image_sha256"],
            "image_bytes": Path(source["source_image_path"]).stat().st_size,
            "license": source["license"],
            "attribution": source["attribution"],
        }
        result.append({
            **common,
            "variant": "point_unmarked_unique_class",
            "task": "locate_unique_object_category",
            "expected_action": "POINT",
            "question": f"Find the only {source['class_name'].lower()} in the image. Inspect it closely.",
            "answer": source["class_name"].lower(),
            "target_patch_index_14x14": center_patch(bbox, grid),
            "target_crop_box_xyxy_normalized": expanded_xyxy(bbox, crop_expansion),
        })
        result.append({
            **common,
            "variant": "stop_global",
            "task": "describe_global_scene",
            "expected_action": "STOP",
            "question": "Describe the overall image in one concise sentence.",
            "answer": source["global_answer"],
            "target_patch_index_14x14": None,
            "target_crop_box_xyxy_normalized": None,
        })
    result.sort(key=lambda row: (row["split"], row["family_id"], row["variant"]))
    return result


def render_audit_sheet(rows: list[dict[str, Any]], output: Path, font_path: Path) -> None:
    columns, lines = 3, 4
    cell_width, cell_height = 520, 480
    canvas = Image.new("RGB", (columns * cell_width, lines * cell_height), "white")
    font = ImageFont.truetype(str(font_path), 15)
    small = ImageFont.truetype(str(font_path), 12)
    for index, row in enumerate(rows):
        column, line = index % columns, index // columns
        left, top = column * cell_width, line * cell_height
        with Image.open(row["image_path"]) as handle:
            source = handle.convert("RGB")
        marked = source.copy()
        width, height = marked.size
        x0, x1, y0, y1 = row["source_bbox_xmin_xmax_ymin_ymax"]
        rectangle = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
        ImageDraw.Draw(marked).rectangle(rectangle, outline=(0, 220, 90), width=max(3, min(width, height) // 125))
        unmarked_view = ImageOps.contain(source, (235, 300), method=Image.Resampling.LANCZOS)
        audit_view = ImageOps.contain(marked, (235, 300), method=Image.Resampling.LANCZOS)
        canvas.paste(unmarked_view, (left + 10 + (235 - unmarked_view.width) // 2, top + 10))
        canvas.paste(audit_view, (left + 270 + (235 - audit_view.width) // 2, top + 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((left + 10, top + 320), "MODEL INPUT: unmarked", fill="black", font=font)
        draw.text((left + 270, top + 320), "AUDIT ONLY: official box", fill=(0, 130, 55), font=font)
        draw.text((left + 10, top + 350), f"{row['split']} | {row['class_name']} | patch {row['target_patch_index_14x14']}", fill="black", font=small)
        draw.text((left + 10, top + 375), row["source_image_id"], fill="black", font=small)
        draw.text((left + 10, top + 400), row["question"][:70], fill="black", font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS or protocol.get("training_authorized") is not False:
        raise SystemExit("generation-only unmarked-grounding protocol required")
    for path, key in (
        (Path(__file__), "builder_sha256"),
        (ROOT / "test_build_openimages_unmarked_grounding_pilot_v1.py", "test_sha256"),
        (ROOT / "build_openimages_evidence_router_pilot_v1.py", "marked_builder_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    for name, item in protocol["sources"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"source hash mismatch: {name}")
    font_path = Path(protocol["audit"]["font_path"])
    if sha256_file(font_path) != protocol["audit"]["font_sha256"]:
        raise SystemExit("font mismatch")
    output_root = Path(protocol["output"]["root"])
    receipt_path = Path(protocol["output"]["receipt"])
    if output_root.exists() or receipt_path.exists():
        raise FileExistsError("unmarked pilot output already exists")

    marked_manifest = Path(protocol["sources"]["marked_pilot_manifest"]["path"])
    all_marked_rows = [json.loads(line) for line in marked_manifest.read_text().splitlines() if line.strip()]
    global_answers = {
        row["family_id"]: row["answer"]
        for row in all_marked_rows
        if row["expected_action"] == "STOP"
    }
    point_rows = [
        {**row, "global_answer": global_answers[row["family_id"]]}
        for row in all_marked_rows
        if row["expected_action"] == "POINT"
    ]
    keys = {(row["source_image_id"], row["class_id"]) for row in point_rows}
    boxes = unique_class_boxes(Path(protocol["sources"]["vrd_annotations"]["path"]), keys)
    rows = build_rows(
        point_rows, boxes, float(protocol["selection"]["crop_expansion_fraction"]),
        int(protocol["selection"]["pointer_grid_size"]),
    )
    expected = protocol["selection"]["expected_unique_families_by_split"]
    for split, count in expected.items():
        actual = sum(row["split"] == split and row["expected_action"] == "POINT" for row in rows)
        if actual != int(count):
            raise RuntimeError(f"unexpected {split} unique-family count: {actual}")
    train_classes = {row["class_id"] for row in rows if row["split"] == "train"}
    development_classes = {row["class_id"] for row in rows if row["split"] == "development"}
    if not development_classes <= train_classes:
        raise RuntimeError("development contains unseen classes")
    image_ids = defaultdict(set)
    for row in rows:
        image_ids[row["split"]].add(row["source_image_id"])
        if sha256_file(Path(row["image_path"])) != row["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {row['source_image_id']}")
    if image_ids["train"] & image_ids["development"]:
        raise RuntimeError("train/development image overlap")

    output_root.mkdir(parents=True)
    manifest = output_root / "manifest.jsonl"
    with manifest.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit_sheets = []
    point_only = [row for row in rows if row["expected_action"] == "POINT"]
    per_sheet = int(protocol["audit"]["items_per_sheet"])
    sheets_per_split = int(protocol["audit"]["sheets_per_split"])
    for split in ("train", "development"):
        candidates = [row for row in point_only if row["split"] == split]
        candidates.sort(key=lambda row: stable_key(int(protocol["audit"]["seed"]), row["family_id"]))
        selected = candidates[:per_sheet * sheets_per_split]
        for sheet_index in range(sheets_per_split):
            sheet_rows = selected[sheet_index * per_sheet:(sheet_index + 1) * per_sheet]
            path = output_root / "audit" / f"openimages-unmarked-grounding-{split}-{sheet_index:02d}.png"
            render_audit_sheet(sheet_rows, path, font_path)
            audit_sheets.append({
                "split": split,
                "path": str(path),
                "sha256": sha256_file(path),
                "items": len(sheet_rows),
                "source_image_ids": [row["source_image_id"] for row in sheet_rows],
            })
    class_counts = {
        split: dict(sorted(Counter(
            row["class_name"] for row in point_only if row["split"] == split
        ).items()))
        for split in ("train", "development")
    }
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_openimages_unmarked_grounding_generation_v1",
        "completed_at": utc_now(),
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest), "rows": len(rows)},
        "families_by_split": {split: len(image_ids[split]) for split in image_ids},
        "class_counts": class_counts,
        "development_class_ids_subset_of_train": True,
        "train_development_image_overlap": 0,
        "model_input_has_rendered_box": False,
        "audit_sheets": audit_sheets,
        "training_started": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
