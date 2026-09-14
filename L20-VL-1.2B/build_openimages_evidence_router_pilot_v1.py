#!/usr/bin/env python3
"""Build a hash-pinned natural-image POINT/STOP router pilot from Open Images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


STATUS = "authorized_openimages_evidence_router_generation_only_v1"
ROOT = Path(__file__).resolve().parent


def stable_key(seed: int, *parts: object) -> str:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).hexdigest()


def expanded_xyxy(bbox: list[float], fraction: float) -> list[float]:
    x0, x1, y0, y1 = bbox
    pad_x = (x1 - x0) * fraction
    pad_y = (y1 - y0) * fraction
    return [max(0.0, x0 - pad_x), max(0.0, y0 - pad_y), min(1.0, x1 + pad_x), min(1.0, y1 + pad_y)]


def center_patch(bbox: list[float], grid: int) -> int:
    x0, x1, y0, y1 = bbox
    column = min(grid - 1, max(0, int(((x0 + x1) / 2.0) * grid)))
    row = min(grid - 1, max(0, int(((y0 + y1) / 2.0) * grid)))
    return row * grid + column


def render_marked(source: Path, bbox: list[float], output: Path, line_fraction: float, quality: int) -> None:
    with Image.open(source) as handle:
        image = handle.convert("RGB")
    width, height = image.size
    x0, x1, y0, y1 = bbox
    rectangle = (
        round(x0 * (width - 1)), round(y0 * (height - 1)),
        round(x1 * (width - 1)), round(y1 * (height - 1)),
    )
    line_width = max(3, round(min(width, height) * line_fraction))
    ImageDraw.Draw(image).rectangle(rectangle, outline=(255, 0, 0), width=line_width)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="JPEG", quality=quality, subsampling=0, optimize=True)


def source_crop(source: Path, box_xyxy: list[float]) -> Image.Image:
    with Image.open(source) as handle:
        image = handle.convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = box_xyxy
    crop = image.crop((round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)))
    return ImageOps.pad(crop, (420, 300), method=Image.Resampling.LANCZOS, color=(127, 127, 127))


def load_sources(protocol: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    source = protocol["sources"]
    images = {}
    with Path(source["openimages_manifest"]["path"]).open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                images[row["image_id"]] = row
    with Path(source["class_names"]["path"]).open(newline="") as handle:
        names = {row[0]: row[1] for row in csv.reader(handle)}
    return images, names


def candidate_targets(protocol: dict[str, Any], images: dict[str, dict[str, Any]], names: dict[str, str]):
    filters = protocol["filters"]
    grouped: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    with Path(protocol["sources"]["vrd_annotations"]["path"]).open(newline="") as handle:
        for row in csv.DictReader(handle):
            image = images.get(row["ImageID"])
            if image is None or row["LabelName1"] not in names:
                continue
            bbox = [float(row["XMin1"]), float(row["XMax1"]), float(row["YMin1"]), float(row["YMax1"])]
            width, height = bbox[1] - bbox[0], bbox[3] - bbox[2]
            area = width * height
            if (
                width < float(filters["minimum_bbox_width"])
                or height < float(filters["minimum_bbox_height"])
                or area < float(filters["minimum_bbox_area"])
                or area > float(filters["maximum_bbox_area"])
            ):
                continue
            key = (row["LabelName1"], *(round(value, 6) for value in bbox))
            grouped.setdefault(row["ImageID"], {})[key] = {
                "bbox": bbox,
                "bbox_area": area,
                "class_id": row["LabelName1"],
                "class_name": names[row["LabelName1"]],
                "relationship": row["RelationshipLabel"],
            }
    seed = int(protocol["seed"])
    selected = []
    preferred_area = float(filters["preferred_bbox_area"])
    for image_id, candidates in grouped.items():
        target = min(
            candidates.values(),
            key=lambda row: (
                abs(math.log(row["bbox_area"]) - math.log(preferred_area)),
                stable_key(seed, image_id, row["class_id"], row["bbox"]),
            ),
        )
        selected.append({"image": images[image_id], "target": target})
    result = []
    for split, cap in protocol["splits"].items():
        split_rows = [row for row in selected if row["image"]["split"] == split]
        split_rows.sort(key=lambda row: stable_key(seed, split, row["image"]["image_id"]))
        if len(split_rows) < int(cap):
            raise RuntimeError(f"insufficient {split} candidates: {len(split_rows)} < {cap}")
        result.extend(split_rows[:int(cap)])
    return result


def render_audit_sheet(rows: list[dict[str, Any]], output: Path, root: Path, font_path: Path) -> None:
    columns, rows_per_sheet = 3, 4
    cell_width, cell_height = 500, 510
    canvas = Image.new("RGB", (columns * cell_width, rows_per_sheet * cell_height), "white")
    font = ImageFont.truetype(str(font_path), 16)
    small = ImageFont.truetype(str(font_path), 13)
    for index, row in enumerate(rows):
        column, line = index % columns, index // columns
        left, top = column * cell_width, line * cell_height
        with Image.open(root / row["image_path"]) as handle:
            marked = ImageOps.contain(handle.convert("RGB"), (460, 320), method=Image.Resampling.LANCZOS)
        source = Path(row["source_image_path"])
        crop = ImageOps.contain(source_crop(source, row["target_crop_box_xyxy_normalized"]), (220, 135))
        canvas.paste(marked, (left + 20 + (460 - marked.width) // 2, top + 10))
        canvas.paste(crop, (left + 20, top + 335))
        draw = ImageDraw.Draw(canvas)
        draw.text((left + 255, top + 342), f"{row['split']} / {row['class_name']}", fill="black", font=font)
        draw.text((left + 255, top + 370), f"patch={row['target_patch_index_14x14']}", fill="black", font=small)
        draw.text((left + 255, top + 395), row["family_id"], fill="black", font=small)
        draw.text((left + 255, top + 420), "red box + source crop", fill="black", font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS or protocol.get("training_authorized") is not False:
        raise SystemExit("generation-only protocol required")
    if sha256_file(Path(__file__)) != protocol["source_code"]["builder_sha256"]:
        raise SystemExit("builder source hash mismatch")
    if sha256_file(ROOT / "test_build_openimages_evidence_router_pilot_v1.py") != protocol["source_code"]["test_sha256"]:
        raise SystemExit("builder test hash mismatch")
    for item in protocol["sources"].values():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    font = Path(protocol["render"]["font_path"])
    if sha256_file(font) != protocol["render"]["font_sha256"]:
        raise SystemExit("font hash mismatch")
    output_root = Path(protocol["output"]["root"])
    receipt_path = Path(protocol["output"]["receipt"])
    if output_root.exists() or receipt_path.exists():
        raise FileExistsError("pilot output already exists")

    images, names = load_sources(protocol)
    targets = candidate_targets(protocol, images, names)
    output_root.mkdir(parents=True)
    manifest_rows = []
    grid = int(protocol["render"]["pointer_grid_size"])
    expansion = float(protocol["render"]["crop_expansion_fraction"])
    for entry in targets:
        image, target = entry["image"], entry["target"]
        source = Path(image["image_path"])
        if sha256_file(source) != image["image_sha256"]:
            raise RuntimeError(f"source image mismatch: {image['image_id']}")
        split = image["split"]
        family_id = f"oi-er-{split}-{image['image_id']}"
        marked_relative = Path("marked") / split / f"{image['image_id']}.jpg"
        marked_path = output_root / marked_relative
        render_marked(
            source, target["bbox"], marked_path,
            float(protocol["render"]["box_line_fraction"]), int(protocol["render"]["jpeg_quality"]),
        )
        crop_box = expanded_xyxy(target["bbox"], expansion)
        common = {
            "schema_version": "2026-09-14-v1",
            "family_id": family_id,
            "split": split,
            "source": "open_images_v6_visual_relationships_plus_localized_narratives",
            "source_image_id": image["image_id"],
            "source_image_path": str(source),
            "source_image_sha256": image["image_sha256"],
            "license": image["attribution"]["license"],
            "attribution": image["attribution"],
            "class_id": target["class_id"],
            "class_name": target["class_name"],
            "source_bbox_xmin_xmax_ymin_ymax": target["bbox"],
            "target_crop_box_xyxy_normalized": crop_box,
            "target_patch_index_14x14": center_patch(target["bbox"], grid),
        }
        manifest_rows.append({
            **common,
            "variant": "point_marked",
            "task": "inspect_marked_object",
            "expected_action": "POINT",
            "image_path": str(marked_relative),
            "image_sha256": sha256_file(marked_path),
            "image_bytes": marked_path.stat().st_size,
            "question": "Inspect the object inside the red box. What object category is it? Answer briefly.",
            "answer": target["class_name"].lower(),
        })
        manifest_rows.append({
            **common,
            "variant": "stop_global",
            "task": "describe_global_scene",
            "expected_action": "STOP",
            "image_path": str(source),
            "image_sha256": image["image_sha256"],
            "image_bytes": image["image_bytes"],
            "question": "Describe the overall image in one concise sentence.",
            "answer": image["response"],
            "target_crop_box_xyxy_normalized": None,
            "target_patch_index_14x14": None,
        })
    manifest_rows.sort(key=lambda row: (row["split"], row["family_id"], row["variant"]))
    manifest = output_root / "manifest.jsonl"
    with manifest.open("x") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    point_rows = [row for row in manifest_rows if row["expected_action"] == "POINT"]
    audit_records = []
    per_sheet = int(protocol["audit"]["items_per_sheet"])
    sheets_per_split = int(protocol["audit"]["sheets_per_split"])
    for split in protocol["splits"]:
        candidates = [row for row in point_rows if row["split"] == split]
        candidates.sort(key=lambda row: stable_key(int(protocol["audit"]["seed"]), row["family_id"]))
        chosen = candidates[:per_sheet * sheets_per_split]
        for sheet_index in range(sheets_per_split):
            sheet_rows = chosen[sheet_index * per_sheet:(sheet_index + 1) * per_sheet]
            path = output_root / "audit" / f"openimages-evidence-router-{split}-{sheet_index:02d}.png"
            render_audit_sheet(sheet_rows, path, output_root, font)
            audit_records.append({
                "split": split,
                "path": str(path),
                "sha256": sha256_file(path),
                "items": len(sheet_rows),
                "family_ids": [row["family_id"] for row in sheet_rows],
            })

    split_counts = {
        split: {
            "families": sum(row["split"] == split and row["expected_action"] == "POINT" for row in manifest_rows),
            "rows": sum(row["split"] == split for row in manifest_rows),
            "POINT": sum(row["split"] == split and row["expected_action"] == "POINT" for row in manifest_rows),
            "STOP": sum(row["split"] == split and row["expected_action"] == "STOP" for row in manifest_rows),
        }
        for split in protocol["splits"]
    }
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_openimages_evidence_router_generation",
        "completed_at": utc_now(),
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest), "rows": len(manifest_rows)},
        "split_counts": split_counts,
        "unique_source_images": len(targets),
        "marked_images": len(point_rows),
        "audit_sheets": audit_records,
        "training_started": False,
        "development_model_access": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
