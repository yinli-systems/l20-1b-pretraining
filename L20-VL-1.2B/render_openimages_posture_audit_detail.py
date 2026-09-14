#!/usr/bin/env python3
"""Render full-image plus target-crop pages for the frozen posture pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_box(image: Image.Image, bbox: list[float]) -> Image.Image:
    output = image.convert("RGB").copy()
    xmin, xmax, ymin, ymax = bbox
    box = (
        round(xmin * output.width), round(ymin * output.height),
        round(xmax * output.width), round(ymax * output.height),
    )
    draw = ImageDraw.Draw(output)
    width = max(3, round(min(output.size) / 100))
    for offset in range(width):
        draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=(255, 20, 20))
    return output


def crop_with_context(image: Image.Image, bbox: list[float], context: float = 0.12) -> Image.Image:
    xmin, xmax, ymin, ymax = bbox
    width, height = xmax - xmin, ymax - ymin
    expanded = (
        max(0.0, xmin - context * width),
        min(1.0, xmax + context * width),
        max(0.0, ymin - context * height),
        min(1.0, ymax + context * height),
    )
    pixels = (
        int(expanded[0] * image.width), int(expanded[2] * image.height),
        max(1, int(expanded[1] * image.width)), max(1, int(expanded[3] * image.height)),
    )
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        raise RuntimeError("invalid expanded crop")
    return image.convert("RGB").crop(pixels)


def panel(record: dict[str, Any], label: str, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    with Image.open(record["image_path"]) as source:
        source.load()
        full = ImageOps.contain(draw_box(source, record["bbox"]), (size[0] // 2 - 12, size[1] - 36))
        crop = ImageOps.contain(crop_with_context(source, record["bbox"]), (size[0] // 2 - 12, size[1] - 36))
    result.paste(full, ((size[0] // 2 - full.width) // 2, 28 + (size[1] - 36 - full.height) // 2))
    crop_x = size[0] // 2 + (size[0] // 2 - crop.width) // 2
    result.paste(crop, (crop_x, 28 + (size[1] - 36 - crop.height) // 2))
    draw.text((6, 5), f"{label}: {record['class_name']} / {record['answer']}  full", fill="black", font=font)
    draw.text((size[0] // 2 + 6, 5), "target crop", fill="black", font=font)
    return result


def pair_tile(pair: dict[str, Any], size: tuple[int, int] = (1440, 410)) -> Image.Image:
    tile = Image.new("RGB", size, (238, 241, 246))
    half = size[0] // 2
    tile.paste(panel(pair["image_a"], "A", (half, size[1] - 24)), (0, 0))
    tile.paste(panel(pair["image_b"], "B", (half, size[1] - 24)), (half, 0))
    ImageDraw.Draw(tile).text((6, size[1] - 18), pair["pair_id"], fill="black", font=ImageFont.load_default())
    return tile


def render(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_posture_detail_audit_render_only":
        raise RuntimeError("detail audit render protocol is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("detail audit render must not authorize training")
    if sha256_file(Path(__file__)) != protocol["renderer_sha256"]:
        raise RuntimeError("detail renderer hash mismatch")
    if sha256_file(Path(__file__).with_name("test_render_openimages_posture_audit_detail.py")) != protocol["renderer_test_sha256"]:
        raise RuntimeError("detail renderer test hash mismatch")
    manifest = Path(protocol["pair_manifest"]["path"])
    if sha256_file(manifest) != protocol["pair_manifest"]["sha256"]:
        raise RuntimeError("posture pair manifest hash mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if len(rows) != protocol["pair_manifest"]["rows"] or len({row["pair_id"] for row in rows}) != len(rows):
        raise RuntimeError("posture pair manifest row identity mismatch")
    directory = Path(protocol["output_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    per_page = protocol["pairs_per_page"]
    records = []
    for page, start in enumerate(range(0, len(rows), per_page), 1):
        page_rows = rows[start : start + per_page]
        sheet = Image.new("RGB", (1440, len(page_rows) * 410), (228, 232, 238))
        for index, row in enumerate(page_rows):
            sheet.paste(pair_tile(row), (0, index * 410))
        path = directory / f"openimages-posture-detail-sheet-{page:02d}.png"
        sheet.save(path, optimize=True)
        records.append({"path": str(path), "sha256": sha256_file(path), "pairs": len(page_rows)})
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_detail_audit_render_only",
        "training_authorized": False,
        "protocol_sha256": sha256_file(protocol_path),
        "renderer_sha256": sha256_file(Path(__file__)),
        "pair_manifest_sha256": sha256_file(manifest),
        "pairs_rendered": len(rows),
        "pages": records,
        "claim_boundary": "Rendering improves human audit visibility only; it does not alter labels, admit data, authorize training, or establish model quality.",
    }
    receipt_path = Path(protocol["receipt"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    receipt = render(args.protocol)
    print(json.dumps({"status": receipt["status"], "pairs": receipt["pairs_rendered"], "pages": len(receipt["pages"])}, sort_keys=True))


if __name__ == "__main__":
    main()
