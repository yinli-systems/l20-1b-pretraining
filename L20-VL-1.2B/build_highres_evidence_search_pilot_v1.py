#!/usr/bin/env python3
"""Build a deterministic high-resolution POINT-or-STOP mechanism corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont


STATUS = "authorized_highres_evidence_search_pilot_generation_only_v1"
COLORS = {
    "red": (204, 54, 57),
    "blue": (43, 111, 191),
    "green": (54, 145, 85),
    "orange": (221, 127, 42),
}
BORDER_COLORS = {
    "purple": (112, 76, 182),
    "teal": (35, 137, 143),
    "gold": (190, 139, 37),
    "navy": (39, 63, 112),
}
SHAPES = ("circle", "square", "triangle", "diamond")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center_patch_index(box: tuple[int, int, int, int], image_size: int, grid_size: int) -> int:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= image_size and 0 <= y1 < y2 <= image_size):
        raise ValueError("invalid target box")
    cx = min(grid_size - 1, int(((x1 + x2) / 2) / image_size * grid_size))
    cy = min(grid_size - 1, int(((y1 + y2) / 2) / image_size * grid_size))
    return cy * grid_size + cx


def normalized_box(box: tuple[int, int, int, int], image_size: int) -> list[float]:
    return [round(value / image_size, 8) for value in box]


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    if shape == "circle":
        draw.ellipse(box, fill=color, outline=(25, 25, 28), width=3)
    elif shape == "square":
        draw.rectangle(box, fill=color, outline=(25, 25, 28), width=3)
    elif shape == "triangle":
        x1, y1, x2, y2 = box
        draw.polygon(((x1 + x2) // 2, y1, x2, y2, x1, y2), fill=color, outline=(25, 25, 28))
    elif shape == "diamond":
        x1, y1, x2, y2 = box
        draw.polygon((((x1 + x2) // 2, y1), (x2, (y1 + y2) // 2), ((x1 + x2) // 2, y2), (x1, (y1 + y2) // 2)), fill=color, outline=(25, 25, 28))
    else:
        raise ValueError(f"unknown shape: {shape}")


def render_scene(
    output: Path,
    image_size: int,
    panel_font: ImageFont.FreeTypeFont,
    digit_font: ImageFont.FreeTypeFont,
    assignments: list[tuple[str, str]],
    digits: list[int],
    border_rgb: tuple[int, int, int],
) -> list[tuple[int, int, int, int]]:
    canvas = Image.new("RGB", (image_size, image_size), (245, 244, 240))
    draw = ImageDraw.Draw(canvas)
    border_width = 24
    draw.rectangle((12, 12, image_size - 13, image_size - 13), outline=border_rgb, width=border_width)
    margin, gap, grid = 64, 12, 4
    cell = (image_size - 2 * margin - (grid - 1) * gap) // grid
    boxes: list[tuple[int, int, int, int]] = []
    for index, ((color_name, shape), digit) in enumerate(zip(assignments, digits)):
        row, column = divmod(index, grid)
        x1 = margin + column * (cell + gap)
        y1 = margin + row * (cell + gap)
        x2, y2 = x1 + cell, y1 + cell
        boxes.append((x1, y1, x2, y2))
        fill = (255, 255, 253) if (row + column) % 2 == 0 else (235, 238, 241)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=fill, outline=(135, 139, 145), width=2)
        shape_box = (x1 + 28, y1 + 62, x1 + 104, y1 + 138)
        draw_shape(draw, shape, shape_box, COLORS[color_name])
        draw.text((x1 + 18, y1 + 18), f"{color_name} {shape}", fill=(55, 56, 60), font=panel_font)
        # The digit is intentionally tiny at full resolution: it is typically
        # unreadable after 1024->224 global resizing but visible in a cell crop.
        draw.text((x1 + 153, y1 + 91), str(digit), fill=(15, 15, 18), font=digit_font, anchor="mm")
        draw.line((x1 + 119, y1 + 107, x1 + 138, y1 + 107), fill=(120, 122, 128), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", compress_level=9, optimize=False)
    return boxes


def build_family(
    family_index: int,
    split: str,
    task: str,
    seed: int,
    output_dir: Path,
    image_size: int,
    grid_size: int,
    panel_font: ImageFont.FreeTypeFont,
    digit_font: ImageFont.FreeTypeFont,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + family_index * 104729)
    assignments = [(color, shape) for color in COLORS for shape in SHAPES]
    rng.shuffle(assignments)
    digits = [rng.randrange(10) for _ in assignments]
    target_index = rng.randrange(len(assignments))
    target_color, target_shape = assignments[target_index]
    swap_index = (target_index + rng.randrange(1, len(assignments))) % len(assignments)
    while digits[swap_index] == digits[target_index]:
        digits[swap_index] = (digits[swap_index] + 1) % 10
    border_names = list(BORDER_COLORS)
    rng.shuffle(border_names)
    variants: list[tuple[str, list[int], str]] = []
    if task == "read_local_digit":
        edited = list(digits)
        edited[target_index], edited[swap_index] = edited[swap_index], edited[target_index]
        variants = [("base", digits, border_names[0]), ("answer_change", edited, border_names[0])]
    elif task == "read_global_border":
        variants = [("base", digits, border_names[0]), ("answer_change", digits, border_names[1])]
    else:
        raise ValueError(f"unknown task: {task}")

    family_id = f"hes-{split}-{family_index:06d}"
    rows: list[dict[str, Any]] = []
    for variant, variant_digits, border_name in variants:
        relative = Path("images") / split / f"{family_id}-{variant}.png"
        path = output_dir / relative
        boxes = render_scene(
            path,
            image_size,
            panel_font,
            digit_font,
            assignments,
            variant_digits,
            BORDER_COLORS[border_name],
        )
        if task == "read_local_digit":
            target_box = boxes[target_index]
            question = f"What digit is printed beside the {target_color} {target_shape}? Answer with one digit."
            answer = str(variant_digits[target_index])
            expected_action = "POINT"
            patch_index = center_patch_index(target_box, image_size, grid_size)
            crop_box = normalized_box(target_box, image_size)
        else:
            question = "What color is the thick outer border? Answer with one color word."
            answer = border_name
            expected_action = "STOP"
            patch_index = None
            crop_box = None
        rows.append({
            "schema_version": "2026-09-14-v1",
            "family_id": family_id,
            "split": split,
            "task": task,
            "variant": variant,
            "question": question,
            "answer": answer,
            "expected_action": expected_action,
            "target_patch_index_14x14": patch_index,
            "target_crop_box_xyxy_normalized": crop_box,
            "image_path": str(relative),
            "image_sha256": sha256_file(path),
            "image_bytes": path.stat().st_size,
            "render_state": {
                "assignments_in_cell_order": [f"{color} {shape}" for color, shape in assignments],
                "digits_in_cell_order": variant_digits,
                "target_cell_index": target_index if task == "read_local_digit" else None,
                "counterfactual_swap_cell_index": swap_index if task == "read_local_digit" else None,
                "border_color": border_name,
            },
        })
    return rows


def make_audit_sheets(rows: list[dict[str, Any]], root: Path, font_path: Path) -> list[Path]:
    selected: list[dict[str, Any]] = []
    for task in ("read_local_digit", "read_global_border"):
        for variant in ("base", "answer_change"):
            selected.extend([row for row in rows if row["task"] == task and row["variant"] == variant][:12])
    font = ImageFont.truetype(str(font_path), 16)
    small = ImageFont.truetype(str(font_path), 13)
    sheets: list[Path] = []
    for sheet_index in range(4):
        chunk = selected[sheet_index * 12:(sheet_index + 1) * 12]
        sheet = Image.new("RGB", (1860, 1720), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(chunk):
            rr, cc = divmod(offset, 3)
            x, y = cc * 620, rr * 430
            image = Image.open(root / row["image_path"]).convert("RGB")
            global_view = image.resize((270, 270), Image.Resampling.LANCZOS)
            sheet.paste(global_view, (x + 15, y + 15))
            crop_box = row["target_crop_box_xyxy_normalized"]
            if crop_box is not None:
                px = tuple(round(value * image.width) for value in crop_box)
                crop = image.crop(px).resize((270, 270), Image.Resampling.LANCZOS)
                sheet.paste(crop, (x + 300, y + 15))
            else:
                draw.rectangle((x + 300, y + 15, x + 570, y + 285), fill=(241, 242, 244))
                draw.text((x + 385, y + 135), "STOP", fill=(70, 70, 75), font=font)
            draw.text((x + 15, y + 300), f"{row['family_id']} | {row['variant']} | {row['expected_action']}", fill=(25, 25, 28), font=small)
            draw.text((x + 15, y + 326), row["question"], fill=(25, 25, 28), font=small)
            draw.text((x + 15, y + 355), f"ANSWER: {row['answer']}", fill=(150, 35, 42), font=font)
        path = root / "audit" / f"highres-evidence-search-sheet-{sheet_index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, format="PNG", compress_level=9, optimize=False)
        sheets.append(path)
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()

    from train_stage_a_full_token import utc_now, write_json_atomic

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("high-resolution evidence-search generation is not authorized")
    if protocol.get("training_authorized") is not False or protocol.get("model_evaluation_authorized") is not False:
        raise RuntimeError("generation protocol cannot authorize training or model evaluation")
    root = Path(__file__).resolve().parent
    for path, key in ((Path(__file__), "builder_sha256"), (root / "test_build_highres_evidence_search_pilot_v1.py", "test_sha256")):
        if sha256_file(path) != protocol["source_code"][key]:
            raise RuntimeError(f"source hash mismatch: {path.name}")
    font_path = Path(protocol["font"]["path"])
    if sha256_file(font_path) != protocol["font"]["sha256"]:
        raise RuntimeError("font hash mismatch")
    output = Path(protocol["output"]["root"])
    if output.exists():
        raise FileExistsError(output)
    receipt_path = Path(protocol["output"]["receipt"])
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    output.mkdir(parents=True)

    image_size = int(protocol["render"]["image_size"])
    panel_font = ImageFont.truetype(str(font_path), int(protocol["render"]["panel_font_pixels"]))
    digit_font = ImageFont.truetype(str(font_path), int(protocol["render"]["digit_font_pixels"]))
    rows: list[dict[str, Any]] = []
    family_index = 0
    for split, family_count in protocol["splits"].items():
        for split_index in range(int(family_count)):
            task = "read_local_digit" if split_index % 2 == 0 else "read_global_border"
            rows.extend(build_family(
                family_index,
                split,
                task,
                int(protocol["seed"]),
                output,
                image_size,
                int(protocol["render"]["pointer_grid_size"]),
                panel_font,
                digit_font,
            ))
            family_index += 1

    hashes = [row["image_sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("exact image collision")
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    sheets = make_audit_sheets(rows, output, font_path)
    counts = Counter((row["split"], row["task"], row["expected_action"]) for row in rows)
    expected_rows = 2 * sum(int(value) for value in protocol["splits"].values())
    if len(rows) != expected_rows:
        raise RuntimeError("unexpected row count")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_highres_evidence_search_pilot_generation",
        "completed_at": utc_now(),
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "builder_sha256": sha256_file(Path(__file__)),
        "font": protocol["font"],
        "families": family_index,
        "rows": len(rows),
        "unique_image_sha256": len(set(hashes)),
        "total_image_bytes": sum(row["image_bytes"] for row in rows),
        "counts": [
            {"split": key[0], "task": key[1], "expected_action": key[2], "rows": value}
            for key, value in sorted(counts.items())
        ],
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest), "bytes": manifest.stat().st_size},
        "audit_sheets": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sheets],
        "model_under_test_accessed": False,
        "training_started": False,
        "model_evaluation_started": False,
        "admission_decision": "pending_primary_visual_audit_and_preprocessing_floor_test",
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "families", "rows", "unique_image_sha256", "total_image_bytes", "admission_decision")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
