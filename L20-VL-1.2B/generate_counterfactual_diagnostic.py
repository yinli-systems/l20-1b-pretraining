#!/usr/bin/env python3
"""Generate a deterministic paired-scene counterfactual diagnostic corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
TASKS = (
    "count",
    "color_binding",
    "shape_binding",
    "left_right_relation",
    "above_below_relation",
)
COLORS = {
    "red": (210, 54, 65),
    "blue": (47, 105, 190),
    "green": (55, 142, 82),
    "yellow": (224, 179, 38),
    "purple": (129, 82, 170),
    "orange": (224, 119, 45),
}
BACKGROUNDS = {
    "warm_white": (247, 244, 237),
    "cool_white": (238, 244, 248),
    "light_gray": (239, 239, 239),
    "pale_mint": (237, 247, 241),
}
SHAPES = ("circle", "square", "triangle")
POSITIONS = (
    (48, 48), (128, 48), (208, 48),
    (48, 128), (128, 128), (208, 128),
    (48, 208), (128, 208), (208, 208),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def plural(shape: str) -> str:
    return shape + "s"


def object_record(identifier: str, color: str, shape: str, position: tuple[int, int], size: int = 25) -> dict:
    return {
        "id": identifier,
        "color": color,
        "shape": shape,
        "x": position[0],
        "y": position[1],
        "size": size,
    }


def add_distractors(objects: list[dict], rng: random.Random, count: int = 3) -> None:
    used_positions = {(item["x"], item["y"]) for item in objects}
    free = [position for position in POSITIONS if position not in used_positions]
    rng.shuffle(free)
    existing = {(item["color"], item["shape"]) for item in objects}
    for index in range(count):
        for _ in range(30):
            color = rng.choice(tuple(COLORS))
            shape = rng.choice(SHAPES)
            if (color, shape) not in existing:
                break
        existing.add((color, shape))
        objects.append(object_record(f"d{index}", color, shape, free[index]))


def add_distractors_at_positions(
    objects: list[dict], rng: random.Random, positions: tuple[tuple[int, int], ...]
) -> None:
    existing = {(item["color"], item["shape"]) for item in objects}
    for index, position in enumerate(positions):
        for _ in range(30):
            color = rng.choice(tuple(COLORS))
            shape = rng.choice(SHAPES)
            if (color, shape) not in existing:
                break
        existing.add((color, shape))
        objects.append(object_record(f"d{index}", color, shape, position))


def build_scene(task: str, base_answer: str, seed: int) -> dict:
    rng = random.Random(seed)
    background, invariant_background = rng.sample(tuple(BACKGROUNDS), 2)
    base = {"background": background, "objects": []}
    relevant = None

    if task in {"left_right_relation", "above_below_relation"}:
        color_a, color_b = rng.sample(tuple(COLORS), 2)
        shape_a, shape_b = rng.sample(SHAPES, 2)
        if task == "left_right_relation":
            y_a, y_b = rng.sample((88, 168), 2)
            left_x, right_x = 72, 184
            a_left = base_answer == "yes"
            pos_a = (left_x if a_left else right_x, y_a)
            pos_b = (right_x if a_left else left_x, y_b)
            relation = "left of"
            edit_factor = "swap_x_positions"
            distractor_positions = ((128, 40), (128, 216))
        else:
            x_a, x_b = rng.sample((88, 168), 2)
            top_y, bottom_y = 72, 184
            a_above = base_answer == "yes"
            pos_a = (x_a, top_y if a_above else bottom_y)
            pos_b = (x_b, bottom_y if a_above else top_y)
            relation = "above"
            edit_factor = "swap_y_positions"
            distractor_positions = ((40, 128), (216, 128))
        base["objects"] = [
            object_record("target_a", color_a, shape_a, pos_a),
            object_record("target_b", color_b, shape_b, pos_b),
        ]
        add_distractors_at_positions(base["objects"], rng, distractor_positions)
        question = f"Is the {color_a} {shape_a} {relation} the {color_b} {shape_b}?"
        oracle_query = {"subject_id": "target_a", "object_id": "target_b", "relation": relation}
        edited = deepcopy(base)
        a = next(item for item in edited["objects"] if item["id"] == "target_a")
        b = next(item for item in edited["objects"] if item["id"] == "target_b")
        if task == "left_right_relation":
            a["x"], b["x"] = b["x"], a["x"]
        else:
            a["y"], b["y"] = b["y"], a["y"]
        relevant = {"factor": edit_factor, "target_ids": ["target_a", "target_b"]}

    elif task == "color_binding":
        target_shape = rng.choice(SHAPES)
        query_color, alternative = rng.sample(tuple(COLORS), 2)
        target_color = query_color if base_answer == "yes" else alternative
        base["objects"] = [object_record("target", target_color, target_shape, rng.choice(POSITIONS))]
        add_distractors(base["objects"], rng, 3)
        for item in base["objects"][1:]:
            if item["shape"] == target_shape:
                item["shape"] = next(shape for shape in SHAPES if shape != target_shape)
        question = f"Is the {target_shape} {query_color}?"
        oracle_query = {"target_id": "target", "color": query_color}
        edited = deepcopy(base)
        target = next(item for item in edited["objects"] if item["id"] == "target")
        target["color"] = alternative if base_answer == "yes" else query_color
        relevant = {
            "factor": "target_color",
            "target_ids": ["target"],
            "before": target_color,
            "after": target["color"],
        }

    elif task == "shape_binding":
        target_color = rng.choice(tuple(COLORS))
        query_shape, alternative = rng.sample(SHAPES, 2)
        target_shape = query_shape if base_answer == "yes" else alternative
        base["objects"] = [object_record("target", target_color, target_shape, rng.choice(POSITIONS))]
        add_distractors(base["objects"], rng, 3)
        for item in base["objects"][1:]:
            if item["color"] == target_color:
                item["color"] = next(color for color in COLORS if color != target_color)
        question = f"Is the {target_color} object a {query_shape}?"
        oracle_query = {"target_id": "target", "shape": query_shape}
        edited = deepcopy(base)
        target = next(item for item in edited["objects"] if item["id"] == "target")
        target["shape"] = alternative if base_answer == "yes" else query_shape
        relevant = {
            "factor": "target_shape",
            "target_ids": ["target"],
            "before": target_shape,
            "after": target["shape"],
        }

    elif task == "count":
        target_color = rng.choice(tuple(COLORS))
        target_shape = rng.choice(SHAPES)
        queried_count = rng.choice((2, 3))
        if base_answer == "yes":
            actual_count = queried_count
        else:
            actual_count = queried_count - 1 if rng.random() < 0.5 else queried_count + 1
        free = list(POSITIONS)
        rng.shuffle(free)
        base["objects"] = [
            object_record(f"target_{index}", target_color, target_shape, free.pop())
            for index in range(actual_count)
        ]
        add_distractors(base["objects"], rng, min(3, len(free)))
        question = f"Are there exactly {queried_count} {target_color} {plural(target_shape)}?"
        oracle_query = {"color": target_color, "shape": target_shape, "count": queried_count}
        edited = deepcopy(base)
        if actual_count < queried_count:
            used = {(item["x"], item["y"]) for item in edited["objects"]}
            position = next(position for position in POSITIONS if position not in used)
            edited["objects"].append(
                object_record("target_added", target_color, target_shape, position)
            )
            operation = "add_matching_object"
        else:
            target = next(
                item for item in reversed(edited["objects"])
                if item["color"] == target_color and item["shape"] == target_shape
            )
            edited["objects"].remove(target)
            operation = "remove_matching_object"
        relevant = {
            "factor": "matching_object_count",
            "target_ids": [target_color + "_" + target_shape],
            "before": actual_count,
            "after": queried_count,
            "operation": operation,
        }
    else:
        raise ValueError(f"unknown task: {task}")

    invariant = deepcopy(base)
    invariant["background"] = invariant_background
    edited["background"] = background
    return {
        "question": question,
        "oracle_query": oracle_query,
        "base_state": base,
        "edited_state": edited,
        "invariant_state": invariant,
        "relevant_edit": {**relevant, "answer_must_change": True},
        "invariant_edit": {
            "factor": "background_palette",
            "before": background,
            "after": invariant_background,
            "answer_must_change": False,
        },
    }


def evaluate_scene(task: str, state: dict, query: dict) -> str:
    objects = {item["id"]: item for item in state["objects"]}
    if task == "left_right_relation":
        result = objects[query["subject_id"]]["x"] < objects[query["object_id"]]["x"]
    elif task == "above_below_relation":
        result = objects[query["subject_id"]]["y"] < objects[query["object_id"]]["y"]
    elif task == "color_binding":
        result = objects[query["target_id"]]["color"] == query["color"]
    elif task == "shape_binding":
        result = objects[query["target_id"]]["shape"] == query["shape"]
    elif task == "count":
        actual = sum(
            item["color"] == query["color"] and item["shape"] == query["shape"]
            for item in state["objects"]
        )
        result = actual == query["count"]
    else:
        raise ValueError(f"unknown task: {task}")
    return "yes" if result else "no"


def build_specs(pair_count: int, seed: int) -> list[dict]:
    strata = len(TASKS) * 2
    if pair_count % strata:
        raise ValueError(f"pair_count must be divisible by {strata}")
    per_stratum = pair_count // strata
    train_count = int(per_stratum * 0.70)
    development_count = int(per_stratum * 0.15)
    specs = []
    for task in TASKS:
        for answer in ("yes", "no"):
            candidates = []
            for index in range(per_stratum):
                scene_seed = stable_seed(seed, task, answer, index)
                family = hashlib.sha256(f"{task}|{answer}|{scene_seed}".encode()).hexdigest()[:24]
                candidates.append((stable_seed("split", family), index, scene_seed, family))
            candidates.sort()
            for rank, (_, index, scene_seed, family) in enumerate(candidates):
                split = (
                    "train" if rank < train_count
                    else "development" if rank < train_count + development_count
                    else "test"
                )
                specs.append(
                    {
                        "pair_id": f"cf-{task}-{answer}-{index:04d}",
                        "scene_family_id": family,
                        "split": split,
                        "task": task,
                        "base_answer": answer,
                        "scene_seed": scene_seed,
                    }
                )
    specs.sort(key=lambda item: item["pair_id"])
    return specs


def draw_shape(draw: ImageDraw.ImageDraw, item: dict, scale: int) -> None:
    x, y, size = item["x"] * scale, item["y"] * scale, item["size"] * scale
    box = (x - size, y - size, x + size, y + size)
    fill = COLORS[item["color"]]
    outline = (36, 39, 44)
    width = max(2, 2 * scale)
    if item["shape"] == "circle":
        draw.ellipse(box, fill=fill, outline=outline, width=width)
    elif item["shape"] == "square":
        draw.rounded_rectangle(box, radius=3 * scale, fill=fill, outline=outline, width=width)
    elif item["shape"] == "triangle":
        points = ((x, y - size), (x - size, y + size), (x + size, y + size))
        draw.polygon(points, fill=fill, outline=outline)
        draw.line(points + (points[0],), fill=outline, width=width, joint="curve")
    else:
        raise ValueError(f"unknown shape: {item['shape']}")


def render_scene(state: dict, output: Path, scale: int) -> None:
    size = 256
    image = Image.new("RGB", (size * scale, size * scale), BACKGROUNDS[state["background"]])
    draw = ImageDraw.Draw(image)
    for item in sorted(state["objects"], key=lambda value: value["id"]):
        draw_shape(draw, item, scale)
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output)


def contact_sheet(task: str, records: list[dict], output: Path) -> None:
    chosen = sorted(records, key=lambda item: stable_seed("sheet", item["pair_id"]))[:12]
    cell = 256
    label_height = 84
    sheet = Image.new("RGB", (cell * 3, len(chosen) * (cell + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for row_index, record in enumerate(chosen):
        top = row_index * (cell + label_height)
        for column, variant in enumerate(("base", "edited", "invariant")):
            with Image.open(record[f"{variant}_image_path"]) as image:
                sheet.paste(image.convert("RGB"), (column * cell, top))
            draw.rectangle(
                (column * cell, top, (column + 1) * cell - 1, top + cell - 1),
                outline=(120, 120, 120),
            )
        label = (
            f"{record['pair_id']} | {record['split']}\n"
            f"Q: {record['question']}\n"
            f"base={record['base_answer']}  edited={record['edited_answer']}  "
            f"invariant={record['invariant_answer']}"
        )
        draw.multiline_text((8, top + cell + 5), label, fill="black", font=font, spacing=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def generate(admission_path: Path, protocol_path: Path, output: Path, receipt_path: Path) -> dict:
    admission = json.loads(admission_path.read_text())
    protocol_hash = sha256_file(protocol_path)
    if admission.get("required_protocol_sha256") != protocol_hash:
        raise RuntimeError("protocol hash does not match diagnostic admission")
    if admission.get("generator_authorized") is not True:
        raise RuntimeError("generator is not authorized")
    if admission.get("external_downloads_authorized") is not False:
        raise RuntimeError("external downloads must remain disabled")
    if admission.get("formal_training_authorized") is not False:
        raise RuntimeError("formal training must remain blocked")
    excluded_image_hashes: set[str] = set()
    exclusion_manifest = admission.get("exclude_manifest")
    if exclusion_manifest is not None:
        exclusion_path = Path(exclusion_manifest["path"])
        if sha256_file(exclusion_path) != exclusion_manifest["sha256"]:
            raise RuntimeError("exclusion manifest hash mismatch")
        for line in exclusion_path.read_text().splitlines():
            if not line:
                continue
            record = json.loads(line)
            excluded_image_hashes.update(
                record[f"{variant}_image_sha256"]
                for variant in ("base", "edited", "invariant")
            )
    manifest_path = output / "manifest.jsonl"
    if output.exists() or receipt_path.exists():
        raise FileExistsError("diagnostic output or receipt already exists")
    output.mkdir(parents=True)

    specs = build_specs(admission["pair_count"], admission["seed"])
    records = []
    image_hashes = Counter()
    total_bytes = 0
    counts = Counter()
    collision_retries = 0
    for number, spec in enumerate(specs, 1):
        edited_answer = "no" if spec["base_answer"] == "yes" else "yes"
        for collision_attempt in range(101):
            actual_scene_seed = (
                spec["scene_seed"]
                if collision_attempt == 0
                else stable_seed(spec["scene_seed"], "collision-retry", collision_attempt)
            )
            scene = build_scene(spec["task"], spec["base_answer"], actual_scene_seed)
            oracle_answers = {
                variant: evaluate_scene(spec["task"], scene[f"{variant}_state"], scene["oracle_query"])
                for variant in ("base", "edited", "invariant")
            }
            expected_answers = {
                "base": spec["base_answer"],
                "edited": edited_answer,
                "invariant": spec["base_answer"],
            }
            if oracle_answers != expected_answers:
                raise RuntimeError(
                    f"independent oracle mismatch for {spec['pair_id']}: "
                    f"{oracle_answers} != {expected_answers}"
                )
            variant_paths = {}
            variant_hashes = {}
            variant_bytes = 0
            for variant in ("base", "edited", "invariant"):
                relative = Path("images") / spec["split"] / f"{spec['pair_id']}-{variant}.png"
                path = output / relative
                render_scene(scene[f"{variant}_state"], path, admission["render_scale"])
                digest = sha256_file(path)
                variant_bytes += path.stat().st_size
                variant_paths[variant] = str(path)
                variant_hashes[variant] = digest
            digests = list(variant_hashes.values())
            if len(set(digests)) == 3 and not any(
                digest in image_hashes or digest in excluded_image_hashes for digest in digests
            ):
                break
            collision_retries += 1
        else:
            raise RuntimeError(f"could not resolve exact image collision for {spec['pair_id']}")
        for digest in variant_hashes.values():
            image_hashes[digest] += 1
        total_bytes += variant_bytes
        record = {
            "pair_id": spec["pair_id"],
            "scene_family_id": spec["scene_family_id"],
            "split": spec["split"],
            "task": spec["task"],
            "question": scene["question"],
            "candidate_answers": ["yes", "no"],
            "base_answer": spec["base_answer"],
            "edited_answer": edited_answer,
            "invariant_answer": spec["base_answer"],
            "base_image_path": variant_paths["base"],
            "edited_image_path": variant_paths["edited"],
            "invariant_image_path": variant_paths["invariant"],
            "base_image_sha256": variant_hashes["base"],
            "edited_image_sha256": variant_hashes["edited"],
            "invariant_image_sha256": variant_hashes["invariant"],
            "relevant_edit": scene["relevant_edit"],
            "invariant_edit": scene["invariant_edit"],
            "scene_state": scene["base_state"],
            "oracle_query": scene["oracle_query"],
            "render_seeds": {
                "scene": actual_scene_seed,
                "base": stable_seed(actual_scene_seed, "base"),
                "edited": stable_seed(actual_scene_seed, "edited"),
                "invariant": stable_seed(actual_scene_seed, "invariant"),
            },
            "collision_retry_count": collision_attempt,
        }
        records.append(record)
        counts[(spec["task"], spec["base_answer"], spec["split"])] += 1
        if number % 250 == 0:
            print(f"generated {number}/{len(specs)} scene families", flush=True)

    if total_bytes > admission["max_total_bytes"]:
        raise RuntimeError("generated image corpus exceeds byte cap")
    duplicates = sum(value - 1 for value in image_hashes.values() if value > 1)
    if duplicates:
        raise RuntimeError(f"unexpected exact duplicate rendered images: {duplicates}")
    with manifest_path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    sheets = []
    for task in TASKS:
        path = ROOT / "evidence" / "human-audit" / f"{output.name}-{task}-sheet.png"
        contact_sheet(task, [record for record in records if record["task"] == task], path)
        sheets.append({"task": task, "path": str(path), "sha256": sha256_file(path)})
    split_counts = defaultdict(Counter)
    for record in records:
        split_counts[record["task"]][record["split"]] += 1
    receipt = {
        "schema_version": "2026-09-13-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_pending_human_audit",
        "formal_training": False,
        "training_prediction_tokens": 0,
        "protocol_sha256": protocol_hash,
        "admission_sha256": sha256_file(admission_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "output": str(output),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "scene_families": len(records),
        "rendered_images": len(records) * 3,
        "unique_image_sha256": len(image_hashes),
        "deterministic_collision_retries": collision_retries,
        "exclusion_manifest": exclusion_manifest,
        "excluded_image_hashes": len(excluded_image_hashes),
        "total_image_bytes": total_bytes,
        "tasks": list(TASKS),
        "per_task_split": {task: dict(sorted(value.items())) for task, value in sorted(split_counts.items())},
        "per_task_answer_split": {
            "|".join(key): value for key, value in sorted(counts.items())
        },
        "answer_invariants": {
            "edited_answer_differs_from_base": all(record["edited_answer"] != record["base_answer"] for record in records),
            "invariant_answer_equals_base": all(record["invariant_answer"] == record["base_answer"] for record in records),
            "variants_share_scene_family_and_split": True,
        },
        "contact_sheets": sheets,
        "claim_boundary": "Programmatic scenes test a controlled mechanism. They do not establish real-image generalization or authorize model training.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=ROOT / "counterfactual_data_admission.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "counterfactual_protocol.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path, default=ROOT / "evidence" / "counterfactual-data-generation.json")
    args = parser.parse_args()
    admission = json.loads(args.admission.read_text())
    output = args.output or Path(admission["destination"])
    generate(args.admission, args.protocol, output, args.receipt)


if __name__ == "__main__":
    main()
