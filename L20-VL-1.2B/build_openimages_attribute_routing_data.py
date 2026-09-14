#!/usr/bin/env python3
"""Build a bounded real-image region/attribute routing pilot and audit sheets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from audit_openimages_attribute_candidates import (
    box_key,
    family_for_attribute,
    load_manifest,
    load_two_column_map,
    normalized_bbox,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent


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


def display_attribute(name: str) -> str:
    mapping = {
        "(made of)Leather": "leather",
        "(made of)Textile": "textile",
        "Plastic": "plastic",
        "Wooden": "wooden",
        "Sit": "sitting",
        "Stand": "standing",
    }
    if name not in mapping:
        raise RuntimeError(f"attribute has no frozen answer form: {name}")
    return mapping[name]


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_real_image_attribute_data_build_only":
        raise RuntimeError("data-build protocol is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("data-build protocol must not authorize training")
    if protocol.get("confirmation_use_authorized") is not False:
        raise RuntimeError("sealed confirmation use must remain unauthorized")
    expected_builder = protocol["source_code"]["builder_sha256"]
    if sha256_file(Path(__file__)) != expected_builder:
        raise RuntimeError("data builder hash mismatch")
    expected_audit = protocol["source_code"]["candidate_audit_script_sha256"]
    if sha256_file(ROOT / "audit_openimages_attribute_candidates.py") != expected_audit:
        raise RuntimeError("candidate audit script hash mismatch")
    for name, record in protocol["evidence"].items():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen evidence hash mismatch: {name}")
    return protocol


def internal_split(row: dict[str, Any], policy: dict[str, Any]) -> str:
    if row["split"] == "development":
        return "sealed_confirmation"
    if row["split"] != "train":
        raise RuntimeError(f"unexpected parent split: {row['split']}")
    bucket = int(stable_hash(policy["seed"], row["image_id"]), 16) % policy["bucket_count"]
    return "selection_dev" if bucket in set(policy["selection_buckets"]) else "train"


def scan_candidates(protocol: dict[str, Any], manifest: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    acquisition = json.loads(Path(protocol["evidence"]["acquisition_receipt"]["path"]).read_text())
    if acquisition.get("status") != "complete_verified_acquisition_only":
        raise RuntimeError("acquisition receipt is not complete")
    sources = acquisition["source_records"]
    attributes = load_two_column_map(Path(sources["attribute_names"]["path"]))
    classes = load_two_column_map(Path(sources["boxable_class_names"]["path"]))
    allowed = {
        (cell["family"], cell["class_name"], attribute)
        for cell in protocol["allowed_contrasts"]
        for pair in cell["attribute_pairs"]
        for attribute in pair
    }
    filters = protocol["quality_filters"]
    counts: Counter[str] = Counter()
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    all_object_attributes: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    raw_rows: list[dict[str, Any]] = []
    vrd_path = Path(sources["visual_relationships"]["path"])
    with vrd_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "ImageID", "LabelName1", "LabelName2", "XMin1", "XMax1", "YMin1", "YMax1",
            "XMin2", "XMax2", "YMin2", "YMax2", "RelationshipLabel",
        }
        if set(reader.fieldnames or ()) != required:
            raise RuntimeError("unexpected Open Images VRD header")
        for row in reader:
            image_id = row["ImageID"]
            if image_id not in manifest or row["RelationshipLabel"] != "is":
                continue
            attribute_name = attributes.get(row["LabelName2"])
            class_name = classes.get(row["LabelName1"])
            if attribute_name is None or class_name is None:
                continue
            family = family_for_attribute(attribute_name)
            if (family, class_name, attribute_name) not in allowed:
                continue
            try:
                bbox = normalized_bbox(row)
            except ValueError:
                counts["invalid_box"] += 1
                continue
            bbox_strings = box_key(bbox)
            attribute_bbox = tuple(
                f"{float(row[name]):.6f}" for name in ("XMin2", "XMax2", "YMin2", "YMax2")
            )
            if bbox_strings != attribute_bbox:
                counts["object_attribute_box_mismatch"] += 1
                continue
            xmin, xmax, ymin, ymax = bbox
            width, height = xmax - xmin, ymax - ymin
            if width < filters["min_edge"] or height < filters["min_edge"] or width * height < filters["min_area"]:
                counts["small_box"] += 1
                continue
            object_key = (image_id, row["LabelName1"], bbox_strings, family)
            all_object_attributes[object_key].add(attribute_name)
            raw_rows.append({
                "image_id": image_id,
                "parent_split": manifest[image_id]["split"],
                "split": internal_split(manifest[image_id], protocol["split_policy"]),
                "class_id": row["LabelName1"],
                "class_name": class_name,
                "attribute_id": row["LabelName2"],
                "attribute_name": attribute_name,
                "answer": display_attribute(attribute_name),
                "family": family,
                "bbox": list(bbox),
                "bbox_key": bbox_strings,
                "bbox_area": width * height,
            })
    conflict_keys = {key for key, values in all_object_attributes.items() if len(values) > 1}
    counts["family_conflict_object_boxes"] = len(conflict_keys)
    for row in raw_rows:
        object_key = (row["image_id"], row["class_id"], row["bbox_key"], row["family"])
        if object_key in conflict_keys:
            counts["family_conflict_annotations"] += 1
            continue
        exact_key = (row["image_id"], row["class_id"], row["attribute_id"], row["bbox_key"])
        current = candidates.get(exact_key)
        if current is None or row["bbox_area"] > current["bbox_area"]:
            candidates[exact_key] = row
    # Keep one target box per image/class/family/attribute. Largest boxes are less ambiguous.
    best: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in candidates.values():
        key = (row["split"], row["image_id"], row["family"], row["class_name"], row["attribute_name"])
        current = best.get(key)
        if current is None or (-row["bbox_area"], row["bbox_key"]) < (-current["bbox_area"], current["bbox_key"]):
            best[key] = row
    counts["eligible_annotations"] = len(best)
    return list(best.values()), dict(sorted(counts.items()))


def enrich_image(row: dict[str, Any], parent: dict[str, dict[str, Any]]) -> dict[str, Any]:
    image = parent[row["image_id"]]
    return {
        "image_id": row["image_id"],
        "image_path": image["image_path"],
        "image_sha256": image["image_sha256"],
        "width": image["width"],
        "height": image["height"],
        "bbox": row["bbox"],
        "class_id": row["class_id"],
        "class_name": row["class_name"],
        "attribute_id": row["attribute_id"],
        "attribute_name": row["attribute_name"],
        "answer": row["answer"],
    }


def choose_control(
    candidates: list[dict[str, Any]],
    forbidden_images: set[str],
    key_parts: tuple[Any, ...],
) -> dict[str, Any] | None:
    eligible = [row for row in candidates if row["image_id"] not in forbidden_images]
    if not eligible:
        return None
    return min(eligible, key=lambda row: stable_hash(*key_parts, row["image_id"], row["bbox_key"]))


def pair_rows(
    candidates: list[dict[str, Any]],
    parent: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_key[(row["split"], row["family"], row["class_name"], row["attribute_name"])].append(row)
    for key, values in by_key.items():
        values.sort(key=lambda row: stable_hash(protocol["seed"], "candidate", *key, row["image_id"], row["bbox_key"]))
    all_by_split_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        all_by_split_family[(row["split"], row["family"])].append(row)

    output: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    contrast_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for split in ("train", "selection_dev", "sealed_confirmation"):
        cap = protocol["pair_caps"][split]
        for cell in protocol["allowed_contrasts"]:
            family, class_name = cell["family"], cell["class_name"]
            for raw_pair in cell["attribute_pairs"]:
                left_name, right_name = raw_pair
                left = by_key.get((split, family, class_name, left_name), [])
                right = by_key.get((split, family, class_name, right_name), [])
                pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
                used_left: set[str] = set()
                used_right: set[str] = set()
                for left_row in left:
                    if left_row["image_id"] in used_left:
                        continue
                    right_options = [
                        row for row in right
                        if row["image_id"] != left_row["image_id"] and row["image_id"] not in used_right
                    ]
                    if not right_options:
                        continue
                    right_row = min(
                        right_options,
                        key=lambda row: stable_hash(
                            protocol["seed"], "pair", split, family, class_name,
                            left_name, right_name, left_row["image_id"], row["image_id"], row["bbox_key"],
                        ),
                    )
                    pairs.append((left_row, right_row))
                    used_left.add(left_row["image_id"])
                    used_right.add(right_row["image_id"])
                    if len(pairs) >= cap:
                        break
                for left_row, right_row in pairs:
                    forbidden = {left_row["image_id"], right_row["image_id"]}
                    left_same = choose_control(
                        by_key.get((split, family, class_name, left_name), []),
                        forbidden,
                        (protocol["seed"], "left_same", split, left_row["image_id"], right_row["image_id"]),
                    )
                    right_same = choose_control(
                        by_key.get((split, family, class_name, right_name), []),
                        forbidden,
                        (protocol["seed"], "right_same", split, left_row["image_id"], right_row["image_id"]),
                    )
                    random_pool = [
                        row for (candidate_split, candidate_family), rows in all_by_split_family.items()
                        if candidate_split == split and candidate_family != family
                        for row in rows
                    ]
                    left_random = choose_control(
                        random_pool, forbidden,
                        (protocol["seed"], "left_random", split, left_row["image_id"], right_row["image_id"]),
                    )
                    right_random = choose_control(
                        random_pool, forbidden | ({left_random["image_id"]} if left_random else set()),
                        (protocol["seed"], "right_random", split, left_row["image_id"], right_row["image_id"]),
                    )
                    if None in (left_same, right_same, left_random, right_random):
                        dropped["missing_control"] += 1
                        continue
                    answers = [left_row["answer"], right_row["answer"]]
                    if int(stable_hash(protocol["seed"], "answer_order", left_row["image_id"], right_row["image_id"]), 16) % 2:
                        answers.reverse()
                    pair_id = stable_hash(
                        protocol["seed"], split, family, class_name,
                        left_name, right_name, left_row["image_id"], left_row["bbox_key"],
                        right_row["image_id"], right_row["bbox_key"],
                    )[:24]
                    article = "an" if class_name[0].lower() in "aeiou" else "a"
                    output.append({
                        "schema_version": "2026-09-14-v1",
                        "pair_id": pair_id,
                        "statistical_cluster_id": pair_id,
                        "split": split,
                        "family": family,
                        "class_name": class_name,
                        "question": (
                            f"The red box marks {article} {class_name.lower()}. Which option best describes "
                            f"the marked object: {answers[0]} or {answers[1]}? Answer with exactly one option."
                        ),
                        "candidate_answers": answers,
                        "image_a": enrich_image(left_row, parent),
                        "image_b": enrich_image(right_row, parent),
                        "controls": {
                            "image_a_same_attribute": enrich_image(left_same, parent),
                            "image_b_same_attribute": enrich_image(right_same, parent),
                            "image_a_random_other_family": enrich_image(left_random, parent),
                            "image_b_random_other_family": enrich_image(right_random, parent),
                        },
                        "intervention_contract": {
                            "ordinary_a_answer": left_row["answer"],
                            "ordinary_b_answer": right_row["answer"],
                            "a_address_b_value_answer": right_row["answer"],
                            "b_address_a_value_answer": left_row["answer"],
                            "same_attribute_exchange_is_no_op": True,
                            "random_other_family_is_diagnostic_only": True,
                        },
                    })
                    contrast_counts[(split, family, class_name, left_name, right_name)] += 1
    output.sort(key=lambda row: (row["split"], row["pair_id"]))
    details = [
        {
            "split": key[0], "family": key[1], "class_name": key[2],
            "left_attribute": key[3], "right_attribute": key[4], "pairs": value,
        }
        for key, value in sorted(contrast_counts.items())
    ]
    return output, {"dropped": dict(sorted(dropped.items())), "contrast_counts": details}


def validate_referenced_images(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected: dict[str, tuple[Path, str]] = {}
    for row in rows:
        records = [row["image_a"], row["image_b"], *row["controls"].values()]
        for record in records:
            item = (Path(record["image_path"]), record["image_sha256"])
            previous = expected.setdefault(record["image_id"], item)
            if previous != item:
                raise RuntimeError(f"inconsistent image identity: {record['image_id']}")
    failures = []
    total_bytes = 0
    for image_id, (path, expected_sha) in sorted(expected.items()):
        if not path.is_file():
            failures.append({"image_id": image_id, "reason": "missing"})
            continue
        total_bytes += path.stat().st_size
        if sha256_file(path) != expected_sha:
            failures.append({"image_id": image_id, "reason": "sha256_mismatch"})
    if failures:
        raise RuntimeError(f"referenced image validation failed: {failures[:5]}")
    return {"unique_images": len(expected), "total_bytes_hashed": total_bytes, "failures": 0}


def split_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    images: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for record in [row["image_a"], row["image_b"], *row["controls"].values()]:
            images[row["split"]].add(record["image_id"])
    pairs = {}
    names = sorted(images)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = images[left] & images[right]
            pairs[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise RuntimeError(f"image overlap across splits: {left}/{right}")
    return {"unique_images_by_split": {name: len(images[name]) for name in names}, "pairwise_overlap": pairs}


def caption_replay_rows(parent: dict[str, dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    eligible = [
        row for row in parent.values()
        if internal_split(row, protocol["split_policy"]) == "train"
    ]
    eligible.sort(key=lambda row: stable_hash(protocol["seed"], "caption_replay", row["image_id"]))
    count = protocol["caption_replay"]["examples"]
    if len(eligible) < count:
        raise RuntimeError("insufficient train-only caption replay rows")
    return [
        {
            "schema_version": "2026-09-14-v1",
            "source": "open_images_localized_narratives_replay",
            "split": "train",
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "prompt": row["prompt"],
            "response": row["response"],
        }
        for row in eligible[:count]
    ]


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
        draw.rectangle(tuple(value + (offset if index >= 2 else -offset) for index, value in enumerate(coordinates)), outline=(255, 32, 32))
    return output


def render_tile(row: dict[str, Any], size: tuple[int, int] = (720, 320)) -> Image.Image:
    tile = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    image_height = size[1] - 54
    half = size[0] // 2
    for index, key in enumerate(("image_a", "image_b")):
        record = row[key]
        with Image.open(record["image_path"]) as image:
            boxed = draw_box(image, record["bbox"])
            fitted = ImageOps.contain(boxed, (half - 12, image_height - 8))
        x = index * half + (half - fitted.width) // 2
        tile.paste(fitted, (x, 28 + (image_height - fitted.height) // 2))
        draw.text((index * half + 6, 6), f"{key[-1].upper()}: {record['class_name']} / {record['answer']}", fill="black", font=font)
    draw.text((6, size[1] - 20), f"{row['split']}  {row['pair_id']}  candidates={row['candidate_answers']}", fill="black", font=font)
    return tile


def render_train_audit_sheets(rows: list[dict[str, Any]], protocol: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    by_contrast: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        key = (row["family"], row["class_name"], tuple(sorted(row["candidate_answers"])))
        by_contrast[key].append(row)
    sample: list[dict[str, Any]] = []
    for key, values in sorted(by_contrast.items()):
        values.sort(key=lambda row: stable_hash(protocol["seed"], "audit", key, row["pair_id"]))
        sample.extend(values[: protocol["human_audit"]["examples_per_contrast"]])
    sample.sort(key=lambda row: stable_hash(protocol["seed"], "audit_order", row["pair_id"]))
    sample = sample[: protocol["human_audit"]["maximum_examples"]]
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    per_sheet = protocol["human_audit"]["examples_per_sheet"]
    for page, start in enumerate(range(0, len(sample), per_sheet), 1):
        page_rows = sample[start : start + per_sheet]
        columns = 2
        tile_size = (720, 320)
        rows_count = (len(page_rows) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_size[0], rows_count * tile_size[1]), (236, 239, 244))
        for index, row in enumerate(page_rows):
            sheet.paste(render_tile(row, tile_size), ((index % columns) * tile_size[0], (index // columns) * tile_size[1]))
        path = directory / f"openimages-attribute-routing-train-sheet-{page:02d}.png"
        sheet.save(path, optimize=True)
        records.append({"path": str(path), "sha256": sha256_file(path), "examples": len(page_rows)})
    return records


def build(protocol_path: Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    parent = load_manifest(Path(protocol["parent_manifest"]["path"]))
    candidates, scan_counts = scan_candidates(protocol, parent)
    rows, pairing = pair_rows(candidates, parent, protocol)
    counts = Counter(row["split"] for row in rows)
    for split, minimum in protocol["minimum_pairs"].items():
        if counts[split] < minimum:
            raise RuntimeError(f"{split} pair gate failed: {counts[split]} < {minimum}")
    overlaps = split_overlap(rows)
    image_validation = validate_referenced_images(rows)
    outputs = protocol["outputs"]
    pair_manifest = Path(outputs["pair_manifest"])
    pair_sha = write_jsonl_atomic(pair_manifest, rows)
    replay = caption_replay_rows(parent, protocol)
    replay_manifest = Path(outputs["caption_replay_manifest"])
    replay_sha = write_jsonl_atomic(replay_manifest, replay)
    sheets = render_train_audit_sheets(rows, protocol, Path(outputs["human_audit_directory"]))
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_pending_human_audit",
        "training_authorized": False,
        "confirmation_use_authorized": False,
        "claim_boundary": (
            "A deterministic real-image pilot was built and mechanically validated. Human visual audit "
            "and a separate frozen training protocol are still required; no parameter update or model claim is authorized."
        ),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "candidate_scan_counts": scan_counts,
        "pair_counts": dict(sorted(counts.items())),
        "pairing": pairing,
        "split_integrity": overlaps,
        "referenced_image_validation": image_validation,
        "pair_manifest": {"path": str(pair_manifest), "sha256": pair_sha, "rows": len(rows)},
        "caption_replay_manifest": {"path": str(replay_manifest), "sha256": replay_sha, "rows": len(replay)},
        "train_only_human_audit_sheets": sheets,
        "sealed_confirmation_rows_rendered_for_human_audit": 0,
    }
    write_json_atomic(Path(outputs["build_receipt"]), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.protocol)
    print(json.dumps({
        "status": receipt["status"],
        "pair_counts": receipt["pair_counts"],
        "pair_manifest_sha256": receipt["pair_manifest"]["sha256"],
        "audit_sheets": len(receipt["train_only_human_audit_sheets"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
