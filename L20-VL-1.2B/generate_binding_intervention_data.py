#!/usr/bin/env python3
"""Generate attribute-preserving binding interventions with multi-question controls.

Each scene pair is rendered once and reused across six questions: two questions
whose answers must flip after the binding edit and four questions whose answers
must stay fixed.  Splits are assigned at scene-pair granularity so questions or
image variants from one scene can never cross partitions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from generate_counterfactual_diagnostic import (
    BACKGROUNDS,
    COLORS,
    POSITIONS,
    SHAPES,
    object_record,
    render_scene,
    sha256_file,
    stable_seed,
)


ROOT = Path(__file__).resolve().parent
CHALLENGES = ("shape_color_swap", "position_color_swap")
QUESTION_ROLES = (
    "affected_positive_to_negative",
    "affected_negative_to_positive",
    "invariant_present_positive",
    "invariant_absent_negative",
    "invariant_control_positive",
    "invariant_control_negative",
)
VARIANTS = ("base", "edited", "invariant")
SPLITS = ("train", "mechanism_dev", "selection_dev", "final_test")


def _far_apart(left: tuple[int, int], right: tuple[int, int], minimum: float = 72.0) -> bool:
    return math.hypot(left[0] - right[0], left[1] - right[1]) >= minimum


def _candidate_positions(distribution: str) -> list[tuple[int, int]]:
    if distribution == "grid":
        return list(POSITIONS)
    if distribution == "jitter":
        coordinates = (39, 61, 91, 123, 157, 189, 217)
        return [(x, y) for x in coordinates for y in coordinates]
    if distribution == "boundary":
        coordinates = (30, 47, 63, 65, 95, 97, 127, 129, 159, 161, 193, 209, 226)
        return [(x, y) for x in coordinates for y in coordinates]
    raise ValueError(f"unsupported distribution: {distribution}")


def _sample_positions(rng: random.Random, distribution: str) -> tuple[tuple[int, int], ...]:
    candidates = _candidate_positions(distribution)
    for _ in range(2000):
        chosen = rng.sample(candidates, 3)
        if all(_far_apart(left, right, 70.0) for index, left in enumerate(chosen) for right in chosen[index + 1 :]):
            return tuple(chosen)
    raise RuntimeError("could not sample three visible, separated object positions")


def _position_binding_positions(rng: random.Random, distribution: str) -> tuple[tuple[int, int], ...]:
    if distribution == "grid":
        y = rng.choice((48, 128, 208))
        return (48, y), (208, y), (128, 128 if y != 128 else 208)
    if distribution == "jitter":
        y_left, y_right = rng.sample((61, 91, 123, 157, 189), 2)
        return (39, y_left), (217, y_right), (123, 123)
    if distribution == "boundary":
        y_left, y_right = rng.sample((47, 65, 95, 129, 161, 193, 209), 2)
        return (30, y_left), (226, y_right), (127, 127)
    raise ValueError(f"unsupported distribution: {distribution}")


def _answer(value: bool) -> str:
    return "yes" if value else "no"


def evaluate_query(state: dict[str, Any], query: dict[str, Any]) -> str:
    """Independent symbolic oracle for every generated question type."""
    objects = state["objects"]
    kind = query["kind"]
    if kind == "shape_color":
        matches = [item for item in objects if item["shape"] == query["shape"]]
        if len(matches) != 1:
            raise RuntimeError("shape_color query must identify exactly one object")
        result = matches[0]["color"] == query["color"]
    elif kind == "leftmost_color":
        leftmost = min(objects, key=lambda item: item["x"])
        if sum(item["x"] == leftmost["x"] for item in objects) != 1:
            raise RuntimeError("leftmost_color query requires a unique leftmost object")
        result = leftmost["color"] == query["color"]
    elif kind == "color_presence":
        result = any(item["color"] == query["color"] for item in objects)
    elif kind == "shape_color_presence":
        result = any(
            item["shape"] == query["shape"] and item["color"] == query["color"]
            for item in objects
        )
    else:
        raise ValueError(f"unsupported query kind: {kind}")
    return _answer(result)


def _question_record(role: str, question: str, query: dict[str, Any]) -> dict[str, Any]:
    return {"question_role": role, "question": question, "oracle_query": query}


def build_scene_pair(challenge: str, seed: int, distribution: str) -> dict[str, Any]:
    """Build one binding edit with affected and unaffected questions."""
    if challenge not in CHALLENGES:
        raise ValueError(f"unsupported challenge: {challenge}")
    rng = random.Random(seed)
    background, invariant_background = rng.sample(tuple(BACKGROUNDS), 2)
    first_color, second_color, control_color, absent_color = rng.sample(tuple(COLORS), 4)
    target_shape, partner_shape, control_shape = rng.sample(SHAPES, 3)
    size = rng.choice((20, 24, 28)) if distribution != "boundary" else rng.choice((18, 20, 22))

    if challenge == "shape_color_swap":
        target_position, partner_position, control_position = _sample_positions(rng, distribution)
        base_objects = [
            object_record("target", first_color, target_shape, target_position, size),
            object_record("partner", second_color, partner_shape, partner_position, size),
            object_record("control", control_color, control_shape, control_position, max(17, size - 2)),
        ]
        affected_kind = "shape_color"
        affected_subject = {"shape": target_shape}
        positive_question = f"Is the {target_shape} {first_color}?"
        negative_question = f"Is the {target_shape} {second_color}?"
    else:
        left_position, right_position, control_position = _position_binding_positions(rng, distribution)
        base_objects = [
            object_record("target", first_color, target_shape, left_position, size),
            object_record("partner", second_color, target_shape, right_position, size),
            object_record("control", control_color, control_shape, control_position, max(17, size - 2)),
        ]
        affected_kind = "leftmost_color"
        affected_subject = {}
        positive_question = f"Is the leftmost object {first_color}?"
        negative_question = f"Is the leftmost object {second_color}?"

    base = {"background": background, "objects": base_objects}
    edited = deepcopy(base)
    edited_by_id = {item["id"]: item for item in edited["objects"]}
    edited_by_id["target"]["color"], edited_by_id["partner"]["color"] = (
        edited_by_id["partner"]["color"],
        edited_by_id["target"]["color"],
    )
    invariant = deepcopy(base)
    invariant["background"] = invariant_background

    questions = [
        _question_record(
            "affected_positive_to_negative",
            positive_question,
            {"kind": affected_kind, **affected_subject, "color": first_color},
        ),
        _question_record(
            "affected_negative_to_positive",
            negative_question,
            {"kind": affected_kind, **affected_subject, "color": second_color},
        ),
        _question_record(
            "invariant_present_positive",
            f"Is there a {first_color} object?",
            {"kind": "color_presence", "color": first_color},
        ),
        _question_record(
            "invariant_absent_negative",
            f"Is there a {absent_color} object?",
            {"kind": "color_presence", "color": absent_color},
        ),
        _question_record(
            "invariant_control_positive",
            f"Is there a {control_color} {control_shape}?",
            {"kind": "shape_color_presence", "shape": control_shape, "color": control_color},
        ),
        _question_record(
            "invariant_control_negative",
            f"Is there a {absent_color} {control_shape}?",
            {"kind": "shape_color_presence", "shape": control_shape, "color": absent_color},
        ),
    ]
    states = {"base": base, "edited": edited, "invariant": invariant}
    for question in questions:
        answers = {
            variant: evaluate_query(state, question["oracle_query"])
            for variant, state in states.items()
        }
        question["answers"] = answers
        should_flip = question["question_role"].startswith("affected_")
        if should_flip != (answers["base"] != answers["edited"]):
            raise RuntimeError(f"answer-change contract failed for {question['question_role']}: {answers}")
        if answers["invariant"] != answers["base"]:
            raise RuntimeError(f"invariant answer changed for {question['question_role']}: {answers}")

    return {
        "base_state": base,
        "edited_state": edited,
        "invariant_state": invariant,
        "questions": questions,
        "relevant_edit": {
            "factor": "swap_target_partner_colors",
            "target_ids": ["target", "partner"],
            "preserves_color_multiset": True,
            "preserves_shape_multiset": True,
            "preserves_object_count": True,
        },
        "invariant_edit": {
            "factor": "background_palette",
            "before": background,
            "after": invariant_background,
            "answer_must_change": False,
        },
    }


def build_specs(partitions: dict[str, dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if set(partitions) != set(SPLITS):
        raise ValueError(f"partitions must be exactly {SPLITS}")
    specs = []
    for split in SPLITS:
        config = partitions[split]
        count = int(config["scene_pairs"])
        if count < len(CHALLENGES) or count % len(CHALLENGES):
            raise ValueError(f"{split} scene_pairs must be a positive multiple of {len(CHALLENGES)}")
        per_challenge = count // len(CHALLENGES)
        for challenge in CHALLENGES:
            for index in range(per_challenge):
                scene_seed = stable_seed(seed, split, challenge, index)
                pair_id = hashlib.sha256(
                    f"binding-v1|{split}|{challenge}|{scene_seed}".encode()
                ).hexdigest()[:24]
                specs.append({
                    "pair_id": f"bind-{split}-{challenge}-{index:05d}",
                    "scene_pair_id": pair_id,
                    "split": split,
                    "distribution": config["distribution"],
                    "challenge": challenge,
                    "scene_seed": scene_seed,
                })
    return sorted(specs, key=lambda item: item["pair_id"])


def _rgb_histogram(path: Path) -> Counter:
    with Image.open(path) as image:
        return Counter(image.convert("RGB").getdata())


def _histogram_l1(left: Counter, right: Counter) -> int:
    return sum(abs(left[key] - right[key]) for key in left.keys() | right.keys())


def _render_audit_sheet(records: list[dict[str, Any]], output: Path) -> None:
    by_pair = {}
    for row in records:
        by_pair.setdefault(row["scene_pair_id"], []).append(row)
    chosen = sorted(by_pair.items(), key=lambda item: stable_seed("audit", item[0]))[:8]
    cell = 256
    label_height = 128
    sheet = Image.new("RGB", (cell * 3, len(chosen) * (cell + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, (_, rows) in enumerate(chosen):
        rows = sorted(rows, key=lambda item: QUESTION_ROLES.index(item["question_role"]))
        top = row_index * (cell + label_height)
        for column, variant in enumerate(VARIANTS):
            with Image.open(rows[0][f"{variant}_image_path"]) as image:
                sheet.paste(image.convert("RGB"), (column * cell, top))
        labels = [
            f"{row['question_role']}: {row['base_answer']}->{row['edited_answer']} | {row['question']}"
            for row in rows
        ]
        draw.multiline_text((5, top + cell + 4), "\n".join(labels), fill="black", font=font, spacing=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def generate(protocol_path: Path, output_override: Path | None = None, receipt_override: Path | None = None) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_binding_intervention_data_generation_v1":
        raise RuntimeError("binding-intervention data generation is not authorized")
    if protocol.get("model_training_authorized_by_this_protocol") is not False:
        raise RuntimeError("data protocol must not authorize model training")
    if protocol.get("external_downloads_authorized") is not False:
        raise RuntimeError("binding-intervention generation must remain download-free")

    output = output_override or Path(protocol["output"])
    receipt_path = receipt_override or Path(protocol["receipt"])
    if output.exists() or receipt_path.exists():
        raise FileExistsError("output or receipt already exists")
    output.mkdir(parents=True)

    generate_partitions = tuple(protocol.get("generate_partitions", SPLITS))
    if not generate_partitions or len(set(generate_partitions)) != len(generate_partitions):
        raise RuntimeError("generate_partitions must be non-empty and unique")
    if any(split not in SPLITS for split in generate_partitions):
        raise RuntimeError("generate_partitions contains an unknown split")
    if "final_test" in generate_partitions and protocol.get("final_test_unseal_authorized") is not True:
        raise RuntimeError("final_test generation remains sealed")
    specs = [
        row
        for row in build_specs(protocol["partitions"], int(protocol["seed"]))
        if row["split"] in generate_partitions
    ]
    records = []
    unique_image_hashes = set()
    total_bytes = 0
    histograms_equal = Counter()
    split_challenge_counts = Counter()
    role_answer_counts = Counter()
    collision_retries = 0

    for number, spec in enumerate(specs, 1):
        for collision_attempt in range(101):
            actual_seed = (
                spec["scene_seed"]
                if collision_attempt == 0
                else stable_seed(spec["scene_seed"], "collision-retry", collision_attempt)
            )
            scene = build_scene_pair(spec["challenge"], actual_seed, spec["distribution"])
            paths = {}
            hashes = {}
            candidate_bytes = 0
            for variant in VARIANTS:
                path = output / "images" / spec["split"] / f"{spec['scene_pair_id']}-{variant}.png"
                render_scene(scene[f"{variant}_state"], path, int(protocol["render_scale"]))
                paths[variant] = path
                hashes[variant] = sha256_file(path)
                candidate_bytes += path.stat().st_size
            if len(set(hashes.values())) != len(VARIANTS):
                collision_retries += 1
                continue
            if set(hashes.values()) & unique_image_hashes:
                collision_retries += 1
                continue
            break
        else:
            raise RuntimeError(f"could not resolve image collision in {spec['scene_pair_id']}")
        unique_image_hashes.update(hashes.values())
        total_bytes += candidate_bytes

        base_histogram = _rgb_histogram(paths["base"])
        edited_histogram = _rgb_histogram(paths["edited"])
        histogram_equal = base_histogram == edited_histogram
        histogram_l1 = _histogram_l1(base_histogram, edited_histogram)
        histograms_equal[(spec["split"], spec["challenge"], histogram_equal)] += 1
        if spec["challenge"] == "position_color_swap" and not histogram_equal:
            raise RuntimeError("position-color swap must preserve the exact rendered RGB histogram")

        color_multisets = {
            variant: Counter(item["color"] for item in scene[f"{variant}_state"]["objects"])
            for variant in ("base", "edited")
        }
        shape_multisets = {
            variant: Counter(item["shape"] for item in scene[f"{variant}_state"]["objects"])
            for variant in ("base", "edited")
        }
        if color_multisets["base"] != color_multisets["edited"] or shape_multisets["base"] != shape_multisets["edited"]:
            raise RuntimeError("attribute multiset changed across binding intervention")

        for question_index, question in enumerate(scene["questions"]):
            family = hashlib.sha256(
                f"{spec['scene_pair_id']}|{question['question_role']}".encode()
            ).hexdigest()[:24]
            answers = question["answers"]
            record = {
                "pair_id": spec["pair_id"],
                "scene_pair_id": spec["scene_pair_id"],
                "scene_family_id": family,
                "statistical_cluster_id": spec["scene_pair_id"],
                "split": spec["split"],
                "distribution": spec["distribution"],
                "challenge": spec["challenge"],
                "task": "binding_intervention",
                "question_index": question_index,
                "question_role": question["question_role"],
                "question": question["question"],
                "oracle_query": question["oracle_query"],
                "candidate_answers": ["yes", "no"],
                "base_answer": answers["base"],
                "edited_answer": answers["edited"],
                "invariant_answer": answers["invariant"],
                "answer_change_required": answers["base"] != answers["edited"],
                **{f"{variant}_image_path": str(paths[variant]) for variant in VARIANTS},
                **{f"{variant}_image_sha256": hashes[variant] for variant in VARIANTS},
                "base_edited_rgb_histogram_equal": histogram_equal,
                "base_edited_rgb_histogram_l1_pixels": histogram_l1,
                "relevant_edit": scene["relevant_edit"],
                "invariant_edit": scene["invariant_edit"],
                "scene_state": scene["base_state"],
                "render_seed": actual_seed,
                "collision_retry_count": collision_attempt,
            }
            records.append(record)
            role_answer_counts[(spec["split"], question["question_role"], answers["base"], answers["edited"])] += 1
        split_challenge_counts[(spec["split"], spec["challenge"])] += 1
        if number % 250 == 0:
            print(f"generated {number}/{len(specs)} scene pairs", flush=True)

    if total_bytes > int(protocol["max_total_image_bytes"]):
        raise RuntimeError("generated corpus exceeds image byte cap")
    manifest = output / "manifest.jsonl"
    with manifest.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    audit_sheets = []
    for split in generate_partitions:
        for challenge in CHALLENGES:
            path = output / "audit" / f"{split}-{challenge}.png"
            _render_audit_sheet(
                [row for row in records if row["split"] == split and row["challenge"] == challenge],
                path,
            )
            audit_sheets.append({"split": split, "challenge": challenge, "path": str(path), "sha256": sha256_file(path)})

    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_pending_human_audit",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "formal_training": False,
        "training_prediction_tokens": 0,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "output": str(output),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "scene_pairs": len(specs),
        "question_families": len(records),
        "questions_per_pair": len(QUESTION_ROLES),
        "rendered_images": len(specs) * len(VARIANTS),
        "unique_image_hashes": len(unique_image_hashes),
        "deterministic_collision_retries": collision_retries,
        "total_image_bytes": total_bytes,
        "generated_partitions": list(generate_partitions),
        "final_test_generated": "final_test" in generate_partitions,
        "split_challenge_scene_pairs": {"|".join(key): value for key, value in sorted(split_challenge_counts.items())},
        "split_role_answer_counts": {"|".join(key): value for key, value in sorted(role_answer_counts.items())},
        "rendered_histogram_checks": {"|".join(map(str, key)): value for key, value in sorted(histograms_equal.items())},
        "invariants": {
            "attribute_multisets_preserved": True,
            "all_affected_questions_flip": all(row["answer_change_required"] for row in records if row["question_role"].startswith("affected_")),
            "all_control_questions_stay_fixed": all(not row["answer_change_required"] for row in records if row["question_role"].startswith("invariant_")),
            "all_invariant_images_preserve_answers": all(row["base_answer"] == row["invariant_answer"] for row in records),
            "position_swap_exact_rgb_histogram_preserved": all(
                row["base_edited_rgb_histogram_equal"] for row in records if row["challenge"] == "position_color_swap"
            ),
            "split_atomic_unit": "scene_pair_id",
        },
        "audit_sheets": audit_sheets,
        "claim_boundary": "This is deterministic synthetic mechanism data pending human audit. It is not natural-image evidence and does not authorize model training.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "binding_intervention_data_protocol.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    generate(args.protocol, args.output, args.receipt)


if __name__ == "__main__":
    main()
