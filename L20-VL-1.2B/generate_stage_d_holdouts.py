#!/usr/bin/env python3
"""Generate independent Stage-D IID and binding-centric OOD holdouts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from generate_counterfactual_diagnostic import (
    BACKGROUNDS,
    COLORS,
    POSITIONS,
    SHAPES,
    TASKS,
    build_scene,
    contact_sheet,
    evaluate_scene,
    render_scene,
    sha256_file,
    stable_seed,
)


ROOT = Path(__file__).resolve().parent
BINDING_TASKS = ("color_binding", "shape_binding")
OOD_COORDINATES = (36, 47, 63, 65, 95, 97, 127, 129, 159, 161, 191, 193, 220)


def _far_apart(left: tuple[int, int], right: tuple[int, int], minimum: float = 72.0) -> bool:
    return math.hypot(left[0] - right[0], left[1] - right[1]) >= minimum


def _sample_position_pair(rng: random.Random, distribution: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if distribution == "iid":
        candidates = list(POSITIONS)
    elif distribution == "ood_geometry":
        candidates = [(x, y) for x in OOD_COORDINATES for y in OOD_COORDINATES]
    else:
        raise ValueError(f"unsupported distribution: {distribution}")
    for _ in range(1000):
        left, right = rng.sample(candidates, 2)
        if _far_apart(left, right):
            return left, right
    raise RuntimeError("could not sample non-overlapping target positions")


def evaluate_binding_scene(task: str, state: dict[str, Any], query: dict[str, Any]) -> str:
    if task == "color_binding":
        matches = [item for item in state["objects"] if item["shape"] == query["shape"]]
        if len(matches) != 1:
            raise RuntimeError("color-binding scene must identify exactly one object by shape")
        result = matches[0]["color"] == query["color"]
    elif task == "shape_binding":
        matches = [item for item in state["objects"] if item["color"] == query["color"]]
        if len(matches) != 1:
            raise RuntimeError("shape-binding scene must identify exactly one object by color")
        result = matches[0]["shape"] == query["shape"]
    else:
        raise ValueError(f"unsupported binding task: {task}")
    return "yes" if result else "no"


def build_binding_swap_scene(
    task: str, base_answer: str, seed: int, distribution: str
) -> dict[str, Any]:
    """Create a pair whose color/shape multisets stay fixed while bindings swap."""
    if task not in BINDING_TASKS or base_answer not in {"yes", "no"}:
        raise ValueError("invalid binding task or answer")
    rng = random.Random(seed)
    background, invariant_background = rng.sample(tuple(BACKGROUNDS), 2)
    query_shape, other_shape = rng.sample(SHAPES, 2)
    query_color, other_color = rng.sample(tuple(COLORS), 2)
    target_position, other_position = _sample_position_pair(rng, distribution)
    size = rng.choice((20, 24, 28)) if distribution == "iid" else rng.choice((18, 22, 26, 30))

    if task == "color_binding":
        target_color = query_color if base_answer == "yes" else other_color
        partner_color = other_color if base_answer == "yes" else query_color
        question = f"Is the {query_shape} {query_color}?"
        query = {"shape": query_shape, "color": query_color}
    else:
        target_color = query_color if base_answer == "yes" else other_color
        partner_color = other_color if base_answer == "yes" else query_color
        question = f"Is the {query_color} object a {query_shape}?"
        query = {"shape": query_shape, "color": query_color}

    base = {
        "background": background,
        "objects": [
            {
                "id": "target_shape",
                "color": target_color,
                "shape": query_shape,
                "x": target_position[0],
                "y": target_position[1],
                "size": size,
            },
            {
                "id": "other_shape",
                "color": partner_color,
                "shape": other_shape,
                "x": other_position[0],
                "y": other_position[1],
                "size": size,
            },
        ],
    }
    distractor_shape = next(shape for shape in SHAPES if shape not in {query_shape, other_shape})
    distractor_colors = [color for color in COLORS if color not in {query_color, other_color}]
    used = {target_position, other_position}
    candidate_positions = list(POSITIONS)
    if distribution == "ood_geometry":
        candidate_positions = [(x, y) for x in OOD_COORDINATES for y in OOD_COORDINATES]
    rng.shuffle(candidate_positions)
    for index in range(2):
        position = next(
            position
            for position in candidate_positions
            if position not in used
            and all(_far_apart(position, existing, minimum=66.0) for existing in used)
        )
        used.add(position)
        base["objects"].append({
            "id": f"d{index}",
            "color": distractor_colors[index],
            "shape": distractor_shape,
            "x": position[0],
            "y": position[1],
            "size": max(16, size - 3),
        })

    edited = deepcopy(base)
    edited_target = next(item for item in edited["objects"] if item["id"] == "target_shape")
    edited_other = next(item for item in edited["objects"] if item["id"] == "other_shape")
    edited_target["color"], edited_other["color"] = edited_other["color"], edited_target["color"]
    invariant = deepcopy(base)
    invariant["background"] = invariant_background
    expected_edit = "no" if base_answer == "yes" else "yes"
    answers = {
        "base": evaluate_binding_scene(task, base, query),
        "edited": evaluate_binding_scene(task, edited, query),
        "invariant": evaluate_binding_scene(task, invariant, query),
    }
    if answers != {"base": base_answer, "edited": expected_edit, "invariant": base_answer}:
        raise RuntimeError(f"binding oracle mismatch: {answers}")
    return {
        "question": question,
        "oracle_query": query,
        "base_state": base,
        "edited_state": edited,
        "invariant_state": invariant,
        "relevant_edit": {
            "factor": "swap_color_shape_bindings",
            "target_ids": ["target_shape", "other_shape"],
            "answer_must_change": True,
            "preserves_color_multiset": True,
            "preserves_shape_multiset": True,
        },
        "invariant_edit": {
            "factor": "background_palette",
            "before": background,
            "after": invariant_background,
            "answer_must_change": False,
        },
    }


def build_specs(
    development_families: int,
    iid_test_families: int,
    ood_binding_families: int,
    seed: int,
) -> list[dict[str, Any]]:
    if development_families % (len(TASKS) * 2):
        raise ValueError("development families must be divisible by task/answer strata")
    if iid_test_families % (len(TASKS) * 2):
        raise ValueError("IID test families must be divisible by task/answer strata")
    if ood_binding_families % (len(BINDING_TASKS) * 2):
        raise ValueError("OOD binding families must be divisible by task/answer strata")
    specs = []
    iid_layouts = (
        ("development", "iid", TASKS, development_families),
        ("iid_test", "iid", TASKS, iid_test_families),
    )
    for split, distribution, tasks, total in iid_layouts:
        per_stratum = total // (len(tasks) * 2)
        for task in tasks:
            for answer in ("yes", "no"):
                for index in range(per_stratum):
                    scene_seed = stable_seed(seed, split, distribution, task, answer, index)
                    family = hashlib.sha256(
                        f"stage-d|{split}|{task}|{answer}|{scene_seed}".encode()
                    ).hexdigest()[:24]
                    specs.append({
                        "pair_id": f"stage-d-{split}-{task}-{answer}-{index:04d}",
                        "scene_family_id": family,
                        "split": split,
                        "distribution": distribution,
                        "task": task,
                        "base_answer": answer,
                        "scene_seed": scene_seed,
                        "binding_swap": False,
                        "challenge": "iid_replication",
                    })
    ood_challenges = (
        ("binding_swap_only", "iid"),
        ("binding_swap_plus_boundary_geometry", "ood_geometry"),
    )
    if ood_binding_families % (len(BINDING_TASKS) * 2 * len(ood_challenges)):
        raise ValueError("OOD binding families must divide evenly across challenge/task/answer strata")
    per_ood_stratum = ood_binding_families // (len(BINDING_TASKS) * 2 * len(ood_challenges))
    for challenge, distribution in ood_challenges:
        for task in BINDING_TASKS:
            for answer in ("yes", "no"):
                for index in range(per_ood_stratum):
                    scene_seed = stable_seed(seed, "ood_binding", challenge, task, answer, index)
                    family = hashlib.sha256(
                        f"stage-d|ood_binding|{challenge}|{task}|{answer}|{scene_seed}".encode()
                    ).hexdigest()[:24]
                    specs.append({
                        "pair_id": f"stage-d-ood_binding-{challenge}-{task}-{answer}-{index:04d}",
                        "scene_family_id": family,
                        "split": "ood_binding",
                        "distribution": distribution,
                        "task": task,
                        "base_answer": answer,
                        "scene_seed": scene_seed,
                        "binding_swap": True,
                        "challenge": challenge,
                    })
    return sorted(specs, key=lambda item: item["pair_id"])


def generate(admission_path: Path, protocol_path: Path) -> dict[str, Any]:
    admission = json.loads(admission_path.read_text())
    if admission.get("required_protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("protocol hash does not match Stage-D data admission")
    if admission.get("generator_authorized") is not True:
        raise RuntimeError("Stage-D holdout generation is not authorized")
    if admission.get("formal_training_authorized") is not False:
        raise RuntimeError("holdout generation must not authorize model training")
    output = Path(admission["destination"])
    receipt_path = Path(admission["receipt"])
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Stage-D output or receipt already exists")
    exclusion_path = Path(admission["exclude_manifest"]["path"])
    if sha256_file(exclusion_path) != admission["exclude_manifest"]["sha256"]:
        raise RuntimeError("Stage-C exclusion manifest hash mismatch")
    excluded_hashes = set()
    excluded_families = set()
    for line in exclusion_path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        excluded_families.add(row["scene_family_id"])
        excluded_hashes.update(row[f"{variant}_image_sha256"] for variant in ("base", "edited", "invariant"))

    specs = build_specs(
        admission["development_families"],
        admission["iid_test_families"],
        admission["ood_binding_families"],
        admission["seed"],
    )
    output.mkdir(parents=True)
    records = []
    image_hashes = set()
    total_bytes = 0
    collision_retries = 0
    for number, spec in enumerate(specs, 1):
        if spec["scene_family_id"] in excluded_families:
            raise RuntimeError("new Stage-D scene family collides with Stage-C")
        expected_edit = "no" if spec["base_answer"] == "yes" else "yes"
        for attempt in range(101):
            actual_seed = spec["scene_seed"] if attempt == 0 else stable_seed(spec["scene_seed"], "retry", attempt)
            if spec["binding_swap"]:
                scene = build_binding_swap_scene(
                    spec["task"], spec["base_answer"], actual_seed, spec["distribution"]
                )
                oracle = evaluate_binding_scene
            else:
                scene = build_scene(spec["task"], spec["base_answer"], actual_seed)
                oracle = evaluate_scene
            expected = {"base": spec["base_answer"], "edited": expected_edit, "invariant": spec["base_answer"]}
            actual = {
                variant: oracle(spec["task"], scene[f"{variant}_state"], scene["oracle_query"])
                for variant in ("base", "edited", "invariant")
            }
            if actual != expected:
                raise RuntimeError(f"oracle mismatch for {spec['pair_id']}: {actual} != {expected}")
            paths = {}
            hashes = {}
            candidate_bytes = 0
            for variant in ("base", "edited", "invariant"):
                path = output / "images" / spec["split"] / f"{spec['pair_id']}-{variant}.png"
                render_scene(scene[f"{variant}_state"], path, admission["render_scale"])
                paths[variant] = str(path)
                hashes[variant] = sha256_file(path)
                candidate_bytes += path.stat().st_size
            if len(set(hashes.values())) == 3 and not (set(hashes.values()) & (image_hashes | excluded_hashes)):
                break
            collision_retries += 1
        else:
            raise RuntimeError(f"could not resolve image collision for {spec['pair_id']}")
        image_hashes.update(hashes.values())
        total_bytes += candidate_bytes
        records.append({
            "pair_id": spec["pair_id"],
            "scene_family_id": spec["scene_family_id"],
            "split": spec["split"],
            "distribution": spec["distribution"],
            "challenge": spec["challenge"],
            "task": spec["task"],
            "question": scene["question"],
            "candidate_answers": ["yes", "no"],
            "base_answer": spec["base_answer"],
            "edited_answer": expected_edit,
            "invariant_answer": spec["base_answer"],
            **{f"{variant}_image_path": paths[variant] for variant in ("base", "edited", "invariant")},
            **{f"{variant}_image_sha256": hashes[variant] for variant in ("base", "edited", "invariant")},
            "relevant_edit": scene["relevant_edit"],
            "invariant_edit": scene["invariant_edit"],
            "scene_state": scene["base_state"],
            "oracle_query": scene["oracle_query"],
            "render_seed": actual_seed,
            "collision_retry_count": attempt,
        })
        if number % 250 == 0:
            print(f"generated {number}/{len(specs)} Stage-D families", flush=True)
    if total_bytes > admission["max_total_bytes"]:
        raise RuntimeError("Stage-D image corpus exceeds byte cap")
    manifest = output / "manifest.jsonl"
    with manifest.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    audit_sheets = []
    for split in ("development", "iid_test", "ood_binding"):
        for task in sorted({row["task"] for row in records if row["split"] == split}):
            path = output / "audit" / f"{split}-{task}.png"
            contact_sheet(
                task,
                [row for row in records if row["split"] == split and row["task"] == task],
                path,
            )
            audit_sheets.append({
                "split": split,
                "task": task,
                "path": str(path),
                "sha256": sha256_file(path),
            })
    counts = Counter((row["split"], row["task"], row["base_answer"]) for row in records)
    split_counts = Counter(row["split"] for row in records)
    challenge_counts = Counter((row["split"], row["challenge"], row["task"], row["base_answer"]) for row in records)
    receipt = {
        "schema_version": "2026-09-13-v1",
        "status": "complete_pending_human_audit",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "formal_training": False,
        "training_prediction_tokens": 0,
        "protocol_sha256": sha256_file(protocol_path),
        "admission_sha256": sha256_file(admission_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "scene_families": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "stratum_counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "challenge_stratum_counts": {"|".join(key): value for key, value in sorted(challenge_counts.items())},
        "rendered_images": len(image_hashes),
        "unique_image_hashes": len(image_hashes),
        "excluded_stage_c_image_hashes": len(excluded_hashes),
        "exact_stage_c_image_overlap": len(image_hashes & excluded_hashes),
        "exact_stage_c_family_overlap": len({row["scene_family_id"] for row in records} & excluded_families),
        "collision_retries": collision_retries,
        "total_image_bytes": total_bytes,
        "audit_sheets": audit_sheets,
        "iid_test_status": "sealed",
        "ood_binding_status": "sealed",
        "claim_boundary": (
            "Generated synthetic families are independent by exact family and image hash. Human visual "
            "audit remains required; these data cannot establish natural-image generalization."
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    generate(args.admission, args.protocol)


if __name__ == "__main__":
    main()
