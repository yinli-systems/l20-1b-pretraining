#!/usr/bin/env python3
"""Frozen multi-view posture evaluation on audited real-image hard negatives."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
CANDIDATES = ("sitting", "standing")
VIEWS = ("full", "marked", "crop")
CONDITIONS = ("true_image", "paired_swap", "blank")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = []
    for pair in pairs:
        for side, donor_side in (("image_a", "image_b"), ("image_b", "image_a")):
            image = pair[side]
            donor = pair[donor_side]
            if image["answer"] == donor["answer"]:
                raise ValueError(f"pair lacks opposite posture labels: {pair['pair_id']}")
            targets.append(
                {
                    "target_id": f"{pair['pair_id']}:{side}",
                    "pair_id": pair["pair_id"],
                    "fold": int(pair["crossfit"]["fold"]),
                    "side": side,
                    "class_name": pair["class_name"],
                    "expected": image["answer"],
                    "image": image,
                    "donor": donor,
                }
            )
    return sorted(targets, key=lambda row: row["target_id"])


def square_crop(image: Image.Image, bbox: list[float], padding_fraction: float) -> Image.Image:
    width, height = image.size
    x0, x1, y0, y1 = bbox
    box_width = max(1.0, (x1 - x0) * width)
    box_height = max(1.0, (y1 - y0) * height)
    pad_x = box_width * padding_fraction
    pad_y = box_height * padding_fraction
    left = max(0, math.floor(x0 * width - pad_x))
    top = max(0, math.floor(y0 * height - pad_y))
    right = min(width, math.ceil(x1 * width + pad_x))
    bottom = min(height, math.ceil(y1 * height + pad_y))
    if right <= left or bottom <= top:
        raise ValueError("invalid crop box")
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas


def render_view(image_record: dict[str, Any], view: str, padding_fraction: float) -> Image.Image:
    if view not in VIEWS:
        raise ValueError(f"unknown view: {view}")
    with Image.open(image_record["image_path"]) as source:
        image = source.convert("RGB")
    if view == "full":
        return image
    if view == "crop":
        return square_crop(image, image_record["bbox"], padding_fraction)
    width, height = image.size
    x0, x1, y0, y1 = image_record["bbox"]
    rectangle = (
        round(x0 * (width - 1)),
        round(y0 * (height - 1)),
        round(x1 * (width - 1)),
        round(y1 * (height - 1)),
    )
    marked = image.copy()
    line_width = max(3, round(min(width, height) * 0.008))
    ImageDraw.Draw(marked).rectangle(rectangle, outline=(255, 0, 0), width=line_width)
    return marked


def prompt_for(view: str, class_name: str) -> str:
    label = class_name.lower()
    prompts = {
        "full": f"Question: Is the {label} sitting or standing?\nAnswer:",
        "marked": f"Question: Is the {label} inside the red box sitting or standing?\nAnswer:",
        "crop": "Question: Is the person in this crop sitting or standing?\nAnswer:",
    }
    return prompts[view]


def bbox_grid_targets(bbox: list[float], grid: int = 7) -> tuple[int, list[int]]:
    x0, x1, y0, y1 = bbox
    center_x = min(grid - 1, max(0, int(((x0 + x1) / 2) * grid)))
    center_y = min(grid - 1, max(0, int(((y0 + y1) / 2) * grid)))
    center = center_y * grid + center_x
    region = []
    for row in range(grid):
        for column in range(grid):
            cell_x = (column + 0.5) / grid
            cell_y = (row + 0.5) / grid
            if x0 <= cell_x <= x1 and y0 <= cell_y <= y1:
                region.append(row * grid + column)
    return center, region or [center]


def interval(values: list[float], clusters: list[str], protocol: dict[str, Any]) -> dict[str, Any]:
    from counterfactual_losses import paired_cluster_bootstrap

    spec = protocol["statistics"]
    result = paired_cluster_bootstrap(
        values,
        clusters,
        resamples=int(spec["bootstrap_resamples"]),
        confidence=float(spec["confidence"]),
        seed=int(spec["bootstrap_seed"]),
    )
    return {
        "estimate": result.estimate,
        "lower_95_ci": result.lower,
        "upper_95_ci": result.upper,
        "pair_clusters": result.clusters,
        "samples": result.samples,
        "resamples": result.resamples,
    }


def pair_joint(records: list[dict[str, Any]]) -> tuple[list[float], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_id"]].append(record)
    values, clusters = [], []
    for pair_id, group in sorted(grouped.items()):
        if len(group) != 2:
            raise ValueError(f"incomplete pair in metrics: {pair_id}")
        values.append(100.0 * float(all(row["correct"] for row in group)))
        clusters.append(pair_id)
    return values, clusters


def summarize_condition(records: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    clusters = [record["pair_id"] for record in records]
    accuracy = [100.0 * float(record["correct"]) for record in records]
    margins = [float(record["correct_minus_incorrect_logprob"]) for record in records]
    joint, joint_clusters = pair_joint(records)
    address_records = [record for record in records if record.get("address") is not None]
    address = None
    if address_records:
        address_clusters = [record["pair_id"] for record in address_records]
        address = {
            "bbox_center_top1_percent": interval(
                [100.0 * float(record["address"]["bbox_center_top1"]) for record in address_records],
                address_clusters,
                protocol,
            ),
            "bbox_region_top1_percent": interval(
                [100.0 * float(record["address"]["bbox_region_top1"]) for record in address_records],
                address_clusters,
                protocol,
            ),
            "bbox_region_attention_mass_percent": interval(
                [100.0 * float(record["address"]["bbox_region_attention_mass"]) for record in address_records],
                address_clusters,
                protocol,
            ),
            "candidate_attention_max_gap": max(
                float(record["address"]["candidate_attention_max_gap"])
                for record in address_records
            ),
        }
    return {
        "per_image_accuracy_percent": interval(accuracy, clusters, protocol),
        "pair_joint_accuracy_percent": interval(joint, joint_clusters, protocol),
        "correct_minus_incorrect_logprob": interval(margins, clusters, protocol),
        "score_ties": sum(record["predicted"] is None for record in records),
        "address": address,
    }


def compare_conditions(
    true_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    true = {record["target_id"]: record for record in true_records}
    control = {record["target_id"]: record for record in control_records}
    if true.keys() != control.keys():
        raise ValueError("condition target sets differ")
    ids = sorted(true)
    differences = [
        100.0 * (float(true[target]["correct"]) - float(control[target]["correct"]))
        for target in ids
    ]
    clusters = [true[target]["pair_id"] for target in ids]
    return interval(differences, clusters, protocol)


def strict_bidirectional_flip(
    true_records: list[dict[str, Any]],
    swapped_records: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    true_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    swapped = {record["target_id"]: record for record in swapped_records}
    for record in true_records:
        true_by_pair[record["pair_id"]].append(record)
    values, clusters = [], []
    for pair_id, group in sorted(true_by_pair.items()):
        if len(group) != 2:
            raise ValueError(f"incomplete pair in flip metric: {pair_id}")
        passes = all(
            record["correct"]
            and swapped[record["target_id"]]["predicted"] == record["donor_expected"]
            for record in group
        )
        values.append(100.0 * float(passes))
        clusters.append(pair_id)
    return interval(values, clusters, protocol)


def verify_images(targets: list[dict[str, Any]]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    mismatches = []
    for target in targets:
        for key in ("image", "donor"):
            record = target[key]
            image_id = record["image_id"]
            if image_id in seen:
                continue
            path = Path(record["image_path"])
            actual = sha256_file(path) if path.is_file() else None
            seen[image_id] = actual or "missing"
            if actual != record["image_sha256"]:
                mismatches.append(
                    {"image_id": image_id, "path": str(path), "expected": record["image_sha256"], "actual": actual}
                )
    return {
        "checked_unique_images": len(seen),
        "mismatches": mismatches,
        "all_match_manifest": not mismatches,
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_posture_zero_shot_development_v1":
        raise RuntimeError("posture zero-shot evaluation is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("zero-shot protocol must not authorize training")
    if protocol.get("model_evaluation_authorized") is not True:
        raise RuntimeError("model evaluation is not authorized")
    source = protocol["source_code"]
    if sha256_file(Path(__file__)) != source["evaluator_sha256"]:
        raise RuntimeError("posture evaluator source hash mismatch")
    test_path = ROOT / "test_evaluate_openimages_posture_zero_shot_v1.py"
    if sha256_file(test_path) != source["test_sha256"]:
        raise RuntimeError("posture evaluator test hash mismatch")
    for filename, key in (
        ("evaluate_naturalbench.py", "candidate_scoring_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("counterfactual_losses.py", "statistics_sha256"),
    ):
        if sha256_file(ROOT / filename) != source[key]:
            raise RuntimeError(f"posture evaluator dependency hash mismatch: {filename}")
    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise RuntimeError("posture crossfit manifest hash mismatch")
    for parent_name, files in protocol["parent_files"].items():
        root = Path(protocol["parents"][parent_name])
        for filename, expected in files.items():
            if sha256_file(root / filename) != expected:
                raise RuntimeError(f"{parent_name} parent mismatch: {filename}")
    return protocol


def score_arm(
    arm_name: str,
    arm: dict[str, Any],
    protocol: dict[str, Any],
    targets: list[dict[str, Any]],
    tokenizer,
    processor,
    vision,
    dtype,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from evaluate_naturalbench import answer_token_mask, collate_candidates
    from modeling import bridge_from_architecture, freeze, vision_features
    from counterfactual_losses import masked_sequence_logprob

    checkpoint = Path(arm["checkpoint"])
    adapter = Path(arm["language_adapter"])
    if sha256_file(checkpoint) != arm["checkpoint_sha256"]:
        raise RuntimeError(f"{arm_name} checkpoint hash mismatch")
    for filename, expected in arm["language_adapter_files"].items():
        if sha256_file(adapter / filename) != expected:
            raise RuntimeError(f"{arm_name} adapter hash mismatch: {filename}")

    language = AutoModelForCausalLM.from_pretrained(
        protocol["parents"]["language"],
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).cuda()
    language = PeftModel.from_pretrained(
        language, adapter, is_trainable=False, local_files_only=True
    )
    freeze(language)
    bridge = bridge_from_architecture(arm["architecture"]).to("cuda", dtype=dtype)
    bridge.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
    freeze(bridge)

    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        view: {condition: [] for condition in CONDITIONS} for view in VIEWS
    }
    batch_targets = int(protocol["evaluation"]["batch_targets"])
    max_tokens = int(protocol["evaluation"]["max_text_tokens"])
    padding = float(protocol["views"]["crop_padding_fraction"])
    layer = int(arm["architecture"].get("vision_feature_layer", -1))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        for view in VIEWS:
            for condition in CONDITIONS:
                for start in range(0, len(targets), batch_targets):
                    batch = targets[start : start + batch_targets]
                    rendered = []
                    for target in batch:
                        if condition == "blank":
                            rendered.append(Image.new("RGB", (224, 224), (127, 127, 127)))
                        else:
                            source = target["image"] if condition == "true_image" else target["donor"]
                            rendered.append(render_view(source, view, padding))
                    pixels = processor(images=rendered, return_tensors="pt")["pixel_values"].to(
                        "cuda", dtype=dtype, non_blocking=True
                    )
                    unique_features = vision_features(vision, pixels, layer)
                    features = unique_features.repeat_interleave(2, dim=0)
                    prompts, answers = [], []
                    for target in batch:
                        prompt = prompt_for(view, target["class_name"])
                        for candidate in CANDIDATES:
                            prompts.append(prompt)
                            answers.append(candidate)
                    input_ids, text_attention, labels = collate_candidates(
                        tokenizer, prompts, answers, max_tokens
                    )
                    input_ids = input_ids.cuda(non_blocking=True)
                    text_attention = text_attention.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    text = language.get_input_embeddings()(input_ids)
                    inputs, attention_mask, targets_with_prefix, readout = bridge.inject_with_readout(
                        text, text_attention, labels, features
                    )
                    if targets_with_prefix is None:
                        raise RuntimeError("candidate targets are missing")
                    logits = language(inputs_embeds=inputs, attention_mask=attention_mask).logits[:, :-1]
                    shifted = targets_with_prefix[:, 1:]
                    token_mask = answer_token_mask(targets_with_prefix, labels, text_attention)
                    if not token_mask.any(dim=1).all():
                        raise RuntimeError("posture candidate has an empty answer span")
                    scores = masked_sequence_logprob(
                        logits,
                        shifted.masked_fill(~token_mask, 0),
                        token_mask,
                    ).reshape(len(batch), 2).cpu()
                    readout_cpu = None
                    if readout is not None:
                        readout_cpu = readout.reshape(len(batch), 2, -1).float().cpu()
                    for index, target in enumerate(batch):
                        left, right = (float(value) for value in scores[index])
                        predicted = None if left == right else CANDIDATES[int(right > left)]
                        expected_index = CANDIDATES.index(target["expected"])
                        correct_score = (left, right)[expected_index]
                        incorrect_score = (right, left)[expected_index]
                        address = None
                        if readout_cpu is not None and condition == "true_image" and view in {"full", "marked"}:
                            candidate_gap = float(
                                (readout_cpu[index, 0] - readout_cpu[index, 1]).abs().max()
                            )
                            mean_attention = readout_cpu[index].mean(dim=0)
                            center, region = bbox_grid_targets(target["image"]["bbox"])
                            prediction = int(mean_attention.argmax())
                            address = {
                                "bbox_center_cell": center,
                                "bbox_region_cells": region,
                                "predicted_cell": prediction,
                                "bbox_center_top1": prediction == center,
                                "bbox_region_top1": prediction in region,
                                "bbox_region_attention_mass": float(mean_attention[region].sum()),
                                "candidate_attention_max_gap": candidate_gap,
                            }
                        records[view][condition].append(
                            {
                                "target_id": target["target_id"],
                                "pair_id": target["pair_id"],
                                "fold": target["fold"],
                                "side": target["side"],
                                "class_name": target["class_name"],
                                "expected": target["expected"],
                                "donor_expected": target["donor"]["answer"],
                                "source_image_id": (
                                    None
                                    if condition == "blank"
                                    else (target["image"] if condition == "true_image" else target["donor"])["image_id"]
                                ),
                                "candidate_sequence_logprobs": {
                                    CANDIDATES[0]: left,
                                    CANDIDATES[1]: right,
                                },
                                "predicted": predicted,
                                "correct": predicted == target["expected"],
                                "correct_minus_incorrect_logprob": correct_score - incorrect_score,
                                "address": address,
                            }
                        )
                    completed = min(start + len(batch), len(targets))
                    if completed == len(targets):
                        print(
                            f"POSTURE_PROGRESS arm={arm_name} view={view} condition={condition} targets={completed}/{len(targets)}",
                            flush=True,
                        )

    metrics = {}
    for view in VIEWS:
        true_records = records[view]["true_image"]
        metrics[view] = {
            "conditions": {
                condition: summarize_condition(records[view][condition], protocol)
                for condition in CONDITIONS
            },
            "true_minus_paired_swap_accuracy_pp": compare_conditions(
                true_records, records[view]["paired_swap"], protocol
            ),
            "true_minus_blank_accuracy_pp": compare_conditions(
                true_records, records[view]["blank"], protocol
            ),
            "strict_bidirectional_answer_flip_percent": strict_bidirectional_flip(
                true_records, records[view]["paired_swap"], protocol
            ),
        }
    result = {
        "arm": arm_name,
        "role": arm["role"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "architecture": arm["architecture"],
        "metrics": metrics,
        "records": records,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    del bridge, language
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare_arms(
    arms: dict[str, dict[str, Any]], left: str, right: str, protocol: dict[str, Any]
) -> dict[str, Any]:
    output = {}
    for view in VIEWS:
        left_records = arms[left]["records"][view]["true_image"]
        right_records = arms[right]["records"][view]["true_image"]
        left_map = {record["target_id"]: record for record in left_records}
        right_map = {record["target_id"]: record for record in right_records}
        ids = sorted(left_map)
        if set(ids) != set(right_map):
            raise ValueError("arm target sets differ")
        output[view] = interval(
            [
                100.0 * (
                    float(left_map[target]["correct"]) - float(right_map[target]["correct"])
                )
                for target in ids
            ],
            [left_map[target]["pair_id"] for target in ids],
            protocol,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-pairs", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(protocol_path)
    pairs = load_jsonl(Path(protocol["data"]["manifest"]))
    if args.limit_pairs is not None:
        if args.limit_pairs < 1:
            raise ValueError("limit-pairs must be positive")
        pairs = sorted(pairs, key=lambda row: row["pair_id"])[: args.limit_pairs]
    targets = flatten_pairs(pairs)
    image_audit = verify_images(targets)
    if not image_audit["all_match_manifest"]:
        raise RuntimeError("posture image integrity audit failed")

    import torch
    from transformers import AutoImageProcessor, AutoTokenizer, SiglipVisionModel

    from modeling import freeze

    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["parents"]["language"], local_files_only=True
    )
    processor = AutoImageProcessor.from_pretrained(
        protocol["parents"]["vision"], local_files_only=True, use_fast=False
    )
    vision = SiglipVisionModel.from_pretrained(
        protocol["parents"]["vision"], dtype=dtype, local_files_only=True
    ).cuda()
    freeze(vision)

    arm_results = {}
    for arm_name in protocol["execution_order"]:
        arm_results[arm_name] = score_arm(
            arm_name,
            protocol["arms"][arm_name],
            protocol,
            targets,
            tokenizer,
            processor,
            vision,
            dtype,
        )
    comparisons = {
        comparison["name"]: {
            "left": comparison["left"],
            "right": comparison["right"],
            "per_image_accuracy_difference_pp": compare_arms(
                arm_results,
                comparison["left"],
                comparison["right"],
                protocol,
            ),
        }
        for comparison in protocol["comparisons"]
    }
    result = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_smoke_only" if args.limit_pairs is not None else "complete_development_only",
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "manifest": {
            "path": protocol["data"]["manifest"],
            "sha256": protocol["data"]["manifest_sha256"],
        },
        "pairs": len(pairs),
        "targets": len(targets),
        "limit_pairs": args.limit_pairs,
        "image_integrity_audit": image_audit,
        "arms": arm_results,
        "comparisons": comparisons,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "completed_at_unix": time.time(),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(args.output, result)
    compact = {
        "status": result["status"],
        "pairs": result["pairs"],
        "targets": result["targets"],
        "image_integrity_audit": image_audit,
        "arms": {
            name: {view: arm["metrics"][view] for view in VIEWS}
            for name, arm in arm_results.items()
        },
        "comparisons": comparisons,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
