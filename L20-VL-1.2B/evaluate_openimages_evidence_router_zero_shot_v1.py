#!/usr/bin/env python3
"""Zero-shot natural-image localization diagnostic for the frozen evidence router."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from safetensors.torch import load_file

from evaluate_openimages_posture_zero_shot_v1 import render_view
from evidence_acquisition import EvidenceAcquisitionRouter, choose_evidence_action
from modeling import freeze, vision_features
from train_highres_evidence_router_v1 import collate_true_answers
from train_stage_a_full_token import release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_openimages_evidence_router_zero_shot_diagnostic_v1"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        raise ValueError("total must be positive")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": probability,
        "lower_95_ci": max(0.0, center - radius),
        "upper_95_ci": min(1.0, center + radius),
    }


def bbox_targets(bbox: list[float], grid: int = 14) -> tuple[int, set[int]]:
    x0, x1, y0, y1 = bbox
    center_x = min(grid - 1, max(0, int(((x0 + x1) / 2.0) * grid)))
    center_y = min(grid - 1, max(0, int(((y0 + y1) / 2.0) * grid)))
    center = center_y * grid + center_x
    region = {
        row * grid + column
        for row in range(grid)
        for column in range(grid)
        if x0 <= (column + 0.5) / grid <= x1 and y0 <= (row + 0.5) / grid <= y1
    }
    return center, region or {center}


def localization_flags(action: int, center: int, region: set[int], grid: int = 14) -> dict[str, bool | int]:
    action_y, action_x = divmod(action, grid)
    center_y, center_x = divmod(center, grid)
    error = max(abs(action_y - center_y), abs(action_x - center_x))
    return {
        "exact_center": action == center,
        "within_one_center": error <= 1,
        "inside_bbox": action in region,
        "chebyshev_center_error": error,
    }


def summarize(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    total = len(records)
    return {
        "exact_center": wilson_interval(sum(record[prefix]["exact_center"] for record in records), total),
        "within_one_center": wilson_interval(sum(record[prefix]["within_one_center"] for record in records), total),
        "inside_bbox": wilson_interval(sum(record[prefix]["inside_bbox"] for record in records), total),
        "mean_chebyshev_center_error": sum(record[prefix]["chebyshev_center_error"] for record in records) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise SystemExit("natural-image zero-shot diagnostic is not authorized")
    if protocol.get("parameter_update_authorized") is not False:
        raise SystemExit("parameter updates must remain disabled")
    for path, key in (
        (Path(__file__), "evaluator_sha256"),
        (ROOT / "test_evaluate_openimages_evidence_router_zero_shot_v1.py", "test_sha256"),
        (ROOT / "evaluate_openimages_posture_zero_shot_v1.py", "posture_evaluator_sha256"),
        (ROOT / "evidence_acquisition.py", "router_module_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
        (ROOT / "train_highres_evidence_router_v1.py", "router_trainer_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"prerequisite hash mismatch: {name}")
    output = Path(protocol["output"]["receipt"])
    predictions = Path(protocol["output"]["predictions"])
    if output.exists() or predictions.exists():
        raise FileExistsError("diagnostic output already exists")

    rows = [json.loads(line) for line in Path(protocol["data"]["manifest"]).read_text().splitlines() if line.strip()]
    if len(rows) != int(protocol["data"]["expected_rows"]):
        raise RuntimeError("unexpected target row count")
    verified = {}
    for row in rows:
        path = Path(row["image_path"])
        actual = sha256_file(path)
        if actual != row["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {row['image_id']}")
        verified[row["image_id"]] = actual
    if len(verified) != len(rows):
        raise RuntimeError("target images are not unique")

    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

    parents = protocol["parents"]
    language_root = Path(parents["language"])
    if sha256_file(language_root / "release-manifest.json") != parents["language_release_manifest_sha256"]:
        raise SystemExit("language release manifest mismatch")
    language_records = release_records(language_root)
    vision_root = Path(parents["vision"])
    for name, expected in parents["vision_artifacts"].items():
        if sha256_file(vision_root / name) != expected:
            raise SystemExit(f"vision artifact mismatch: {name}")
    checkpoint = protocol["checkpoint"]
    if sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise SystemExit("router checkpoint mismatch")

    dtype = torch.bfloat16
    processor = AutoImageProcessor.from_pretrained(vision_root, local_files_only=True, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(language_root, local_files_only=True)
    vision = SiglipVisionModel.from_pretrained(vision_root, dtype=dtype, local_files_only=True).cuda()
    language = AutoModelForCausalLM.from_pretrained(
        language_root, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    freeze(vision)
    freeze(language)
    router = EvidenceAcquisitionRouter(
        vision_dim=int(protocol["router"]["vision_dim"]),
        language_dim=int(protocol["router"]["language_dim"]),
        rank=int(protocol["router"]["rank"]),
    ).to("cuda", dtype=dtype)
    router.load_state_dict(load_file(checkpoint["path"], device="cpu"), strict=True)
    freeze(router)

    prompted = [
        {
            **row,
            "question": (
                f"Which posture is shown by the {row['class_name'].lower()} inside the red box: "
                "sitting or standing?"
            ),
        }
        for row in rows
    ]
    input_ids, attention, labels = collate_true_answers(
        tokenizer, prompted, int(protocol["evaluation"]["max_text_tokens"])
    )
    batch_size = int(protocol["evaluation"]["batch_rows"])
    records = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(prompted), batch_size):
            batch = prompted[start:start + batch_size]
            images = [
                render_view(row, "marked", float(protocol["evaluation"]["box_padding_fraction"]))
                for row in batch
            ]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda", dtype=dtype)
            features = vision_features(
                vision, pixels, int(protocol["architecture"]["vision_feature_layer"])
            )
            ids = input_ids[start:start + len(batch)].cuda()
            text = language.get_input_embeddings()(ids)
            attn = attention[start:start + len(batch)].cuda()
            target_labels = labels[start:start + len(batch)].cuda()
            logits, predicted_gain = router(features, text, attn, target_labels)
            raw_pointer = logits[:, :196].argmax(dim=-1)
            deployed = choose_evidence_action(
                logits,
                predicted_gain,
                float(protocol["router"]["minimum_predicted_gain"]),
                float(protocol["router"]["minimum_action_probability"]),
            )
            probabilities = torch.softmax(logits.float(), dim=-1)
            for row, raw, action, gain, probability in zip(
                batch,
                raw_pointer.cpu().tolist(),
                deployed.cpu().tolist(),
                predicted_gain.cpu().tolist(),
                probabilities.max(dim=-1).values.cpu().tolist(),
            ):
                center, region = bbox_targets(row["bbox"], 14)
                deployed_patch = raw if action == 196 else action
                records.append({
                    "image_id": row["image_id"],
                    "class_name": row["class_name"],
                    "answer": row["answer"],
                    "bbox": row["bbox"],
                    "target_center_patch": center,
                    "target_region_patches": sorted(region),
                    "raw_pointer_action": raw,
                    "deployed_action": action,
                    "deployed_acquired": action != 196,
                    "deployed_probability": probability,
                    "predicted_gain": gain,
                    "raw": localization_flags(raw, center, region),
                    "deployed_or_raw_if_stop": localization_flags(deployed_patch, center, region),
                })

    with predictions.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    acquired = sum(record["deployed_acquired"] for record in records)
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_openimages_evidence_router_zero_shot_diagnostic",
        "completed_at": utc_now(),
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "checkpoint": checkpoint,
        "verified_images": len(verified),
        "prompt_template": "Which posture is shown by the {class} inside the red box: sitting or standing?",
        "metrics": {
            "raw_pointer": summarize(records, "raw"),
            "deployed_action": {
                "acquisition_rate": wilson_interval(acquired, len(records)),
                "localization_if_stop_is_replaced_by_raw_pointer": summarize(records, "deployed_or_raw_if_stop"),
            },
            "predicted_gain": {
                "minimum": min(record["predicted_gain"] for record in records),
                "mean": sum(record["predicted_gain"] for record in records) / len(records),
                "maximum": max(record["predicted_gain"] for record in records),
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "verified_parents": {
            "language": language_records,
            "vision_artifacts": parents["vision_artifacts"],
        },
        "predictions": {
            "path": str(predictions),
            "sha256": sha256_file(predictions),
            "bytes": predictions.stat().st_size,
        },
        "parameter_updates": 0,
        "decision": "diagnostic_only_no_training_decision",
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(output, receipt)
    print(json.dumps({
        "metrics": receipt["metrics"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "peak_cuda_memory_bytes": receipt["peak_cuda_memory_bytes"],
        "decision": receipt["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
