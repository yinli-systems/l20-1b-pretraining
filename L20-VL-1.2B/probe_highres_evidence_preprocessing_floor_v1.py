#!/usr/bin/env python3
"""Frozen-vision floor for global-resize versus source-resolution oracle crops."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image


STATUS = "authorized_highres_evidence_preprocessing_floor_probe_only_v1"


def load_rows(manifest: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] in allowed_splits and row["task"] == "read_local_digit"]
    if any(row["expected_action"] != "POINT" for row in selected):
        raise RuntimeError("local digit row is not a POINT example")
    if any(row["target_crop_box_xyxy_normalized"] is None for row in selected):
        raise RuntimeError("local digit row is missing a crop box")
    return selected


def patch_index_at(x: float, y: float, width: float, height: float, grid: int) -> int:
    if not (0 <= x < width and 0 <= y < height and width > 0 and height > 0 and grid > 0):
        raise ValueError("invalid patch coordinate")
    column = min(grid - 1, int(x / width * grid))
    row = min(grid - 1, int(y / height * grid))
    return row * grid + column


def digit_patch_indices(row: dict[str, Any], image_size: int, grid: int) -> tuple[int, int]:
    box = row["target_crop_box_xyxy_normalized"]
    x1, y1, x2, y2 = [value * image_size for value in box]
    cell_width, cell_height = x2 - x1, y2 - y1
    digit_x, digit_y = x1 + 153.0, y1 + 91.0
    global_index = patch_index_at(digit_x, digit_y, image_size, image_size, grid)
    crop_index = patch_index_at(153.0, 91.0, cell_width, cell_height, grid)
    return global_index, crop_index


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, random_state=20260914),
    )
    probe.fit(train_x, train_y)
    return probe.predict(test_x)


def metrics(rows: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, float | int]:
    correct = np.asarray([str(prediction) == row["answer"] for prediction, row in zip(predictions, rows)])
    by_family: dict[str, list[bool]] = defaultdict(list)
    for row, value in zip(rows, correct):
        by_family[row["family_id"]].append(bool(value))
    if any(len(values) != 2 for values in by_family.values()):
        raise RuntimeError("every development family must contain exactly two variants")
    return {
        "rows": len(rows),
        "families": len(by_family),
        "row_accuracy": float(correct.mean()),
        "family_joint_accuracy": float(np.mean([all(values) for values in by_family.values()])),
    }


def paired_family_bootstrap(
    rows: list[dict[str, Any]],
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    family_ids = sorted({row["family_id"] for row in rows})
    indices = {family_id: [index for index, row in enumerate(rows) if row["family_id"] == family_id] for family_id in family_ids}
    a = np.asarray([str(value) == row["answer"] for value, row in zip(prediction_a, rows)], dtype=np.float64)
    b = np.asarray([str(value) == row["answer"] for value, row in zip(prediction_b, rows)], dtype=np.float64)
    family_delta = np.asarray([float((a[indices[key]] - b[indices[key]]).mean()) for key in family_ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(family_ids), size=(samples, len(family_ids)))
    estimates = family_delta[draws].mean(axis=1)
    return {
        "point_estimate": float(family_delta.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": samples,
        "cluster_unit": "scene_family",
    }


def extract_features(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, int]:
    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel

    model_path = protocol["vision_model"]["path"]
    processor = AutoImageProcessor.from_pretrained(
        model_path, local_files_only=True, use_fast=False
    )
    vision = SiglipVisionModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda().eval()
    for parameter in vision.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    global_vectors: list[np.ndarray] = []
    crop_vectors: list[np.ndarray] = []
    root = Path(protocol["data"]["root"])
    batch_size = int(protocol["probe"]["batch_size"])
    grid = int(protocol["probe"]["vision_patch_grid"])
    image_size = int(protocol["data"]["image_size"])
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images, crops, global_indices, crop_indices = [], [], [], []
            for row in batch:
                with Image.open(root / row["image_path"]) as source:
                    image = source.convert("RGB")
                box = tuple(round(value * image.width) for value in row["target_crop_box_xyxy_normalized"])
                images.append(image)
                crops.append(image.crop(box))
                global_index, crop_index = digit_patch_indices(row, image_size, grid)
                global_indices.append(global_index)
                crop_indices.append(crop_index)
            pixels = processor(images=images + crops, return_tensors="pt")["pixel_values"].to(
                device="cuda", dtype=torch.bfloat16
            )
            features = vision(pixel_values=pixels).last_hidden_state.float().cpu().numpy()
            count = len(batch)
            global_vectors.extend(features[index, global_indices[index]] for index in range(count))
            crop_vectors.extend(features[count + index, crop_indices[index]] for index in range(count))
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    return np.asarray(global_vectors), np.asarray(crop_vectors), elapsed, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()

    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("preprocessing floor probe is not authorized")
    for field in ("language_model_access_authorized", "parameter_update_authorized", "sealed_test_access_authorized"):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    root = Path(__file__).resolve().parent
    for path, key in ((Path(__file__), "probe_sha256"), (root / "test_probe_highres_evidence_preprocessing_floor_v1.py", "test_sha256")):
        if sha256_file(path) != protocol["source_code"][key]:
            raise RuntimeError(f"source hash mismatch: {path.name}")
    for group in ("prerequisites", "vision_artifacts"):
        for name, item in protocol[group].items():
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"{group} hash mismatch: {name}")
    receipt_path = Path(protocol["output"]["receipt"])
    cache_path = Path(protocol["output"]["feature_cache"])
    prediction_path = Path(protocol["output"]["development_predictions"])
    for path in (receipt_path, cache_path, prediction_path):
        if path.exists():
            raise FileExistsError(path)

    rows = load_rows(Path(protocol["data"]["manifest"]), {"train", "development"})
    if any(row["split"] == "sealed_test" for row in rows):
        raise RuntimeError("sealed test row reached preprocessing floor probe")
    global_x, crop_x, vision_seconds, peak_memory = extract_features(rows, protocol)
    labels = np.asarray([int(row["answer"]) for row in rows])
    split = np.asarray([row["split"] for row in rows])
    train, development = split == "train", split == "development"
    global_prediction = fit_probe(global_x[train], labels[train], global_x[development])
    crop_prediction = fit_probe(crop_x[train], labels[train], crop_x[development])
    dev_rows = [row for row in rows if row["split"] == "development"]
    global_metrics = metrics(dev_rows, global_prediction)
    crop_metrics = metrics(dev_rows, crop_prediction)
    delta = paired_family_bootstrap(
        dev_rows,
        crop_prediction,
        global_prediction,
        int(protocol["statistics"]["bootstrap_samples"]),
        int(protocol["statistics"]["seed"]),
    )
    gate = protocol["gate"]
    passed = (
        crop_metrics["row_accuracy"] >= float(gate["minimum_crop_row_accuracy"])
        and crop_metrics["family_joint_accuracy"] >= float(gate["minimum_crop_family_joint_accuracy"])
        and global_metrics["row_accuracy"] <= float(gate["maximum_global_row_accuracy"])
        and delta["ci95_low"] > float(gate["minimum_crop_minus_global_ci95_low"])
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        global_features=global_x,
        crop_features=crop_x,
        labels=labels,
        splits=split,
        family_ids=np.asarray([row["family_id"] for row in rows]),
    )
    with prediction_path.open("x") as handle:
        for row, global_value, crop_value in zip(dev_rows, global_prediction, crop_prediction):
            handle.write(json.dumps({
                "family_id": row["family_id"],
                "variant": row["variant"],
                "answer": row["answer"],
                "global_prediction": str(global_value),
                "crop_prediction": str(crop_value),
            }, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_highres_evidence_preprocessing_floor_probe",
        "completed_at": utc_now(),
        "decision": "pass_crop_information_floor" if passed else "reject_or_redesign_crop_information_floor",
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "probe_sha256": sha256_file(Path(__file__)),
        "vision_model": protocol["vision_model"],
        "rows_profiled": len(rows),
        "train_rows": int(train.sum()),
        "development_rows": int(development.sum()),
        "global_224_oracle_digit_patch": global_metrics,
        "source_crop_224_oracle_digit_patch": crop_metrics,
        "crop_minus_global_row_accuracy": delta,
        "gate": gate,
        "gate_passed": passed,
        "vision_forward_seconds": vision_seconds,
        "peak_cuda_memory_bytes": peak_memory,
        "feature_cache": {"path": str(cache_path), "sha256": sha256_file(cache_path), "bytes": cache_path.stat().st_size},
        "development_predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path), "bytes": prediction_path.stat().st_size},
        "language_model_accessed": False,
        "parameter_updates": 0,
        "sealed_test_rows_accessed": 0,
        "next_allowed_step": (
            "Freeze an oracle-crop responder pilot and non-value baselines on train/development only."
            if passed
            else "Do not train the router; change the rendering/resolution/crop design under a new version and repeat the floor test."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({key: receipt[key] for key in ("decision", "global_224_oracle_digit_patch", "source_crop_224_oracle_digit_patch", "crop_minus_global_row_accuracy", "vision_forward_seconds", "peak_cuda_memory_bytes")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
