#!/usr/bin/env python3
"""Pair-atomic out-of-fold probes over frozen SigLIP2 posture features."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from evaluate_openimages_posture_zero_shot_v1 import (
    flatten_pairs,
    load_jsonl,
    render_view,
    sha256_file,
    verify_images,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parent
REPRESENTATIONS = ("full_global", "full_roi", "marked_global", "crop_global")


def patch_region_indices(bbox: list[float], grid: int = 14) -> list[int]:
    x0, x1, y0, y1 = bbox
    indices = []
    for row in range(grid):
        for column in range(grid):
            x = (column + 0.5) / grid
            y = (row + 0.5) / grid
            if x0 <= x <= x1 and y0 <= y <= y1:
                indices.append(row * grid + column)
    if indices:
        return indices
    center_x = min(grid - 1, max(0, int(((x0 + x1) / 2) * grid)))
    center_y = min(grid - 1, max(0, int(((y0 + y1) / 2) * grid)))
    return [center_y * grid + center_x]


def roi_pool(features: np.ndarray, bboxes: list[list[float]], grid: int = 14) -> np.ndarray:
    if features.ndim != 3 or features.shape[1] != grid * grid:
        raise ValueError("features must be [batch, grid*grid, width]")
    if len(bboxes) != features.shape[0]:
        raise ValueError("bbox batch mismatch")
    return np.stack(
        [features[index, patch_region_indices(bbox, grid)].mean(axis=0) for index, bbox in enumerate(bboxes)]
    )


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


def evaluate_predictions(
    targets: list[dict[str, Any]], predictions: np.ndarray, scores: np.ndarray, protocol: dict[str, Any]
) -> dict[str, Any]:
    labels = np.asarray([target["expected"] == "standing" for target in targets], dtype=np.int64)
    correct = predictions == labels
    clusters = [target["pair_id"] for target in targets]
    per_image = interval([100.0 * float(value) for value in correct], clusters, protocol)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[target["pair_id"]].append(index)
    joint_values, order_values, pair_ids = [], [], []
    for pair_id, indices in sorted(grouped.items()):
        if len(indices) != 2 or set(labels[indices]) != {0, 1}:
            raise ValueError(f"invalid paired labels: {pair_id}")
        sitting = next(index for index in indices if labels[index] == 0)
        standing = next(index for index in indices if labels[index] == 1)
        joint_values.append(100.0 * float(correct[indices].all()))
        order_values.append(100.0 * float(scores[standing] > scores[sitting]))
        pair_ids.append(pair_id)
    per_class = {}
    for class_name in sorted({target["class_name"] for target in targets}):
        indices = [index for index, target in enumerate(targets) if target["class_name"] == class_name]
        per_class[class_name] = {
            "images": len(indices),
            "accuracy_percent": 100.0 * float(correct[indices].mean()),
        }
    return {
        "per_image_accuracy_percent": per_image,
        "pair_joint_accuracy_percent": interval(joint_values, pair_ids, protocol),
        "within_pair_standing_score_gt_sitting_percent": interval(order_values, pair_ids, protocol),
        "per_class": per_class,
    }


def crossfit_probe(
    features: np.ndarray, targets: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([target["expected"] == "standing" for target in targets], dtype=np.int64)
    folds = np.asarray([target["fold"] for target in targets], dtype=np.int64)
    predictions = np.full(len(targets), -1, dtype=np.int64)
    scores = np.full(len(targets), np.nan, dtype=np.float64)
    fold_receipts = []
    probe = protocol["probe"]
    for fold in range(int(protocol["data"]["fold_count"])):
        train = folds != fold
        test = folds == fold
        if set(labels[train]) != {0, 1} or set(labels[test]) != {0, 1}:
            raise RuntimeError(f"fold {fold} lacks both labels")
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(probe["logistic_c"]),
                solver=probe["solver"],
                max_iter=int(probe["max_iter"]),
                random_state=int(probe["random_state"]),
            ),
        )
        estimator.fit(features[train], labels[train])
        predictions[test] = estimator.predict(features[test])
        scores[test] = estimator.decision_function(features[test])
        fold_receipts.append(
            {
                "fold": fold,
                "train_images": int(train.sum()),
                "test_images": int(test.sum()),
                "train_pairs": len({targets[index]["pair_id"] for index in np.flatnonzero(train)}),
                "test_pairs": len({targets[index]["pair_id"] for index in np.flatnonzero(test)}),
                "pair_overlap": sorted(
                    {targets[index]["pair_id"] for index in np.flatnonzero(train)}
                    & {targets[index]["pair_id"] for index in np.flatnonzero(test)}
                ),
                "iterations": int(estimator.named_steps["logisticregression"].n_iter_[0]),
            }
        )
    if (predictions < 0).any() or not np.isfinite(scores).all():
        raise RuntimeError("crossfit predictions are incomplete")
    if any(receipt["pair_overlap"] for receipt in fold_receipts):
        raise RuntimeError("pair leakage detected")
    metrics = evaluate_predictions(targets, predictions, scores, protocol)
    records = [
        {
            "target_id": target["target_id"],
            "pair_id": target["pair_id"],
            "fold": target["fold"],
            "class_name": target["class_name"],
            "expected": target["expected"],
            "predicted": "standing" if predictions[index] else "sitting",
            "standing_decision_score": float(scores[index]),
            "correct": bool(predictions[index] == labels[index]),
        }
        for index, target in enumerate(targets)
    ]
    return metrics, records


def extract_features(targets, protocol, processor, vision, dtype) -> dict[str, np.ndarray]:
    import torch

    from modeling import vision_features

    batch_size = int(protocol["feature_extraction"]["batch_images"])
    layer = int(protocol["feature_extraction"]["vision_feature_layer"])
    padding = float(protocol["feature_extraction"]["crop_padding_fraction"])
    collected: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            batch = targets[start : start + batch_size]
            full = [render_view(target["image"], "full", padding) for target in batch]
            marked = [render_view(target["image"], "marked", padding) for target in batch]
            crops = [render_view(target["image"], "crop", padding) for target in batch]
            pixels = processor(images=full + marked + crops, return_tensors="pt")["pixel_values"].to(
                "cuda", dtype=dtype, non_blocking=True
            )
            output = vision_features(vision, pixels, layer).float().cpu().numpy()
            count = len(batch)
            full_features = output[:count]
            marked_features = output[count : 2 * count]
            crop_features = output[2 * count :]
            collected["full_global"].append(full_features.mean(axis=1))
            collected["full_roi"].append(
                roi_pool(full_features, [target["image"]["bbox"] for target in batch])
            )
            collected["marked_global"].append(marked_features.mean(axis=1))
            collected["crop_global"].append(crop_features.mean(axis=1))
            print(
                f"POSTURE_PROBE_FEATURES targets={min(start + count, len(targets))}/{len(targets)} elapsed_seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
    return {name: np.concatenate(values, axis=0) for name, values in collected.items()}


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_frozen_siglip2_posture_probe_v1":
        raise RuntimeError("frozen SigLIP2 posture probe is not authorized")
    if protocol.get("model_parameter_updates_authorized") is not False:
        raise RuntimeError("probe protocol must not authorize model updates")
    source = protocol["source_code"]
    files = {
        "probe_sha256": Path(__file__),
        "test_sha256": ROOT / "test_probe_openimages_posture_siglip2_v1.py",
        "view_renderer_sha256": ROOT / "evaluate_openimages_posture_zero_shot_v1.py",
        "modeling_sha256": ROOT / "modeling.py",
        "statistics_sha256": ROOT / "counterfactual_losses.py",
    }
    for key, source_path in files.items():
        if sha256_file(source_path) != source[key]:
            raise RuntimeError(f"probe source mismatch: {source_path.name}")
    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise RuntimeError("probe manifest hash mismatch")
    vision = Path(protocol["vision_parent"])
    for filename, expected in protocol["vision_parent_files"].items():
        if sha256_file(vision / filename) != expected:
            raise RuntimeError(f"vision parent mismatch: {filename}")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.feature_cache.exists():
        raise FileExistsError("probe outputs must not already exist")
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(protocol_path)
    pairs = load_jsonl(Path(protocol["data"]["manifest"]))
    targets = flatten_pairs(pairs)
    if len(targets) != int(protocol["data"]["targets"]):
        raise RuntimeError("probe target count mismatch")
    image_audit = verify_images(targets)
    if not image_audit["all_match_manifest"]:
        raise RuntimeError("probe image integrity audit failed")

    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel

    from modeling import freeze

    dtype = torch.bfloat16
    processor = AutoImageProcessor.from_pretrained(
        protocol["vision_parent"], local_files_only=True, use_fast=False
    )
    vision = SiglipVisionModel.from_pretrained(
        protocol["vision_parent"], dtype=dtype, local_files_only=True
    ).cuda()
    freeze(vision)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    features = extract_features(targets, protocol, processor, vision, dtype)
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.feature_cache,
        target_ids=np.asarray([target["target_id"] for target in targets]),
        **features,
    )
    cache_hash = sha256_file(args.feature_cache)

    representations = {}
    for name in REPRESENTATIONS:
        metrics, records = crossfit_probe(features[name], targets, protocol)
        representations[name] = {
            "feature_width": int(features[name].shape[1]),
            "metrics": metrics,
            "records": records,
        }
    comparisons = {}
    for left, right in protocol["comparisons"]:
        left_records = {row["target_id"]: row for row in representations[left]["records"]}
        right_records = {row["target_id"]: row for row in representations[right]["records"]}
        ids = sorted(left_records)
        comparisons[f"{left}_minus_{right}"] = interval(
            [
                100.0 * (
                    float(left_records[target]["correct"])
                    - float(right_records[target]["correct"])
                )
                for target in ids
            ],
            [left_records[target]["pair_id"] for target in ids],
            protocol,
        )

    result = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_development_only",
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "manifest": {
            "path": protocol["data"]["manifest"],
            "sha256": protocol["data"]["manifest_sha256"],
        },
        "pairs": len(pairs),
        "targets": len(targets),
        "image_integrity_audit": image_audit,
        "feature_cache": {"path": str(args.feature_cache), "sha256": cache_hash},
        "representations": representations,
        "comparisons": comparisons,
        "model_parameter_updates_performed": False,
        "probe_parameter_updates_performed": True,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.monotonic() - started,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pairs": result["pairs"],
                "targets": result["targets"],
                "representations": {
                    name: value["metrics"] for name, value in representations.items()
                },
                "comparisons": comparisons,
                "peak_allocated_gib": result["peak_allocated_gib"],
                "wall_seconds": result["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
