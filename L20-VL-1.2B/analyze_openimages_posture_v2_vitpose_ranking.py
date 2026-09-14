#!/usr/bin/env python3
"""Calibrate a soft ViTPose review-pool rank on the frozen v2 development audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Callable

import audit_openimages_posture_with_vitpose as pose


TARGET_CLASSES = ("Boy", "Man", "Woman")


def score(row: dict) -> float:
    return float(row["pose_features"]["mean_lower_body_score"])


def auc(rows: list[dict], scorer: Callable[[dict], float] = score) -> float:
    positive = [scorer(row) for row in rows if row["human_decision"] == "accept"]
    negative = [scorer(row) for row in rows if row["human_decision"] == "reject"]
    if not positive or not negative:
        raise ValueError("AUROC requires both accepted and rejected rows")
    wins = sum((left > right) + 0.5 * (left == right) for left in positive for right in negative)
    return wins / (len(positive) * len(negative))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def stratified_bootstrap_auc(rows: list[dict], seed: int, replicates: int) -> list[float]:
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(row["class_name"], row["human_decision"])].append(row)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sample = []
        for key in sorted(strata):
            values = strata[key]
            sample.extend(rng.choice(values) for _ in values)
        estimates.append(auc(sample))
    return sorted(estimates)


def percentile(values: list[float], probability: float) -> float:
    index = min(len(values) - 1, max(0, math.floor(probability * len(values))))
    return values[index]


def analyze(rows: list[dict], seed: int, replicates: int) -> dict:
    standing = [
        row for row in rows
        if row["answer"] == "standing" and row["class_name"] in TARGET_CLASSES
    ]
    if len(standing) != 72:
        raise ValueError(f"expected 72 target standing rows, got {len(standing)}")
    boot = stratified_bootstrap_auc(standing, seed, replicates)
    per_class = {}
    for class_name in TARGET_CLASSES:
        values = [row for row in standing if row["class_name"] == class_name]
        ranked = sorted(values, key=lambda row: (-score(row), row["image_id"]))
        top_k = {}
        for count in (8, 12, 16, 20):
            selected = ranked[:count]
            accepted = sum(row["human_decision"] == "accept" for row in selected)
            top_k[str(count)] = {
                "reviewed": count,
                "accepted": accepted,
                "precision": accepted / count,
                "wilson_95_ci": wilson_interval(accepted, count),
            }
        per_class[class_name] = {
            "images": len(values),
            "accepted": sum(row["human_decision"] == "accept" for row in values),
            "auroc": auc(values),
            "top_k": top_k,
        }
    return {
        "development_images": len(standing),
        "accepted": sum(row["human_decision"] == "accept" for row in standing),
        "rejected": sum(row["human_decision"] == "reject" for row in standing),
        "pooled_auroc": auc(standing),
        "stratified_bootstrap_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--replay-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    replay = json.loads(args.replay_receipt.read_text())
    expected_hash = replay["feature_manifest"]["sha256"]
    if pose.sha256_file(args.features) != expected_hash:
        raise RuntimeError("feature manifest hash does not match replay receipt")
    rows = pose.load_jsonl(args.features)
    result = analyze(rows, args.seed, args.bootstrap_replicates)
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_soft_ranking_calibration",
        "training_authorized": False,
        "inputs": {
            "features": {"path": str(args.features), "sha256": pose.sha256_file(args.features)},
            "replay_receipt": {
                "path": str(args.replay_receipt),
                "sha256": pose.sha256_file(args.replay_receipt),
            },
        },
        "ranking_feature": "pose_features.mean_lower_body_score",
        "result": result,
        "decision": "use_mean_lower_body_score_as_soft_review_pool_rank_only",
        "hard_gate_authorized": False,
        "reason": "The ranking signal is strong on the retrospective v2 development census and stable in direction across Boy, Man, and Woman, but top-k confidence intervals remain wide. It may prioritize a disjoint review pool but cannot accept targets without visual review.",
        "next_allowed_step": "Freeze the score before selecting the disjoint v3 review pool; visually audit every selected target and retain all automated rejects outside the training set.",
        "claim_boundary": "This is retrospective curation calibration, not independent validation and not a downstream multimodal model result.",
    }
    pose.write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
