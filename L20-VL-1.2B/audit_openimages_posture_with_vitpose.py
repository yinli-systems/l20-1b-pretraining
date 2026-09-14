#!/usr/bin/env python3
"""Extract target-box pose evidence for the frozen Open Images posture pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
LEG_INDICES = {
    "left": (11, 13, 15),
    "right": (12, 14, 16),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_record(record: dict[str, Any], label: str) -> None:
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"{label} hash mismatch")


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_posture_pose_feature_audit_only":
        raise RuntimeError("pose feature audit is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("pose feature audit must not authorize training")
    if sha256_file(Path(__file__)) != protocol["source_code"]["audit_sha256"]:
        raise RuntimeError("pose feature audit source hash mismatch")
    if sha256_file(ROOT / "test_audit_openimages_posture_with_vitpose.py") != protocol["source_code"]["test_sha256"]:
        raise RuntimeError("pose feature audit test hash mismatch")
    validate_record(protocol["inputs"]["pair_manifest"], "pair manifest")
    validate_record(protocol["inputs"]["human_audit"], "human audit")
    for name, record in protocol["model"]["artifacts"].items():
        validate_record(record, f"model artifact {name}")
    return protocol


def angle_degrees(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    first = a - vertex
    second = c - vertex
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return float("nan")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def segment_verticality(a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    length = float(np.linalg.norm(delta))
    return float("nan") if length == 0 else abs(float(delta[1])) / length


def summarize_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox_xywh: list[float],
    confidence_threshold: float,
) -> dict[str, Any]:
    if keypoints.shape != (17, 2) or scores.shape != (17,):
        raise ValueError("expected COCO 17-keypoint pose output")
    visible = scores >= confidence_threshold
    lower = np.array([11, 12, 13, 14, 15, 16])
    leg_metrics = []
    for side, (hip, knee, ankle) in LEG_INDICES.items():
        if bool(visible[[hip, knee, ankle]].all()):
            leg_metrics.append({
                "side": side,
                "minimum_joint_score": float(scores[[hip, knee, ankle]].min()),
                "thigh_verticality": segment_verticality(keypoints[hip], keypoints[knee]),
                "shank_verticality": segment_verticality(keypoints[knee], keypoints[ankle]),
                "knee_angle_degrees": angle_degrees(keypoints[hip], keypoints[knee], keypoints[ankle]),
                "hip_to_knee_vertical_fraction_of_box": abs(float(keypoints[knee, 1] - keypoints[hip, 1])) / bbox_xywh[3],
                "knee_to_ankle_vertical_fraction_of_box": abs(float(keypoints[ankle, 1] - keypoints[knee, 1])) / bbox_xywh[3],
            })
    aggregate = {}
    for metric in ("thigh_verticality", "shank_verticality", "knee_angle_degrees"):
        values = [row[metric] for row in leg_metrics if math.isfinite(row[metric])]
        aggregate[f"median_{metric}"] = median(values) if values else None
    return {
        "confidence_threshold": confidence_threshold,
        "visible_keypoints": int(visible.sum()),
        "visible_lower_body_keypoints": int(visible[lower].sum()),
        "complete_visible_legs": len(leg_metrics),
        "mean_keypoint_score": float(scores.mean()),
        "mean_lower_body_score": float(scores[lower].mean()),
        "minimum_bilateral_hip_score": float(scores[[11, 12]].min()),
        "minimum_bilateral_knee_score": float(scores[[13, 14]].min()),
        "minimum_bilateral_ankle_score": float(scores[[15, 16]].min()),
        "leg_metrics": leg_metrics,
        **aggregate,
    }


def bbox_to_xywh(row: dict[str, Any], width: int, height: int) -> list[float]:
    xmin, xmax, ymin, ymax = row["bbox"]
    return [xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height]


def extract(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    pairs = load_jsonl(Path(protocol["inputs"]["pair_manifest"]["path"]))
    human = json.loads(Path(protocol["inputs"]["human_audit"]["path"]).read_text())
    rejected_ids = {row["pair_id"] for row in human["rejections"]}
    examples = []
    for pair in pairs:
        for side in ("image_a", "image_b"):
            examples.append({
                "pair_id": pair["pair_id"],
                "pair_human_accepted": pair["pair_id"] not in rejected_ids,
                "side": side,
                **pair[side],
            })
    model_path = protocol["model"]["path"]
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    device = torch.device(protocol["inference"]["device"] if torch.cuda.is_available() else "cpu")
    model = VitPoseForPoseEstimation.from_pretrained(model_path, local_files_only=True).to(device).eval()
    torch.manual_seed(protocol["inference"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol["inference"]["seed"])
    output = []
    batch_size = protocol["inference"]["batch_size"]
    threshold = protocol["feature_extraction"]["joint_score_threshold"]
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        images, boxes, sizes = [], [], []
        for row in batch:
            image = Image.open(row["image_path"]).convert("RGB")
            images.append(image)
            sizes.append((image.height, image.width))
            boxes.append(np.asarray([bbox_to_xywh(row, image.width, image.height)], dtype=np.float32))
        inputs = processor(images=images, boxes=boxes, return_tensors="pt").to(device)
        with torch.inference_mode():
            predictions = model(**inputs)
        poses = processor.post_process_pose_estimation(predictions, boxes=boxes, target_sizes=sizes)
        for row, pose_group, box in zip(batch, poses, boxes):
            if len(pose_group) != 1:
                raise RuntimeError(f"expected exactly one target pose for {row['pair_id']} {row['side']}")
            pose = pose_group[0]
            keypoints = pose["keypoints"].detach().cpu().numpy()
            scores = pose["scores"].detach().cpu().numpy()
            output.append({
                "schema_version": "2026-09-14-v1",
                "pair_id": row["pair_id"],
                "pair_human_accepted": row["pair_human_accepted"],
                "side": row["side"],
                "image_id": row["image_id"],
                "answer": row["answer"],
                "class_name": row["class_name"],
                "bbox_xywh_pixels": [float(value) for value in box[0]],
                "keypoints": [
                    {"name": name, "x": float(point[0]), "y": float(point[1]), "score": float(score)}
                    for name, point, score in zip(KEYPOINT_NAMES, keypoints, scores)
                ],
                "pose_features": summarize_pose(keypoints, scores, box[0].tolist(), threshold),
            })
        for image in images:
            image.close()
    return sorted(output, key=lambda row: (row["pair_id"], row["side"]))


def build_receipt(protocol_path: Path, protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_path = Path(protocol["outputs"]["feature_manifest"])
    output_hash = write_jsonl_atomic(output_path, rows)
    groups = {}
    for accepted in (False, True):
        values = [row for row in rows if row["pair_human_accepted"] is accepted]
        groups["human_accepted" if accepted else "human_rejected"] = {
            "images": len(values),
            "mean_visible_lower_body_keypoints": mean(row["pose_features"]["visible_lower_body_keypoints"] for row in values),
            "mean_lower_body_score": mean(row["pose_features"]["mean_lower_body_score"] for row in values),
            "images_with_complete_visible_leg": sum(row["pose_features"]["complete_visible_legs"] > 0 for row in values),
        }
    return {
        "schema_version": "2026-09-14-v1",
        "status": "complete_feature_extraction_only",
        "training_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "model_revision": protocol["model"]["revision"],
        "examples": len(rows),
        "pairs": len({row["pair_id"] for row in rows}),
        "device": protocol["inference"]["device"],
        "feature_manifest": {"path": str(output_path), "sha256": output_hash, "rows": len(rows)},
        "retrospective_groups": groups,
        "claim_boundary": "These are frozen-pilot pose features for designing a second-pilot gate. They are not a validation result, do not authorize training, and cannot estimate downstream model quality.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = validate_protocol(args.protocol)
    rows = extract(protocol)
    receipt = build_receipt(args.protocol, protocol, rows)
    write_json_atomic(Path(protocol["outputs"]["receipt"]), receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
