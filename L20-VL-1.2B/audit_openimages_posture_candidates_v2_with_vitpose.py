#!/usr/bin/env python3
"""Replay ViTPose features on the exhaustively audited posture v2 targets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

import audit_openimages_posture_with_vitpose as pose


ROOT = Path(__file__).resolve().parent


def validate_record(record: dict[str, Any], label: str) -> None:
    path = Path(record["path"])
    if not path.is_file() or pose.sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"{label} hash mismatch")


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_posture_candidates_v2_pose_feature_replay_only":
        raise RuntimeError("v2 pose feature replay is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("pose feature replay must not authorize training")
    if pose.sha256_file(Path(__file__)) != protocol["source_code"]["audit_sha256"]:
        raise RuntimeError("v2 pose replay source hash mismatch")
    test_path = ROOT / "test_audit_openimages_posture_candidates_v2_with_vitpose.py"
    if pose.sha256_file(test_path) != protocol["source_code"]["test_sha256"]:
        raise RuntimeError("v2 pose replay test source hash mismatch")
    for name, record in protocol["inputs"].items():
        validate_record(record, name)
    for name, record in protocol["model"]["artifacts"].items():
        validate_record(record, f"model artifact {name}")
    audit = json.loads(Path(protocol["inputs"]["human_audit"]["path"]).read_text())
    if audit.get("decision") != "reject_openimages_posture_candidates_v2":
        raise RuntimeError("v2 human-audit rejection receipt is missing")
    return protocol


def merge_examples(manifest: list[dict], decisions: list[dict]) -> list[dict]:
    if len(manifest) != 192 or len(decisions) != 192:
        raise ValueError("expected exactly 192 manifest rows and 192 decisions")
    by_id = {row["image_id"]: row for row in decisions}
    if len(by_id) != 192:
        raise ValueError("human decisions contain duplicate image ids")
    manifest_ids = {row["image_id"] for row in manifest}
    if manifest_ids != set(by_id):
        raise ValueError("candidate and human-decision image ids differ")
    output = []
    for row in manifest:
        decision = by_id[row["image_id"]]
        if decision["class_name"] != row["class_name"] or decision["posture"] != row["answer"]:
            raise ValueError(f"decision metadata mismatch for {row['image_id']}")
        output.append({**row, "human_decision": decision["decision"], "human_reason": decision["reason"]})
    return output


def extract(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    manifest = pose.load_jsonl(Path(protocol["inputs"]["candidate_manifest"]["path"]))
    decisions = pose.load_jsonl(Path(protocol["inputs"]["human_decisions"]["path"]))
    examples = merge_examples(manifest, decisions)
    processor = AutoImageProcessor.from_pretrained(protocol["model"]["path"], local_files_only=True, use_fast=False)
    device = torch.device(protocol["inference"]["device"] if torch.cuda.is_available() else "cpu")
    model = VitPoseForPoseEstimation.from_pretrained(protocol["model"]["path"], local_files_only=True).to(device).eval()
    torch.manual_seed(protocol["inference"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol["inference"]["seed"])

    output = []
    threshold = protocol["feature_extraction"]["joint_score_threshold"]
    batch_size = protocol["inference"]["batch_size"]
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        images, boxes, sizes = [], [], []
        for row in batch:
            image = Image.open(row["image_path"]).convert("RGB")
            images.append(image)
            sizes.append((image.height, image.width))
            boxes.append(np.asarray([pose.bbox_to_xywh(row, image.width, image.height)], dtype=np.float32))
        inputs = processor(images=images, boxes=boxes, return_tensors="pt").to(device)
        with torch.inference_mode():
            predictions = model(**inputs)
        groups = processor.post_process_pose_estimation(predictions, boxes=boxes, target_sizes=sizes)
        for row, group, box in zip(batch, groups, boxes):
            if len(group) != 1:
                raise RuntimeError(f"expected one target pose for {row['image_id']}")
            keypoints = group[0]["keypoints"].detach().cpu().numpy()
            scores = group[0]["scores"].detach().cpu().numpy()
            output.append(
                {
                    "schema_version": "2026-09-14-v2",
                    "image_id": row["image_id"],
                    "class_name": row["class_name"],
                    "answer": row["answer"],
                    "human_decision": row["human_decision"],
                    "human_reason": row["human_reason"],
                    "bbox": row["bbox"],
                    "bbox_xywh_pixels": [float(value) for value in box[0]],
                    "pose_features": pose.summarize_pose(keypoints, scores, box[0].tolist(), threshold),
                }
            )
        for image in images:
            image.close()
    return sorted(output, key=lambda row: row["image_id"])


def build_receipt(protocol_path: Path, protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_path = Path(protocol["outputs"]["feature_manifest"])
    output_hash = pose.write_jsonl_atomic(output_path, rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[f'{row["answer"]}/{row["human_decision"]}'].append(row)
    summaries = {}
    for key, values in sorted(grouped.items()):
        summaries[key] = {
            "images": len(values),
            "mean_visible_lower_body_keypoints": mean(
                row["pose_features"]["visible_lower_body_keypoints"] for row in values
            ),
            "mean_lower_body_score": mean(row["pose_features"]["mean_lower_body_score"] for row in values),
            "images_with_complete_visible_leg": sum(
                row["pose_features"]["complete_visible_legs"] > 0 for row in values
            ),
        }
    return {
        "schema_version": "2026-09-14-v2",
        "status": "complete_feature_replay_only",
        "training_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": pose.sha256_file(protocol_path)},
        "model_revision": protocol["model"]["revision"],
        "device": protocol["inference"]["device"],
        "examples": len(rows),
        "feature_manifest": {"path": str(output_path), "sha256": output_hash, "rows": len(rows)},
        "retrospective_groups": summaries,
        "claim_boundary": "This is retrospective curation analysis on a development audit. It cannot automatically accept any image, validate the data, authorize training, or support downstream model claims.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = validate_protocol(args.protocol)
    rows = extract(protocol)
    receipt = build_receipt(args.protocol, protocol, rows)
    pose.write_json_atomic(Path(protocol["outputs"]["receipt"]), receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
