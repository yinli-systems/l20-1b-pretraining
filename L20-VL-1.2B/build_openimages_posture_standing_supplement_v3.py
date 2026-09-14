#!/usr/bin/env python3
"""Build a disjoint standing-only supplement for exhaustive visual review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import audit_openimages_posture_with_vitpose as pose
import build_openimages_posture_pilot as v1
import build_openimages_posture_candidates_v2 as v2
from render_openimages_posture_audit_detail import panel


ROOT = Path(__file__).resolve().parent
TARGET_CLASSES = {"Boy", "Man", "Woman"}
AMBIGUOUS_POSTURE_PATTERN = re.compile(
    r"\b(crouch(?:ed|ing)?|kneel(?:ed|ing)?|squat(?:ted|ting)?|"
    r"crawl(?:ed|ing)?|lying|laying)\b",
    re.IGNORECASE,
)
OBVIOUS_TRUNCATION_CONTEXT_PATTERN = re.compile(
    r"\b(portraits?|headshots?|selfies?|close[ -]?ups?|podiums?)\b",
    re.IGNORECASE,
)


def validate_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_disjoint_standing_supplement_audit_only":
        raise RuntimeError("standing supplement audit is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("standing supplement protocol must not authorize training")
    if v1.sha256_file(Path(__file__)) != protocol["source_code"]["builder_sha256"]:
        raise RuntimeError("standing supplement builder source hash mismatch")
    test_path = ROOT / "test_build_openimages_posture_standing_supplement_v3.py"
    if v1.sha256_file(test_path) != protocol["source_code"]["test_sha256"]:
        raise RuntimeError("standing supplement test source hash mismatch")

    base_record = protocol["base_protocol"]
    base_path = Path(base_record["path"])
    if v1.sha256_file(base_path) != base_record["sha256"]:
        raise RuntimeError("base posture protocol hash mismatch")
    base = v1.validate_protocol(base_path)

    for name, record in protocol["prerequisites"].items():
        candidate = Path(record["path"])
        if v1.sha256_file(candidate) != record["sha256"]:
            raise RuntimeError(f"prerequisite hash mismatch: {name}")
    audit = json.loads(Path(protocol["prerequisites"]["v2_human_audit"]["path"]).read_text())
    if audit.get("decision") != "reject_openimages_posture_candidates_v2":
        raise RuntimeError("v2 rejection receipt is missing")
    if sorted(audit.get("failing_strata", [])) != ["Boy/standing", "Man/standing", "Woman/standing"]:
        raise RuntimeError("v2 failing-strata diagnosis changed")
    calibration = json.loads(Path(protocol["prerequisites"]["v2_pose_ranking_calibration"]["path"]).read_text())
    if calibration.get("decision") != "use_mean_lower_body_score_as_soft_review_pool_rank_only":
        raise RuntimeError("frozen soft-ranking calibration is missing")
    for name, record in protocol["pose_model"]["artifacts"].items():
        candidate = Path(record["path"])
        if v1.sha256_file(candidate) != record["sha256"]:
            raise RuntimeError(f"pose model artifact hash mismatch: {name}")
    return protocol, base


def load_pair_image_ids(path: Path) -> set[str]:
    excluded: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        excluded.add(pair["image_a"]["image_id"])
        excluded.add(pair["image_b"]["image_id"])
    return excluded


def load_manifest_image_ids(path: Path) -> set[str]:
    return {
        json.loads(line)["image_id"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def passes_standing_filters(row: dict[str, Any], filters: dict[str, Any]) -> tuple[bool, str]:
    if row["class_name"] not in TARGET_CLASSES:
        return False, "non_target_class"
    if row["attribute_name"] != "Stand":
        return False, "non_standing_attribute"
    text = f'{row["caption"]} {row.get("attribution", {}).get("title", "")}'
    if AMBIGUOUS_POSTURE_PATTERN.search(text):
        return False, "ambiguous_posture_term"
    if OBVIOUS_TRUNCATION_CONTEXT_PATTERN.search(text):
        return False, "obvious_truncation_context_term"

    xmin, xmax, ymin, ymax = row["bbox"]
    width, height = xmax - xmin, ymax - ymin
    aspect = height / width
    if width < filters["min_width"]:
        return False, "small_width"
    if width > filters["max_width"]:
        return False, "excessive_width_closeup_risk"
    if height < filters["min_height"]:
        return False, "small_height"
    if width * height < filters["min_area"]:
        return False, "small_area"
    if aspect < filters["min_bbox_height_to_width"]:
        return False, "short_bbox_aspect"
    return True, "pass"


def standing_filter(
    candidates: list[dict[str, Any]],
    excluded: set[str],
    filters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted, counts = [], Counter()
    for row in candidates:
        if row["image_id"] in excluded:
            counts["prior_pilot_image_exclusion"] += 1
            continue
        keep, reason = passes_standing_filters(row, filters)
        counts[reason] += 1
        if keep:
            accepted.append(row)
    counts["standing_filter_pass"] = len(accepted)
    return accepted, dict(sorted(counts.items()))


def select_for_download(protocol: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["class_name"]].append(row)
    if set(grouped) != TARGET_CLASSES:
        raise RuntimeError(f"missing standing strata: {sorted(TARGET_CLASSES - set(grouped))}")
    count = protocol["sampling"]["download_candidates_per_stratum"]
    output = []
    for class_name in sorted(grouped):
        rows = sorted(
            grouped[class_name],
            key=lambda row: v1.stable_hash(
                protocol["seed"], "download", class_name, "Stand", row["image_id"], row["bbox_key"]
            ),
        )
        if len(rows) < count:
            raise RuntimeError(f"insufficient candidates for {class_name}/Stand: {len(rows)} < {count}")
        for row in rows[:count]:
            output.append(
                {
                    **row,
                    "selection_sha256": v1.stable_hash(
                        protocol["seed"], "download", class_name, "Stand", row["image_id"], row["bbox_key"]
                    ),
                }
            )
    return output


def add_pose_ranking_features(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    processor = AutoImageProcessor.from_pretrained(
        protocol["pose_model"]["path"], local_files_only=True, use_fast=False
    )
    device = torch.device(protocol["pose_ranking"]["device"] if torch.cuda.is_available() else "cpu")
    model = VitPoseForPoseEstimation.from_pretrained(
        protocol["pose_model"]["path"], local_files_only=True
    ).to(device).eval()
    torch.manual_seed(protocol["pose_ranking"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol["pose_ranking"]["seed"])

    output = []
    batch_size = protocol["pose_ranking"]["batch_size"]
    threshold = protocol["pose_ranking"]["joint_score_threshold"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
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
                    **row,
                    "curation_pose_features": pose.summarize_pose(
                        keypoints, scores, box[0].tolist(), threshold
                    ),
                }
            )
        for image in images:
            image.close()
    return output


def rank_review_rows(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(row)
    count = protocol["sampling"]["review_candidates_per_stratum"]
    selected = []
    for class_name in sorted(TARGET_CLASSES):
        rows = sorted(
            grouped[class_name],
            key=lambda row: (
                -row["curation_pose_features"]["mean_lower_body_score"],
                v1.stable_hash(protocol["seed"], "review_tie", class_name, row["image_id"]),
            ),
        )
        if len(rows) < count:
            raise RuntimeError(f"review-pool gate failed for {class_name}/Stand: {len(rows)} < {count}")
        selected.extend(rows[:count])
    return sorted(
        selected,
        key=lambda row: (
            row["class_name"],
            -row["curation_pose_features"]["mean_lower_body_score"],
            row["image_id"],
        ),
    )


def choose_review_pool(
    protocol: dict[str, Any], downloaded: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduplicated, duplicate_rejections = v1.deduplicate_images(downloaded)
    ranked = add_pose_ranking_features(protocol, deduplicated)
    return rank_review_rows(protocol, ranked), duplicate_rejections


def render_review_sheets(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    directory = Path(protocol["outputs"]["audit_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    per_page = protocol["human_audit"]["images_per_page"]
    page_records = []
    for page, start in enumerate(range(0, len(rows), per_page), 1):
        page_rows = rows[start : start + per_page]
        sheet = Image.new("RGB", (1440, 320 * len(page_rows)), (228, 232, 238))
        for index, row in enumerate(page_rows):
            tile = panel(row, f'{row["class_name"]} / standing', (1440, 300))
            sheet.paste(tile, (0, index * 320))
            ImageDraw.Draw(sheet).text(
                (6, index * 320 + 302),
                f'{row["image_id"]}  {row["selection_sha256"][:16]}',
                fill="black",
                font=ImageFont.load_default(),
            )
        output = directory / f"openimages-posture-v3-standing-sheet-{page:02d}.png"
        sheet.save(output, optimize=True)
        page_records.append({"path": str(output), "sha256": v1.sha256_file(output), "images": len(page_rows)})
    return page_records


def build(protocol_path: Path) -> dict[str, Any]:
    protocol, base = validate_protocol(protocol_path)
    runtime = {**base, **protocol}
    runtime["limits"] = protocol["limits"]
    runtime["outputs"] = protocol["outputs"]
    root = Path(protocol["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(root).free
    if free_before < protocol["limits"]["minimum_free_bytes_before"]:
        raise RuntimeError(f"insufficient free disk: {free_before}")

    captions = v1.load_captions(Path(base["sources"]["localized_narratives"]["path"]))
    candidates, scan_counts = v1.scan_vrd(base, captions)
    eligible, metadata_counts = v1.join_metadata(base, candidates)
    excluded_v1 = load_pair_image_ids(Path(protocol["prerequisites"]["v1_pair_manifest"]["path"]))
    excluded_v2 = load_manifest_image_ids(Path(protocol["prerequisites"]["v2_candidate_manifest"]["path"]))
    excluded = excluded_v1 | excluded_v2
    strict, strict_counts = standing_filter(eligible, excluded, protocol["standing_filters"])
    requested = select_for_download(protocol, strict)
    downloaded, failures = v1.download_images(runtime, requested)
    review_rows, duplicate_rejections = choose_review_pool(protocol, downloaded)

    selected_ids = {row["image_id"] for row in review_rows}
    removed_files, removed_bytes = 0, 0
    for path in Path(protocol["outputs"]["image_directory"]).glob("*.jpg"):
        if path.stem not in selected_ids:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    manifest = Path(protocol["outputs"]["candidate_manifest"])
    manifest_hash = v2.write_jsonl_atomic(manifest, review_rows)
    sheets = render_review_sheets(protocol, review_rows)
    overlap = selected_ids & excluded
    if overlap:
        raise RuntimeError(f"prior-pilot overlap detected: {sorted(overlap)}")

    receipt = {
        "schema_version": "2026-09-14-v3",
        "status": "complete_pending_exhaustive_target_review",
        "training_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": v1.sha256_file(protocol_path)},
        "builder_sha256": v1.sha256_file(Path(__file__)),
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(root).free,
        "excluded_v1_images": len(excluded_v1),
        "excluded_v2_images": len(excluded_v2),
        "excluded_unique_images": len(excluded),
        "verified_prior_pilot_overlap": len(overlap),
        "pose_model_revision": protocol["pose_model"]["revision"],
        "pose_ranking_feature": protocol["pose_ranking"]["feature"],
        "pose_ranking_role": "soft review-pool ranking only; every selected target still requires visual acceptance",
        "scan_counts": scan_counts,
        "metadata_counts": metadata_counts,
        "standing_filter_counts": strict_counts,
        "download_candidates": len(requested),
        "download_successes": len(downloaded),
        "download_failures": failures,
        "near_or_exact_duplicate_rejections": duplicate_rejections,
        "removed_unselected_files": removed_files,
        "removed_unselected_bytes": removed_bytes,
        "review_images": len(review_rows),
        "review_by_stratum": [
            {"class_name": key, "attribute_name": "Stand", "images": count}
            for key, count in sorted(Counter(row["class_name"] for row in review_rows).items())
        ],
        "candidate_manifest": {"path": str(manifest), "sha256": manifest_hash, "rows": len(review_rows)},
        "audit_sheets": sheets,
        "claim_boundary": "This independent standing supplement remains untrusted until every target is visually reviewed. No pairing, training, model selection, benchmark claim, or release is authorized.",
    }
    v2.write_json_atomic(Path(protocol["outputs"]["receipt"]), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    receipt = build(args.protocol)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "review_images": receipt["review_images"],
                "sheets": len(receipt["audit_sheets"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
