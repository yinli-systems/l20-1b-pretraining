#!/usr/bin/env python3
"""Build a caption-corroborated sitting supplement after v1 audit failure."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import build_openimages_posture_confirmation_candidates_v1 as confirmation_v1
import build_openimages_posture_pilot as posture_v1
import build_openimages_posture_standing_supplement_v3 as pose_ranker
from render_openimages_posture_audit_detail import panel


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_openimages_validation_sitting_supplement_audit_only_v2"
TARGET_CLASSES = ("Boy", "Girl", "Man", "Woman")
CLASS_PATTERNS = {
    "Boy": re.compile(r"\b(boy|boys|child|children|kid|kids|toddler|toddlers|baby|babies)\b", re.I),
    "Girl": re.compile(r"\b(girl|girls|child|children|kid|kids|toddler|toddlers|baby|babies)\b", re.I),
    "Man": re.compile(r"\b(man|men|male|gentleman)\b", re.I),
    "Woman": re.compile(r"\b(woman|women|female|lady|ladies)\b", re.I),
}
SITTING_PATTERN = re.compile(r"\b(sit|sits|sitting|seated)\b", re.I)
SUPPORT_PATTERN = re.compile(
    r"\b(chairs?|bench(?:es)?|seats?|sofas?|couch(?:es)?|stools?|wheelchairs?|wagons?|"
    r"bicycles?|bikes?|motorcycles?|horses?|beds?|rocks?|steps?|stairs?|ground|"
    r"floor|laps?|boats?|kayaks?|rowers?|tables?|desks?|highchairs?|swings?|carousels?)\b",
    re.I,
)
AMBIGUOUS_PATTERN = re.compile(
    r"\b(crouch(?:ed|ing)?|kneel(?:ed|ing)?|squat(?:ted|ting)?|"
    r"crawl(?:ed|ing)?|lying|laying)\b",
    re.I,
)


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
    return posture_v1.sha256_file(path)


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("sitting supplement audit is not authorized")
    for field in (
        "training_authorized",
        "model_evaluation_authorized",
        "pairing_authorized",
        "automatic_training_start",
        "automatic_evaluation_start",
    ):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    source = protocol["source_code"]
    checks = (
        (Path(__file__), "builder_sha256"),
        (ROOT / "test_build_openimages_posture_confirmation_sitting_supplement_v2.py", "test_sha256"),
        (ROOT / "build_openimages_posture_confirmation_candidates_v1.py", "confirmation_v1_builder_sha256"),
        (ROOT / "build_openimages_posture_pilot.py", "download_dedup_sha256"),
        (ROOT / "build_openimages_posture_standing_supplement_v3.py", "pose_ranker_sha256"),
        (ROOT / "render_openimages_posture_audit_detail.py", "audit_renderer_sha256"),
    )
    for file_path, key in checks:
        if posture_v1.sha256_file(file_path) != source[key]:
            raise RuntimeError(f"source hash mismatch: {file_path.name}")
    for name, item in protocol["prerequisites"].items():
        if posture_v1.sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"prerequisite hash mismatch: {name}")
    caption_receipt = json.loads(
        Path(protocol["prerequisites"]["caption_acquisition_receipt"]["path"]).read_text()
    )
    if caption_receipt.get("status") != "complete_verified_validation_caption_acquisition_only":
        raise RuntimeError("caption acquisition is incomplete")
    negative = json.loads(Path(protocol["prerequisites"]["v1_negative_audit"]["path"]).read_text())
    if negative.get("decision") != "reject_confirmation_candidate_pool_v1":
        raise RuntimeError("v1 negative audit is missing")
    if negative["reviewed_stratum"].get("class_name") != "Boy" or negative["reviewed_stratum"].get("answer") != "sitting":
        raise RuntimeError("v1 failing stratum changed")
    for name, item in protocol["pose_model"]["artifacts"].items():
        if posture_v1.sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"pose model artifact mismatch: {name}")
    return protocol


def load_captions(path: Path) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        grouped[str(row["image_id"])].append(str(row["caption"]).strip())
    return {image_id: " ".join(values) for image_id, values in grouped.items()}


def caption_evidence(class_name: str, caption: str) -> tuple[bool, int, dict[str, bool]]:
    class_match = bool(CLASS_PATTERNS[class_name].search(caption))
    sitting_match = bool(SITTING_PATTERN.search(caption))
    support_match = bool(SUPPORT_PATTERN.search(caption))
    ambiguous = bool(AMBIGUOUS_PATTERN.search(caption))
    pass_gate = class_match and not ambiguous
    if class_name in {"Man", "Woman"}:
        pass_gate = pass_gate and (sitting_match or support_match)
    tier = 0 if sitting_match and support_match else 1 if sitting_match else 2 if support_match else 3
    return pass_gate, tier, {
        "class_match": class_match,
        "sitting_term": sitting_match,
        "support_term": support_match,
        "ambiguous_posture_term": ambiguous,
    }


def filter_candidates(
    rows: list[dict[str, Any]],
    captions: dict[str, str],
    excluded_ids: set[str],
    max_width: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    counts: Counter[str] = Counter()
    for row in rows:
        if row["attribute_name"] != "Sit":
            continue
        if row["image_id"] in excluded_ids:
            counts["v1_candidate_pool_exclusion"] += 1
            continue
        xmin, xmax, _, _ = row["bbox"]
        if xmax - xmin > max_width:
            counts["excessive_width_closeup_risk"] += 1
            continue
        caption = captions.get(row["image_id"], "")
        if not caption:
            counts["missing_caption"] += 1
            continue
        passed, tier, evidence = caption_evidence(row["class_name"], caption)
        if not passed:
            if evidence["ambiguous_posture_term"]:
                counts["ambiguous_caption_rejection"] += 1
            elif not evidence["class_match"]:
                counts[f"class_caption_rejection_{row['class_name']}"] += 1
            else:
                counts[f"adult_context_rejection_{row['class_name']}"] += 1
            continue
        output.append(
            {
                **row,
                "curation_caption": caption,
                "curation_caption_evidence": evidence,
                "curation_caption_tier": tier,
            }
        )
        counts[f"caption_gate_pass_{row['class_name']}"] += 1
    return output, dict(sorted(counts.items()))


def assign_by_scarcity(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["image_id"], row["class_name"])
        current = best.get(key)
        if current is None or (
            row["curation_caption_tier"], -row["bbox_area"], row["bbox_key"]
        ) < (
            current["curation_caption_tier"], -current["bbox_area"], current["bbox_key"]
        ):
            best[key] = row
    class_counts = Counter(row["class_name"] for row in best.values())
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in best.values():
        by_image[row["image_id"]].append(row)
    assigned = []
    conflicts = 0
    for image_id, values in sorted(by_image.items()):
        if len(values) > 1:
            conflicts += 1
        assigned.append(
            min(
                values,
                key=lambda row: (
                    class_counts[row["class_name"]],
                    row["class_name"],
                    posture_v1.stable_hash(
                        seed, "scarcity_tie", image_id, row["class_name"], row["bbox_key"]
                    ),
                ),
            )
        )
    return assigned, {
        "multi_class_images_resolved": conflicts,
        **{f"assigned_{name}": sum(row["class_name"] == name for row in assigned) for name in TARGET_CLASSES},
    }


def select_for_download(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(row)
    if set(grouped) != set(TARGET_CLASSES):
        raise RuntimeError(f"missing sitting classes: {sorted(set(TARGET_CLASSES) - set(grouped))}")
    selected = []
    for class_name in TARGET_CLASSES:
        values = sorted(
            grouped[class_name],
            key=lambda row: (
                row["curation_caption_tier"],
                posture_v1.stable_hash(
                    protocol["seed"], "download", class_name, row["image_id"], row["bbox_key"]
                ),
            ),
        )
        quota = int(protocol["sampling"]["download_candidates_per_class"][class_name])
        if len(values) < quota:
            raise RuntimeError(f"insufficient {class_name} candidates: {len(values)} < {quota}")
        selected.extend(
            {
                **row,
                "selection_sha256": posture_v1.stable_hash(
                    protocol["seed"], "download", class_name, row["image_id"], row["bbox_key"]
                ),
            }
            for row in values[:quota]
        )
    return selected


def candidate_fingerprints(path: Path) -> tuple[set[str], set[str], list[int]]:
    ids, hashes, dhashes = set(), set(), []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.add(row["image_id"])
        hashes.add(row["image_sha256"])
        dhashes.append(int(row["dhash64"], 16))
    return ids, hashes, dhashes


def reject_candidate_pool_overlap(
    rows: list[dict[str, Any]], prior_manifest: Path, max_hamming: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids, hashes, dhashes = candidate_fingerprints(prior_manifest)
    kept, rejected = [], []
    for row in rows:
        reason = None
        if row["image_id"] in ids:
            reason = "image_id"
        elif row["image_sha256"] in hashes:
            reason = "exact_sha256"
        else:
            distance = min(bin(int(row["dhash64"], 16) ^ prior).count("1") for prior in dhashes)
            if distance <= max_hamming:
                reason = f"prior_pool_dhash_hamming_{distance}"
        if reason is None:
            kept.append(row)
        else:
            rejected.append({"image_id": row["image_id"], "reason": reason})
    return kept, rejected


def choose_review_pool(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deduplicated, within_rejections = posture_v1.deduplicate_images(rows)
    development_kept, development_rejections = confirmation_v1.reject_development_overlap(
        deduplicated,
        Path(protocol["prerequisites"]["development_manifest"]["path"]),
        int(protocol["deduplication"]["max_dhash_hamming"]),
    )
    prior_kept, prior_rejections = reject_candidate_pool_overlap(
        development_kept,
        Path(protocol["prerequisites"]["v1_candidate_manifest"]["path"]),
        int(protocol["deduplication"]["max_dhash_hamming"]),
    )
    ranked = pose_ranker.add_pose_ranking_features(protocol, prior_kept)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        grouped[row["class_name"]].append(row)
    selected = []
    for class_name in TARGET_CLASSES:
        values = sorted(
            grouped[class_name],
            key=lambda row: (
                row["curation_caption_tier"],
                -row["curation_pose_features"]["mean_lower_body_score"],
                posture_v1.stable_hash(protocol["seed"], "review_tie", class_name, row["image_id"]),
            ),
        )
        quota = int(protocol["sampling"]["review_candidates_per_class"][class_name])
        if len(values) < quota:
            raise RuntimeError(f"post-dedup {class_name} review gate failed: {len(values)} < {quota}")
        selected.extend(values[:quota])
    return (
        sorted(selected, key=lambda row: (row["class_name"], row["image_id"])),
        {
            "within_supplement_rejections": within_rejections,
            "development_overlap_rejections": development_rejections,
            "v1_candidate_pool_overlap_rejections": prior_rejections,
        },
    )


def render_review_sheets(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    directory = Path(protocol["outputs"]["audit_directory"])
    directory.mkdir(parents=True)
    per_page = int(protocol["human_audit"]["images_per_page"])
    records = []
    for page_number, start in enumerate(range(0, len(rows), per_page), 1):
        page_rows = rows[start : start + per_page]
        sheet = Image.new("RGB", (1440, 320 * len(page_rows)), (228, 232, 238))
        for index, row in enumerate(page_rows):
            tile = panel(row, f"{row['class_name']} / sitting", (1440, 300))
            sheet.paste(tile, (0, index * 320))
            ImageDraw.Draw(sheet).text(
                (6, index * 320 + 302),
                f"{row['image_id']}  {row['selection_sha256'][:16]}",
                fill="black",
                font=ImageFont.load_default(),
            )
        output = directory / f"openimages-posture-confirmation-sitting-v2-sheet-{page_number:02d}.png"
        sheet.save(output, optimize=True)
        records.append({"path": str(output), "sha256": posture_v1.sha256_file(output), "images": len(page_rows)})
    return records


def build(protocol_path: Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    protected = [Path(protocol["outputs"][key]) for key in ("root", "audit_directory", "receipt")]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"sitting supplement output exists: {existing}")
    root = Path(protocol["outputs"]["root"])
    free_before = shutil.disk_usage(root.parent).free
    if free_before < int(protocol["limits"]["minimum_free_bytes_before"]):
        raise RuntimeError(f"insufficient free disk: {free_before}")
    root.mkdir(parents=True)

    candidates, scan_counts = confirmation_v1.scan_vrd(protocol)
    eligible, metadata_counts = confirmation_v1.join_metadata(protocol, candidates)
    captions = load_captions(Path(protocol["sources"]["captions"]["path"]))
    excluded_ids = {
        json.loads(line)["image_id"]
        for line in Path(protocol["prerequisites"]["v1_candidate_manifest"]["path"]).read_text().splitlines()
        if line
    }
    filtered, caption_counts = filter_candidates(
        eligible,
        captions,
        excluded_ids,
        float(protocol["geometry_filters"]["Sit"]["max_width"]),
    )
    assigned, assignment_counts = assign_by_scarcity(filtered, int(protocol["seed"]))
    requested = select_for_download(protocol, assigned)
    downloaded, failures = confirmation_v1.download_images(protocol, requested)
    review_rows, overlap_audit = choose_review_pool(protocol, downloaded)

    selected_ids = {row["image_id"] for row in review_rows}
    image_directory = Path(protocol["outputs"]["image_directory"])
    removed_files, removed_bytes = 0, 0
    for path in image_directory.glob("*.jpg"):
        if path.stem not in selected_ids:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    manifest = Path(protocol["outputs"]["candidate_manifest"])
    manifest_sha256 = write_jsonl_atomic(manifest, review_rows)
    sheets = render_review_sheets(protocol, review_rows)
    by_class = Counter(row["class_name"] for row in review_rows)
    receipt = {
        "schema_version": "2026-09-14-v2",
        "status": "complete_pending_exhaustive_primary_agent_and_independent_human_review",
        "training_authorized": False,
        "model_evaluation_authorized": False,
        "pairing_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": posture_v1.sha256_file(protocol_path)},
        "builder_sha256": posture_v1.sha256_file(Path(__file__)),
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(root).free,
        "scan_counts": scan_counts,
        "metadata_counts": metadata_counts,
        "caption_filter_counts": caption_counts,
        "scarcity_assignment_counts": assignment_counts,
        "download_candidates": len(requested),
        "download_successes": len(downloaded),
        "download_failures": failures,
        "overlap_audit": overlap_audit,
        "removed_unselected_files": removed_files,
        "removed_unselected_bytes": removed_bytes,
        "review_images": len(review_rows),
        "review_by_class": dict(sorted(by_class.items())),
        "caption_visible_to_reviewer": False,
        "pose_score_visible_to_reviewer": False,
        "model_under_test_accessed": False,
        "candidate_manifest": {"path": str(manifest), "sha256": manifest_sha256, "rows": len(review_rows)},
        "audit_sheets": sheets,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(Path(protocol["outputs"]["receipt"]), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    receipt = build(args.protocol)
    print(json.dumps({
        "status": receipt["status"],
        "review_images": receipt["review_images"],
        "review_by_class": receipt["review_by_class"],
        "download_failures": len(receipt["download_failures"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
