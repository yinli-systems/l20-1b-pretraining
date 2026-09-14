#!/usr/bin/env python3
"""Build a disjoint, stricter Open Images posture candidate pool for full review."""
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

import build_openimages_posture_pilot as v1
from render_openimages_posture_audit_detail import panel


ROOT = Path(__file__).resolve().parent
AMBIGUOUS_POSTURE_PATTERN = re.compile(
    r"\b(crouch(?:ed|ing)?|kneel(?:ed|ing)?|squat(?:ted|ting)?|"
    r"crawl(?:ed|ing)?|lying|laying)\b",
    re.IGNORECASE,
)
SEATED_SUPPORT_PATTERN = re.compile(
    r"\b(chairs?|benches?|seats?|sofas?|couches?|stools?|tables?|desks?|"
    r"wheelchairs?|wagons?|bicycles?|bikes?|motorcycles?|horses?|beds?|"
    r"rocks?|steps?|stairs?|ground|floor|laps?|boats?|kayaks?|rowers?)\b",
    re.IGNORECASE,
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
    return v1.sha256_file(path)


def validate_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_disjoint_posture_candidate_audit_only":
        raise RuntimeError("posture candidate audit is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("posture candidate audit must not authorize training")
    if v1.sha256_file(Path(__file__)) != protocol["source_code"]["builder_sha256"]:
        raise RuntimeError("candidate builder source hash mismatch")
    test_path = ROOT / "test_build_openimages_posture_candidates_v2.py"
    if v1.sha256_file(test_path) != protocol["source_code"]["test_sha256"]:
        raise RuntimeError("candidate builder test hash mismatch")
    base_record = protocol["base_protocol"]
    base_path = Path(base_record["path"])
    if v1.sha256_file(base_path) != base_record["sha256"]:
        raise RuntimeError("base posture protocol hash mismatch")
    base = v1.validate_protocol(base_path)
    for name in ("first_pilot_pairs", "first_pilot_human_audit", "pose_gate_calibration"):
        record = protocol["prerequisites"][name]
        candidate = Path(record["path"])
        if v1.sha256_file(candidate) != record["sha256"]:
            raise RuntimeError(f"prerequisite hash mismatch: {name}")
    if json.loads(Path(protocol["prerequisites"]["first_pilot_human_audit"]["path"]).read_text())["decision"] != "reject_openimages_posture_pilot_v1":
        raise RuntimeError("first pilot was not formally rejected")
    if json.loads(Path(protocol["prerequisites"]["pose_gate_calibration"]["path"]).read_text())["decision"] != "do_not_use_vitpose_as_an_automatic_hard_gate":
        raise RuntimeError("pose hard-gate rejection is missing")
    return protocol, base


def load_excluded_image_ids(path: Path) -> set[str]:
    excluded = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        excluded.add(pair["image_a"]["image_id"])
        excluded.add(pair["image_b"]["image_id"])
    return excluded


def passes_strict_filters(row: dict[str, Any], filters: dict[str, Any]) -> tuple[bool, str]:
    if AMBIGUOUS_POSTURE_PATTERN.search(row["caption"]):
        return False, "ambiguous_posture_term"
    if row["attribute_name"] == "Sit" and not SEATED_SUPPORT_PATTERN.search(row["caption"]):
        return False, "sitting_without_support_term"
    xmin, xmax, ymin, ymax = row["bbox"]
    width, height = xmax - xmin, ymax - ymin
    if width < filters["min_edge"] or height < filters["min_edge"]:
        return False, "small_edge"
    if width * height < filters["min_area"]:
        return False, "small_area"
    return True, "pass"


def strict_filter(
    candidates: list[dict[str, Any]],
    excluded: set[str],
    filters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted, counts = [], Counter()
    for row in candidates:
        if row["image_id"] in excluded:
            counts["first_pilot_image_exclusion"] += 1
            continue
        keep, reason = passes_strict_filters(row, filters)
        counts[reason] += 1
        if keep:
            accepted.append(row)
    counts["strict_filter_pass"] = len(accepted)
    return accepted, dict(sorted(counts.items()))


def select_for_download(protocol: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["class_name"], row["attribute_name"])].append(row)
    expected = {(name, attribute) for name in v1.CLASS_PATTERNS for attribute in v1.ATTRIBUTE_PATTERNS}
    if set(grouped) != expected:
        raise RuntimeError(f"missing strict strata: {sorted(expected - set(grouped))}")
    count = protocol["sampling"]["download_candidates_per_stratum"]
    output = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda row: v1.stable_hash(protocol["seed"], "download", *key, row["image_id"], row["bbox_key"]),
        )
        if len(rows) < count:
            raise RuntimeError(f"insufficient strict candidates for {key}: {len(rows)} < {count}")
        for row in rows[:count]:
            output.append({
                **row,
                "selection_sha256": v1.stable_hash(
                    protocol["seed"], "download", *key, row["image_id"], row["bbox_key"]
                ),
            })
    return output


def choose_review_pool(protocol: dict[str, Any], downloaded: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduplicated, duplicate_rejections = v1.deduplicate_images(downloaded)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated:
        grouped[(row["class_name"], row["attribute_name"])].append(row)
    count = protocol["sampling"]["review_candidates_per_stratum"]
    selected = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: v1.stable_hash(protocol["seed"], "review", *key, row["image_id"]))
        if len(rows) < count:
            raise RuntimeError(f"post-download review-pool gate failed for {key}: {len(rows)} < {count}")
        selected.extend(rows[:count])
    return sorted(selected, key=lambda row: (row["class_name"], row["attribute_name"], row["image_id"])), duplicate_rejections


def render_review_sheets(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    directory = Path(protocol["outputs"]["audit_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    per_page = protocol["human_audit"]["images_per_page"]
    page_records = []
    for page, start in enumerate(range(0, len(rows), per_page), 1):
        page_rows = rows[start : start + per_page]
        sheet = Image.new("RGB", (1440, 320 * len(page_rows)), (228, 232, 238))
        for index, row in enumerate(page_rows):
            tile = panel(row, f"{row['class_name']} / {row['answer']}", (1440, 300))
            sheet.paste(tile, (0, index * 320))
            ImageDraw.Draw(sheet).text(
                (6, index * 320 + 302),
                f"{row['image_id']}  {row['selection_sha256'][:16]}",
                fill="black",
                font=ImageFont.load_default(),
            )
        output = directory / f"openimages-posture-v2-candidates-sheet-{page:02d}.png"
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
    excluded = load_excluded_image_ids(Path(protocol["prerequisites"]["first_pilot_pairs"]["path"]))
    strict, strict_counts = strict_filter(eligible, excluded, protocol["strict_filters"])
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
    manifest_hash = write_jsonl_atomic(manifest, review_rows)
    sheets = render_review_sheets(protocol, review_rows)
    receipt = {
        "schema_version": "2026-09-14-v2",
        "status": "complete_pending_exhaustive_target_review",
        "training_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": v1.sha256_file(protocol_path)},
        "builder_sha256": v1.sha256_file(Path(__file__)),
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(root).free,
        "excluded_first_pilot_images": len(excluded),
        "scan_counts": scan_counts,
        "metadata_counts": metadata_counts,
        "strict_filter_counts": strict_counts,
        "download_candidates": len(requested),
        "download_successes": len(downloaded),
        "download_failures": failures,
        "near_or_exact_duplicate_rejections": duplicate_rejections,
        "removed_unselected_files": removed_files,
        "removed_unselected_bytes": removed_bytes,
        "review_images": len(review_rows),
        "review_by_stratum": [
            {"class_name": key[0], "attribute_name": key[1], "images": count}
            for key, count in sorted(Counter((row["class_name"], row["attribute_name"]) for row in review_rows).items())
        ],
        "candidate_manifest": {"path": str(manifest), "sha256": manifest_hash, "rows": len(review_rows)},
        "audit_sheets": sheets,
        "claim_boundary": "A disjoint candidate review pool was acquired and mechanically checked. Every target remains untrusted until exhaustive visual review; no pairing, training, or model claim is authorized.",
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
        "sheets": len(receipt["audit_sheets"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
