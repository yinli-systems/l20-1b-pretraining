#!/usr/bin/env python3
"""Finalize balanced posture pairs using only exhaustively human-accepted targets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import build_openimages_posture_pilot as v1
import build_openimages_posture_candidates_v2 as v2


ROOT = Path(__file__).resolve().parent


def validate_record(record: dict[str, Any], label: str) -> None:
    path = Path(record["path"])
    if not path.is_file() or v1.sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"{label} hash mismatch")


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_human_accepted_posture_pair_finalization_only":
        raise RuntimeError("posture pair finalization is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("pair finalization must not authorize training")
    if v1.sha256_file(Path(__file__)) != protocol["source_code"]["builder_sha256"]:
        raise RuntimeError("pair finalizer source hash mismatch")
    test_path = ROOT / "test_finalize_openimages_posture_pairs_v1.py"
    if v1.sha256_file(test_path) != protocol["source_code"]["test_sha256"]:
        raise RuntimeError("pair finalizer test source hash mismatch")
    for name, record in protocol["inputs"].items():
        validate_record(record, name)
    v2_audit = json.loads(Path(protocol["inputs"]["v2_human_audit"]["path"]).read_text())
    v3_audit = json.loads(Path(protocol["inputs"]["v3_human_audit"]["path"]).read_text())
    if v2_audit.get("decision") != "reject_openimages_posture_candidates_v2":
        raise RuntimeError("expected frozen v2 candidate-pool rejection")
    if v3_audit.get("decision") != "accept_openimages_posture_standing_supplement_v3":
        raise RuntimeError("v3 standing supplement has not passed human review")
    return protocol


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_human_accepted(
    manifest: list[dict],
    decisions: list[dict],
    source_pilot: str,
) -> list[dict]:
    by_id = {row["image_id"]: row for row in decisions}
    if len(by_id) != len(decisions):
        raise ValueError(f"duplicate decision ids in {source_pilot}")
    manifest_ids = {row["image_id"] for row in manifest}
    if manifest_ids != set(by_id):
        raise ValueError(f"manifest and decision ids differ in {source_pilot}")
    output = []
    for row in manifest:
        decision = by_id[row["image_id"]]
        if decision["class_name"] != row["class_name"] or decision["posture"] != row["answer"]:
            raise ValueError(f"decision metadata mismatch for {row['image_id']}")
        if decision["decision"] == "accept":
            output.append(
                {
                    **row,
                    "human_audit": {
                        "source_pilot": source_pilot,
                        "decision": "accept",
                        "reason": decision["reason"],
                    },
                }
            )
    return output


def choose_targets(protocol: dict[str, Any], candidates: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[(row["class_name"], row["answer"])].append(row)
    selected = []
    for class_name, count in sorted(protocol["pair_counts_by_class"].items()):
        for answer in ("sitting", "standing"):
            values = sorted(
                grouped[(class_name, answer)],
                key=lambda row: v1.stable_hash(
                    protocol["seed"], "final_target", class_name, answer, row["image_id"]
                ),
            )
            if len(values) < count:
                raise RuntimeError(
                    f"insufficient accepted {class_name}/{answer} targets: {len(values)} < {count}"
                )
            selected.extend(values[:count])
    return sorted(selected, key=lambda row: (row["class_name"], row["answer"], row["image_id"]))


def make_pairs(protocol: dict[str, Any], selected: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        grouped[(row["class_name"], row["answer"])].append(row)
    pairs = []
    for class_name, count in sorted(protocol["pair_counts_by_class"].items()):
        sitting = sorted(
            grouped[(class_name, "sitting")],
            key=lambda row: v1.stable_hash(protocol["seed"], "pair_sitting", class_name, row["image_id"]),
        )
        standing = sorted(
            grouped[(class_name, "standing")],
            key=lambda row: v1.stable_hash(protocol["seed"], "pair_standing", class_name, row["image_id"]),
        )
        if len(sitting) != count or len(standing) != count:
            raise RuntimeError(f"unbalanced selected targets for {class_name}")
        for left, right in zip(sitting, standing):
            pair_id = v1.stable_hash(
                protocol["seed"], "pair", class_name, left["image_id"], right["image_id"]
            )[:24]
            pairs.append(
                {
                    "schema_version": "2026-09-14-final-v1",
                    "pair_id": pair_id,
                    "class_name": class_name,
                    "candidate_answers": ["sitting", "standing"],
                    "image_a": left,
                    "image_b": right,
                }
            )
    return sorted(pairs, key=lambda row: row["pair_id"])


def build(protocol_path: Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    v2_manifest = load_jsonl(Path(protocol["inputs"]["v2_candidate_manifest"]["path"]))
    v2_decisions = load_jsonl(Path(protocol["inputs"]["v2_human_decisions"]["path"]))
    v3_manifest = load_jsonl(Path(protocol["inputs"]["v3_candidate_manifest"]["path"]))
    v3_decisions = load_jsonl(Path(protocol["inputs"]["v3_human_decisions"]["path"]))
    v2_accepted = merge_human_accepted(v2_manifest, v2_decisions, "posture_candidates_v2")
    v3_accepted = merge_human_accepted(v3_manifest, v3_decisions, "standing_supplement_v3")
    accepted = v2_accepted + v3_accepted
    if any(row["human_audit"]["decision"] != "accept" for row in accepted):
        raise RuntimeError("non-accepted target escaped audit filtering")

    deduplicated, duplicate_rejections = v1.deduplicate_images(accepted)
    selected = choose_targets(protocol, deduplicated)
    pairs = make_pairs(protocol, selected)
    selected_ids = {row["image_id"] for row in selected}
    if len(selected_ids) != len(selected):
        raise RuntimeError("selected target image ids are not unique")
    if len({row["image_sha256"] for row in selected}) != len(selected):
        raise RuntimeError("selected targets contain exact image duplicates")

    targets_path = Path(protocol["outputs"]["target_manifest"])
    pairs_path = Path(protocol["outputs"]["pair_manifest"])
    target_hash = v2.write_jsonl_atomic(targets_path, selected)
    pair_hash = v2.write_jsonl_atomic(pairs_path, pairs)
    available_counts = Counter((row["class_name"], row["answer"]) for row in deduplicated)
    selected_counts = Counter((row["class_name"], row["answer"]) for row in selected)
    receipt = {
        "schema_version": "2026-09-14-final-v1",
        "status": "complete_human_accepted_balanced_pairs",
        "training_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": v1.sha256_file(protocol_path)},
        "builder_sha256": v1.sha256_file(Path(__file__)),
        "human_accepted_inputs": {"v2": len(v2_accepted), "v3": len(v3_accepted), "combined": len(accepted)},
        "cross_pilot_exact_or_dhash_lte_4_rejections": duplicate_rejections,
        "available_after_deduplication": [
            {"class_name": key[0], "answer": key[1], "images": count}
            for key, count in sorted(available_counts.items())
        ],
        "selected_by_stratum": [
            {"class_name": key[0], "answer": key[1], "images": count}
            for key, count in sorted(selected_counts.items())
        ],
        "target_manifest": {"path": str(targets_path), "sha256": target_hash, "rows": len(selected)},
        "pair_manifest": {"path": str(pairs_path), "sha256": pair_hash, "rows": len(pairs)},
        "verified_rejected_targets_used": 0,
        "verified_unique_selected_image_ids": len(selected_ids),
        "claim_boundary": "This receipt finalizes an audited data manifest only. It does not define train/evaluation splits, authorize parameter updates, validate transfer, support benchmark claims, or authorize release.",
    }
    v2.write_json_atomic(Path(protocol["outputs"]["receipt"]), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.protocol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
