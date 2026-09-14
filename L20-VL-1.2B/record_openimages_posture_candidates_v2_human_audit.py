#!/usr/bin/env python3
"""Freeze the exhaustive human audit of Open Images posture candidates v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


REJECTIONS = {
    "41f17db2d5b37487": "posture_hidden",
    "53191d005d7214b9": "posture_hidden",
    "61dab6d3881c32d4": "blurred_and_posture_hidden",
    "6912576e269003a2": "upper_body_only",
    "74fd642210a6f157": "seat_and_posture_hidden",
    "7c52e8fa173e5fdc": "torso_only",
    "005cdcff488f2811": "torso_only",
    "0da88b331f75a3d4": "portrait_only",
    "23e4a81d91617167": "crowd_and_posture_hidden",
    "34719999168a042a": "back_view_and_lower_body_hidden",
    "55faceebbe7cfb58": "upper_body_only",
    "a5a955b115887bfa": "crouching_or_seated_label_conflict",
    "a9f9cc2bd1da1fc9": "lower_body_hidden",
    "ad1b0f23ab541f70": "tilted_view_and_lower_body_hidden",
    "c7e3e943f81a0cf2": "torso_only",
    "d8a672f440bbb86d": "doorway_occlusion_and_lower_body_hidden",
    "e1869fbbc301b823": "upper_body_only",
    "e29714e94bbc3821": "upper_body_only",
    "e966627728bada03": "upper_body_only",
    "fb813d9ed6b9b3a8": "lower_body_hidden_near_chair",
    "0b62ed6119fc271a": "no_visible_seat_or_seated_geometry",
    "2786e2d03a781c67": "lying_label_conflict",
    "51eeb0a73956238e": "upright_with_no_visible_seat",
    "577e10dbe90f79b8": "lower_body_hidden",
    "7ec1d2c3d4f04280": "torso_only_at_table",
    "afd634b40df3ff6e": "torso_only_with_no_visible_support",
    "b68962770b1a8766": "closeup_with_no_visible_support",
    "4c152259c21d7429": "torso_only",
    "64ced89f41024f74": "upper_body_only_at_table",
    "9a78a7881d6f9b86": "upper_body_only",
    "9c47bb9b509b8140": "target_being_held",
    "dd3f1ccc103618df": "portrait_only",
    "e8f3327bd09092dd": "back_view_and_upper_body_only",
    "e8fe15d38a65c26b": "portrait_only",
    "fef0c245a181a7f4": "side_view_and_upper_body_only",
    "957b09bf2eea4cc9": "secondary_media_or_illustration",
    "084365d1569b06b5": "portrait_and_torso_only",
    "1e2ad7a7179ed109": "group_and_lower_body_hidden",
    "31c22d2d86edab6e": "boat_occlusion_and_posture_ambiguous",
    "3da8fbddac078061": "portrait_only",
    "442c742077b02b0f": "crowd_back_view_and_upper_body_only",
    "522c470037ced98e": "party_and_torso_only",
    "59baba1e7ddc3920": "railing_occlusion_and_posture_ambiguous",
    "65b270889d3a1fc8": "upper_body_only",
    "66917c0cb4f0c8a5": "torso_only",
    "cd1f9085b28c226f": "party_and_torso_only",
    "d08f42ec9a8182df": "event_occlusion_and_upper_body_only",
    "e5cfaa6e33dc355e": "party_and_torso_only",
    "f1f5a20ded350896": "party_and_torso_only",
    "f22ead09a1c9d117": "selfie_closeup_and_posture_hidden",
    "462763bd23adbe39": "table_occlusion_and_posture_ambiguous",
    "58f4d1c42a83cb9b": "leaning_over_chair_and_posture_ambiguous",
    "bee5ae70b42898b7": "lower_body_hidden_with_no_visible_support",
    "ddf047392699495b": "cafe_torso_only_with_no_visible_support",
    "22107dc13e4dc979": "portrait_only",
    "34b28ca6e02b20c2": "wedding_torso_only",
    "3985955c286e9ea5": "horse_occlusion_and_lower_body_hidden",
    "6f290ba3478dbdd8": "crowd_closeup_and_torso_only",
    "919bbbc20631a6e5": "party_and_upper_body_only",
    "91b7c1640cd903fc": "podium_occlusion_and_lower_body_hidden",
    "938450e01bcf9186": "stage_and_lower_body_hidden",
    "9642e9fcd8d2f9c5": "graduation_crowd_and_lower_body_hidden",
    "aa1b809b24ebb66b": "group_and_upper_body_only",
    "b7ee4918878a848e": "guitar_occlusion_and_lower_body_hidden",
    "cb89071489cd83f4": "stage_group_and_lower_body_hidden",
    "cc0e6edb624e537e": "shirt_occlusion_and_torso_only",
    "ce40c043940db984": "podium_occlusion_and_lower_body_hidden",
}

EXPECTED_ACCEPTED_BY_STRATUM = {
    "Boy/sitting": 18,
    "Boy/standing": 10,
    "Girl/sitting": 17,
    "Girl/standing": 16,
    "Man/sitting": 23,
    "Man/standing": 10,
    "Woman/sitting": 20,
    "Woman/standing": 11,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def posture(row: dict) -> str:
    return row["answer"]


def build_decisions(rows: list[dict]) -> list[dict]:
    image_ids = [row["image_id"] for row in rows]
    if len(rows) != 192 or len(set(image_ids)) != 192:
        raise ValueError("candidate manifest must contain exactly 192 unique images")
    unknown = sorted(set(REJECTIONS) - set(image_ids))
    if unknown:
        raise ValueError(f"audit contains unknown image ids: {unknown}")

    decisions = []
    for index, row in enumerate(rows, start=1):
        reason = REJECTIONS.get(row["image_id"])
        decisions.append(
            {
                "audit_index": index,
                "audit_page": ((index - 1) // 8) + 1,
                "image_id": row["image_id"],
                "class_name": row["class_name"],
                "posture": posture(row),
                "bbox": row["bbox"],
                "bbox_area": row["bbox_area"],
                "decision": "reject" if reason else "accept",
                "reason": reason or "posture_visually_decidable",
            }
        )
    return decisions


def summarize(decisions: list[dict]) -> dict:
    totals = Counter()
    accepted = Counter()
    for item in decisions:
        key = f'{item["class_name"]}/{item["posture"]}'
        totals[key] += 1
        if item["decision"] == "accept":
            accepted[key] += 1
    if dict(accepted) != EXPECTED_ACCEPTED_BY_STRATUM:
        raise ValueError(f"unexpected accepted counts: {dict(accepted)}")

    strata = {}
    for key in sorted(totals):
        strata[key] = {
            "reviewed": totals[key],
            "accepted": accepted[key],
            "rejected": totals[key] - accepted[key],
            "acceptance_rate": accepted[key] / totals[key],
            "minimum_required": 16,
            "passes": accepted[key] >= 16,
        }
    return strata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    decisions = build_decisions(rows)
    strata = summarize(decisions)

    args.decisions_out.parent.mkdir(parents=True, exist_ok=True)
    args.decisions_out.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in decisions)
    )
    decisions_sha256 = sha256_file(args.decisions_out)
    reviewed_id_digest = hashlib.sha256("\n".join(row["image_id"] for row in rows).encode()).hexdigest()

    passing = [key for key, value in strata.items() if value["passes"]]
    failing = [key for key, value in strata.items() if not value["passes"]]
    accepted_total = sum(value["accepted"] for value in strata.values())
    receipt = {
        "schema_version": "2026-09-14-v1",
        "audit_type": "exhaustive_single_reviewer_visual_target_census",
        "review_scope": "all 192 targets using the full image with target box and enlarged target crop",
        "acceptance_rule": "Real person; requested posture visually decidable without caption. Reject hidden or ambiguous posture, crouching, kneeling, squatting, wrong target, label conflict, and secondary media.",
        "candidate_manifest": {
            "path": str(args.manifest),
            "sha256": sha256_file(args.manifest),
            "reviewed_image_ids_sha256": reviewed_id_digest,
        },
        "decisions": {
            "path": str(args.decisions_out),
            "sha256": decisions_sha256,
            "reviewed": len(decisions),
            "accepted": accepted_total,
            "rejected": len(decisions) - accepted_total,
            "acceptance_rate": accepted_total / len(decisions),
        },
        "strata": strata,
        "passing_strata": passing,
        "failing_strata": failing,
        "decision": "reject_openimages_posture_candidates_v2",
        "training_authorized": False,
        "reason": "Three standing strata fail the frozen minimum of 16 human-accepted targets: Boy/standing, Man/standing, and Woman/standing.",
        "diagnosis": "The shared minimum bbox width favored close-up standing targets whose lower bodies were hidden. Sitting strata were substantially cleaner because support objects and bent-body geometry remained visible.",
        "next_allowed_step": "Use this audited set only as development evidence to freeze posture-conditional geometry for a disjoint standing supplement; visually audit every new target before pairing or training.",
        "claim_boundary": "No pairing, parameter update, benchmark claim, or release is authorized by this receipt.",
    }
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
