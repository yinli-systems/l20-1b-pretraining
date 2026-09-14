#!/usr/bin/env python3
"""Freeze the exhaustive human audit of standing supplement v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


REJECTIONS = {
    "2bbf244bba8e70ab": "class_label_or_target_conflict",
    "3b2efcdf6a5b81e1": "running_not_standing",
    "5289961b58272206": "running_or_dynamic_one_leg_pose",
    "05a91922c598e2c7": "basketball_action_not_standing",
    "a7da7ad3b7d20075": "upper_body_only",
    "6a000e510d81d0b0": "bent_or_crouched_posture",
    "6828d261e2a07545": "vehicle_occlusion_and_lower_body_hidden",
    "da88c72c542dc909": "railing_occlusion_and_leaning_posture",
    "38e7212d3b598936": "restaurant_closeup_and_lower_body_hidden",
    "75f9153575977e9e": "running_not_standing",
    "1fc7ddc5dc4d8900": "lower_body_out_of_frame",
    "697fffb3ed64a705": "upper_body_only",
    "a2ad41e8cbead1e3": "target_supported_and_lower_body_submerged",
    "ffc001bab3297f1d": "table_occlusion_and_lower_body_hidden",
    "48f698cabd66a380": "lower_body_out_of_frame",
    "76c8ba0ad5b5e499": "torso_only",
    "39bbd2a23ff701f1": "lower_body_out_of_frame",
    "de7fe55c824f2d3a": "crowd_occlusion_and_target_conflict",
    "29677a44c35ca3a1": "chair_occlusion_and_lower_body_hidden",
    "3fee7a94ccd5dbae": "crowd_and_lower_body_hidden",
    "4ce68fdc0cbbb1c8": "desk_occlusion_and_lower_body_hidden",
    "7db3bdbe1f5d3437": "multiple_people_in_target_box",
    "4e5549ea8aa64ca7": "secondary_media_or_composited_graphic",
    "f3bf6d103823de5f": "bending_over_table_and_posture_ambiguous",
    "0f9a1e52af4c1191": "guitar_occlusion_and_lower_body_hidden",
    "28bce9d0790ec1ec": "lower_body_out_of_frame",
    "025f84faebef8866": "photographer_closeup_and_lower_body_hidden",
    "e8ff9560d1cad830": "bending_over_table_and_lower_body_hidden",
    "1da1452e5f87561d": "baseball_action_not_standing",
    "abf3e982838cfdbd": "lectern_occlusion_and_lower_body_hidden",
    "c68e91c131fe4bab": "foreground_container_occludes_lower_body",
    "5fee97c0ead14d4e": "crowd_and_upper_body_only",
    "7c715670fadf7af1": "podium_occlusion_and_torso_only",
    "7d43fe4af9da6059": "crowd_and_upper_body_only",
    "5108241bf8d437d1": "conference_crowd_and_upper_body_only",
    "b5ad1d51a855be9f": "parade_crowd_and_upper_body_only",
    "a05e52dc10b8584d": "office_crowd_and_upper_body_only",
    "0459b74c5e16b460": "torso_only",
    "f46a9734d7de0221": "balcony_occlusion_and_upper_body_only",
    "24ae1919098798db": "lower_body_out_of_frame",
    "ad6a1eb8ae060703": "stage_and_upper_body_only",
    "f787da74cc7abe85": "guitar_occlusion_and_upper_body_only",
    "48528377e973270f": "crowd_and_torso_only",
    "aca09c4c66966471": "walking_not_standing",
    "d8b68410c882a4fc": "multiple_people_in_target_box",
    "117f3ee1293337d2": "classroom_desk_occlusion_and_lower_body_hidden",
    "c3e011a80f61f55b": "protest_crowd_and_lower_body_out_of_frame",
    "82466728762f2571": "saxophone_and_stage_occlusion",
    "6f268691cb302ac6": "crowd_and_lower_body_hidden",
    "d166f963351b8b4b": "target_cut_off_at_image_edge",
    "e26fabc320c4b5da": "back_view_and_lower_body_out_of_frame",
    "6ab6e3b380e8d281": "market_stall_occlusion_and_lower_body_hidden",
    "b0ab273da7e45108": "target_being_held_not_standing",
    "609ef348d55b25f9": "nightclub_crowd_and_torso_only",
    "df22fe879ec6a2c6": "banner_occlusion_and_upper_body_only",
    "cdee3b53ab0d6a4d": "presentation_table_occlusion_and_lower_body_hidden",
    "b4d9ed3d067f38b4": "guitar_occlusion_and_upper_body_only",
    "8159de44273675cc": "lower_body_out_of_frame",
    "f3c1006077b62d27": "restaurant_crowd_and_lower_body_hidden",
    "886e1727b336792b": "child_target_and_lower_body_hidden",
    "0f4e9506eb7932c9": "runner_torso_only",
    "8109b7d8db01b576": "costume_torso_only",
    "51249b9533a403af": "salon_occlusion_and_torso_only",
    "4e23db3ad6e8e7d2": "award_stage_and_lower_body_hidden",
    "3691043805514ee3": "bending_over_table_and_posture_ambiguous",
}

EXPECTED_ACCEPTED_BY_STRATUM = {
    "Boy/standing": 29,
    "Man/standing": 24,
    "Woman/standing": 26,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_decisions(rows: list[dict]) -> list[dict]:
    image_ids = [row["image_id"] for row in rows]
    if len(rows) != 144 or len(set(image_ids)) != 144:
        raise ValueError("v3 candidate manifest must contain exactly 144 unique images")
    unknown = sorted(set(REJECTIONS) - set(image_ids))
    if unknown:
        raise ValueError(f"audit contains unknown image ids: {unknown}")
    output = []
    for index, row in enumerate(rows, start=1):
        reason = REJECTIONS.get(row["image_id"])
        output.append(
            {
                "audit_index": index,
                "audit_page": ((index - 1) // 8) + 1,
                "image_id": row["image_id"],
                "class_name": row["class_name"],
                "posture": row["answer"],
                "pose_review_rank_score": row["curation_pose_features"]["mean_lower_body_score"],
                "decision": "reject" if reason else "accept",
                "reason": reason or "standing_posture_visually_decidable",
            }
        )
    return output


def summarize(decisions: list[dict]) -> dict:
    totals, accepted = Counter(), Counter()
    for item in decisions:
        key = f'{item["class_name"]}/{item["posture"]}'
        totals[key] += 1
        if item["decision"] == "accept":
            accepted[key] += 1
    if dict(accepted) != EXPECTED_ACCEPTED_BY_STRATUM:
        raise ValueError(f"unexpected accepted counts: {dict(accepted)}")
    return {
        key: {
            "reviewed": totals[key],
            "accepted": accepted[key],
            "rejected": totals[key] - accepted[key],
            "acceptance_rate": accepted[key] / totals[key],
            "minimum_required": 20,
            "passes": accepted[key] >= 20,
        }
        for key in sorted(totals)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--decisions-out", required=True, type=Path)
    parser.add_argument("--receipt-out", required=True, type=Path)
    args = parser.parse_args()

    build_receipt = json.loads(args.build_receipt.read_text())
    if sha256_file(args.manifest) != build_receipt["candidate_manifest"]["sha256"]:
        raise RuntimeError("candidate manifest hash does not match build receipt")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    decisions = build_decisions(rows)
    strata = summarize(decisions)

    args.decisions_out.parent.mkdir(parents=True, exist_ok=True)
    args.decisions_out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions))
    accepted_total = sum(row["decision"] == "accept" for row in decisions)
    receipt = {
        "schema_version": "2026-09-14-v3",
        "audit_type": "exhaustive_single_reviewer_visual_target_census",
        "inter_rater_agreement_measured": False,
        "review_scope": "all 144 standing targets using the full image with target box and enlarged target crop",
        "acceptance_rule": "Real person; standing posture visually decidable without caption. Reject posture/lower-body occlusion, dynamic non-standing pose, wrong target, label conflict, and secondary media.",
        "pose_score_role": "review ordering only; not used as visual evidence and not an acceptance gate",
        "inputs": {
            "candidate_manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
            "build_receipt": {"path": str(args.build_receipt), "sha256": sha256_file(args.build_receipt)},
        },
        "decisions": {
            "path": str(args.decisions_out),
            "sha256": sha256_file(args.decisions_out),
            "reviewed": len(decisions),
            "accepted": accepted_total,
            "rejected": len(decisions) - accepted_total,
            "acceptance_rate": accepted_total / len(decisions),
        },
        "strata": strata,
        "decision": "accept_openimages_posture_standing_supplement_v3",
        "all_strata_pass": all(value["passes"] for value in strata.values()),
        "training_authorized": False,
        "next_allowed_step": "Construct and hash a final balanced pair manifest from human-accepted v2 targets plus human-accepted v3 standing targets; rerun exact and perceptual deduplication before any training protocol is considered.",
        "claim_boundary": "This accepts a curation supplement only. It does not yet authorize pairing, parameter updates, model selection, benchmark claims, or release.",
    }
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
