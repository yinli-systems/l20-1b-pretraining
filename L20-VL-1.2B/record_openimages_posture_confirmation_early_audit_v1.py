#!/usr/bin/env python3
"""Record a fail-closed early-stop decision for a confirmation candidate pool."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


ALLOWED_REASONS = {
    "accept_unambiguous",
    "posture_occluded",
    "ambiguous_non_sitting_posture",
    "held_not_sitting",
    "crouching_or_squatting",
    "kneeling_or_crawling",
    "wrong_class_or_target",
    "secondary_media",
}


def validate_decisions(
    review: dict[str, Any], manifest_rows: list[dict[str, Any]]
) -> tuple[int, int, Counter[str]]:
    target_rows = [
        row
        for row in manifest_rows
        if row["class_name"] == review["class_name"] and row["answer"] == review["answer"]
    ]
    expected_ids = {row["image_id"] for row in target_rows}
    decisions = review.get("decisions", [])
    actual_ids = [row["image_id"] for row in decisions]
    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("duplicate reviewed image id")
    if set(actual_ids) != expected_ids:
        raise RuntimeError("review does not cover exactly the frozen stratum")
    reasons: Counter[str] = Counter()
    for decision in decisions:
        reason = decision.get("reason")
        if reason not in ALLOWED_REASONS:
            raise RuntimeError(f"unknown review reason: {reason}")
        accepted = decision.get("accepted")
        if not isinstance(accepted, bool):
            raise RuntimeError("accepted must be boolean")
        if accepted != (reason == "accept_unambiguous"):
            raise RuntimeError("acceptance and reason disagree")
        reasons[reason] += 1
    accepted = reasons["accept_unambiguous"]
    return len(decisions), accepted, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    protocol = json.loads(args.protocol.read_text())
    build = json.loads(args.build_receipt.read_text())
    review = json.loads(args.review.read_text())
    if build.get("status") != "complete_pending_exhaustive_human_target_review":
        raise RuntimeError("candidate pool build is incomplete")
    if build["protocol"]["sha256"] != sha256_file(args.protocol):
        raise RuntimeError("build/protocol hash mismatch")
    if build["candidate_manifest"]["sha256"] != sha256_file(args.manifest):
        raise RuntimeError("build/manifest hash mismatch")
    if review["candidate_manifest_sha256"] != sha256_file(args.manifest):
        raise RuntimeError("review/manifest hash mismatch")
    if review.get("independent_human_review_completed") is not False:
        raise RuntimeError("reviewer boundary changed")

    manifest_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    reviewed, accepted, reasons = validate_decisions(review, manifest_rows)
    minimum = int(protocol["sampling"]["minimum_human_accepted_per_stratum"])
    if accepted >= minimum:
        raise RuntimeError("early-stop rejection is invalid because the stratum passed")
    total = int(build["candidate_manifest"]["rows"])
    result = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_early_stopped_negative_visual_audit",
        "completed_at": utc_now(),
        "decision": "reject_confirmation_candidate_pool_v1",
        "reason": "The fully reviewed Boy/sitting stratum cannot meet the frozen minimum acceptance gate, so the complete balanced pool cannot be formed.",
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "build_receipt": {"path": str(args.build_receipt), "sha256": sha256_file(args.build_receipt)},
        "candidate_manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
        "review": {"path": str(args.review), "sha256": sha256_file(args.review)},
        "recorder_sha256": sha256_file(Path(__file__)),
        "reviewed_stratum": {
            "class_name": review["class_name"],
            "answer": review["answer"],
            "reviewed": reviewed,
            "accepted": accepted,
            "rejected": reviewed - accepted,
            "acceptance_rate": accepted / reviewed,
            "frozen_minimum_required": minimum,
            "passes": False,
            "reason_counts": dict(sorted(reasons.items())),
        },
        "pool_review_progress": {
            "reviewed": reviewed,
            "unreviewed_after_fail_closed_early_stop": total - reviewed,
            "total": total,
        },
        "training_started": False,
        "model_evaluation_started": False,
        "pairing_started": False,
        "next_allowed_step": "Construct a new validation-split sitting supplement with stronger model-independent support-context and class/visibility gates, then visually audit it before pairing.",
        "claim_boundary": review["claim_boundary"],
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
