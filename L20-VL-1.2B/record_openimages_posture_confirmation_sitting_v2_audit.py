#!/usr/bin/env python3
"""Materialize a fail-closed receipt for sitting-supplement-v2 visual review."""
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


def validate_strata(review: dict[str, Any], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_classes: set[str] = set()
    summaries = []
    for stratum in review.get("strata", []):
        class_name = stratum["class_name"]
        if class_name in seen_classes:
            raise RuntimeError("duplicate reviewed class")
        seen_classes.add(class_name)
        expected = {
            row["image_id"]
            for row in manifest
            if row["class_name"] == class_name and row["answer"] == review["answer"]
        }
        decisions = stratum.get("decisions", [])
        ids = [item["image_id"] for item in decisions]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError(f"review does not cover exactly the frozen {class_name} stratum")
        reasons: Counter[str] = Counter()
        for item in decisions:
            if item.get("reason") not in ALLOWED_REASONS:
                raise RuntimeError(f"unknown reason: {item.get('reason')}")
            if not isinstance(item.get("accepted"), bool):
                raise RuntimeError("accepted must be boolean")
            if item["accepted"] != (item["reason"] == "accept_unambiguous"):
                raise RuntimeError("acceptance and reason disagree")
            reasons[item["reason"]] += 1
        summaries.append({
            "class_name": class_name,
            "answer": review["answer"],
            "reviewed": len(decisions),
            "accepted": reasons["accept_unambiguous"],
            "rejected": len(decisions) - reasons["accept_unambiguous"],
            "acceptance_rate": reasons["accept_unambiguous"] / len(decisions),
            "reason_counts": dict(sorted(reasons.items())),
        })
    if not summaries:
        raise RuntimeError("no reviewed strata")
    return summaries


def validate_sheet_scopes(review: dict[str, Any], build: dict[str, Any], audit_dir: Path, sha256_file) -> None:
    frozen = {
        index: item["sha256"] for index, item in enumerate(build["audit_sheets"], 1)
    }
    seen: set[int] = set()
    for item in review.get("reviewed_sheet_scopes", []):
        number = int(item["sheet"])
        if number in seen or frozen.get(number) != item["sha256"]:
            raise RuntimeError("reviewed sheet scope is not frozen")
        seen.add(number)
        path = audit_dir / f"openimages-posture-confirmation-sitting-v2-sheet-{number:02d}.png"
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"review sheet hash mismatch: {number}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    protocol = json.loads(args.protocol.read_text())
    build = json.loads(args.build_receipt.read_text())
    review = json.loads(args.review.read_text())
    if build.get("status") != "complete_pending_exhaustive_primary_agent_and_independent_human_review":
        raise RuntimeError("candidate build is incomplete")
    if build["protocol"]["sha256"] != sha256_file(args.protocol):
        raise RuntimeError("build/protocol hash mismatch")
    if build["candidate_manifest"]["sha256"] != sha256_file(args.manifest):
        raise RuntimeError("build/manifest hash mismatch")
    if review["candidate_manifest_sha256"] != sha256_file(args.manifest):
        raise RuntimeError("review/manifest hash mismatch")
    if review.get("independent_human_review_completed") is not False:
        raise RuntimeError("reviewer boundary changed")

    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    summaries = validate_strata(review, manifest)
    validate_sheet_scopes(review, build, args.audit_dir, sha256_file)
    minimum = int(protocol["sampling"]["minimum_primary_agent_accepted_per_class"])
    for item in summaries:
        item["frozen_minimum_required"] = minimum
        item["passes"] = item["accepted"] >= minimum
    failed = [item["class_name"] for item in summaries if not item["passes"]]
    if not failed:
        raise RuntimeError("early-stop rejection requires a failed complete stratum")
    reviewed = sum(item["reviewed"] for item in summaries)
    result = {
        "schema_version": "2026-09-14-v2",
        "status": "complete_early_stopped_negative_visual_audit",
        "completed_at": utc_now(),
        "decision": "reject_openimages_posture_confirmation_sitting_supplement_v2",
        "reason": f"Fully reviewed strata failed the frozen acceptance gate: {', '.join(failed)}.",
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "build_receipt": {"path": str(args.build_receipt), "sha256": sha256_file(args.build_receipt)},
        "candidate_manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
        "review": {"path": str(args.review), "sha256": sha256_file(args.review)},
        "recorder_sha256": sha256_file(Path(__file__)),
        "reviewed_strata": summaries,
        "failed_strata": failed,
        "pool_review_progress": {
            "reviewed": reviewed,
            "unreviewed_after_fail_closed_early_stop": len(manifest) - reviewed,
            "total": len(manifest),
        },
        "training_started": False,
        "model_evaluation_started": False,
        "pairing_started": False,
        "next_allowed_step": "Freeze a benchmark from a cleaner target-level source or redesign the diagnostic around labels that can be verified from pixels without age/gender ambiguity.",
        "claim_boundary": review["claim_boundary"],
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
