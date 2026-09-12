"""Fail-closed validation for the post-training stage contract."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_STAGE_IDS = [
    "stage-0-base-selection",
    "stage-1-reasoning-bootstrap-pilot",
    "stage-2-dual-mode-sft",
    "stage-3-preference-pilot",
    "stage-4-rlvr",
    "stage-5-redistillation",
    "stage-6-soup-and-base-merge",
]


def validate(protocol: dict) -> dict:
    errors: list[str] = []
    if protocol.get("automatic_promotion") is not False:
        errors.append("automatic_promotion must be false")
    if protocol.get("released_parent", {}).get("prediction_tokens") != 19_999_703_040:
        errors.append("released parent token count changed")

    data = protocol.get("global_data_rules", {})
    for field in ("benchmark_seeded_training_data", "evaluation_split_training_use"):
        if data.get(field) != "forbidden":
            errors.append(f"{field} must be forbidden")
    required = set(data.get("required_record_fields", []))
    for field in ("source_revision", "license", "normalized_hash", "problem_family", "verification_result"):
        if field not in required:
            errors.append(f"missing required provenance field: {field}")

    sources = protocol.get("candidate_source_registry", [])
    if not sources:
        errors.append("candidate source registry is empty")
    for source in sources:
        revision = source.get("revision", "")
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            errors.append(f"source revision is not a pinned commit: {source.get('dataset')}")
        if not source.get("license") or not source.get("admission"):
            errors.append(f"source lacks license or admission status: {source.get('dataset')}")
    tulu = next((source for source in sources if source.get("dataset", "").startswith("allenai/tulu-3")), {})
    if not tulu.get("admission", "").startswith("blocked_pending"):
        errors.append("Tulu mixture must remain blocked pending component-level license audit")

    stages = protocol.get("stages", [])
    ids = [stage.get("id") for stage in stages]
    if ids != EXPECTED_STAGE_IDS:
        errors.append("stage order or identity changed")
    for stage in stages:
        mixture = stage.get("mixture")
        if mixture is not None and not math.isclose(sum(mixture.values()), 1.0, abs_tol=1e-12):
            errors.append(f"{stage.get('id')} mixture does not sum to one")
        if not stage.get("gate"):
            errors.append(f"{stage.get('id')} has no admission gate")

    if stages:
        stage0 = stages[0]
        if stage0.get("status") != "blocked_live_verification" or stage0.get("train") is not False:
            errors.append("stage 0 must remain non-training and blocked on live verification")
    rlvr = next((stage for stage in stages if stage.get("id") == "stage-4-rlvr"), {})
    ladder = rlvr.get("single_l20_episode_ladder", [])
    if ladder != sorted(set(ladder)) or not ladder or ladder[-1] > 100_000:
        errors.append("single-L20 RLVR ladder must be unique, increasing, and capped at 100k")
    if "2,000,000" not in rlvr.get("not_authorized", ""):
        errors.append("literature-scale RLVR must be explicitly unauthorized")

    return {
        "valid": not errors,
        "protocol_version": protocol.get("protocol_version"),
        "stage_count": len(stages),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path, nargs="?", default=Path("posttraining_protocol.json"))
    args = parser.parse_args()
    result = validate(json.loads(args.protocol.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
