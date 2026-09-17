#!/usr/bin/env python3
"""Fail-closed static validation for the frozen 150B MoE data plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data" / "moe7b_150b_mixture_v1.json"
REQUIRED_SOURCES = {
    "fineweb_edu",
    "dclm_baseline",
    "fineweb2_multilingual",
    "stack_v3_permissive",
    "finemath",
    "olmoe_science_reference",
}
REQUIRED_GATES = {
    "immutable_source_revision_and_file_hashes",
    "license_terms_and_attribution_approved",
    "provenance_complete",
    "privacy_and_pii_policy_passed",
    "document_exact_deduplication_passed",
    "cross_source_semantic_near_deduplication_passed",
    "evaluation_contamination_scan_passed",
    "human_content_audit_passed",
    "train_validation_test_isolation_passed",
    "tokenizer_identity_verified",
    "packed_manifest_reproducible",
    "unique_token_floor_passed",
}


def validate(plan: dict) -> dict:
    errors: list[str] = []
    budget = plan.get("budget", {})
    target = budget.get("target_exposed_tokens")
    unique_floor = budget.get("minimum_unique_tokens")
    if target != 150_000_000_000:
        errors.append(f"target_exposed_tokens must be 150B, got {target!r}")
    if unique_floor != 142_500_000_000:
        errors.append(f"minimum_unique_tokens must be 142.5B, got {unique_floor!r}")
    if budget.get("maximum_replay_fraction") != 0.05:
        errors.append("maximum_replay_fraction must be exactly 0.05")

    sources = plan.get("sources", [])
    source_by_id = {source.get("source_id"): source for source in sources}
    if len(source_by_id) != len(sources):
        errors.append("source_id values must be unique")
    if set(source_by_id) != REQUIRED_SOURCES:
        errors.append(
            f"source set mismatch: expected {sorted(REQUIRED_SOURCES)}, got {sorted(source_by_id)}"
        )
    source_total = sum(source.get("target_tokens", 0) for source in sources)
    source_unique_total = sum(source.get("minimum_unique_tokens", 0) for source in sources)
    if source_total != target:
        errors.append(f"source quotas sum to {source_total}, expected {target}")
    if source_unique_total != unique_floor:
        errors.append(
            f"source unique floors sum to {source_unique_total}, expected {unique_floor}"
        )
    for source in sources:
        revision = source.get("revision", "")
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            errors.append(f"{source.get('source_id')}: revision is not a full lowercase git SHA")
        if source.get("minimum_unique_tokens", 0) * 20 != source.get("target_tokens", 0) * 19:
            errors.append(f"{source.get('source_id')}: unique floor is not exactly 95%")

    stages = plan.get("stages", [])
    stage_total = 0
    accumulated = {source_id: 0 for source_id in REQUIRED_SOURCES}
    for stage in stages:
        quotas = stage.get("source_quotas", {})
        if set(quotas) != REQUIRED_SOURCES:
            errors.append(f"stage {stage.get('stage')}: source set mismatch")
        quota_total = sum(quotas.values())
        if quota_total != stage.get("target_tokens"):
            errors.append(
                f"stage {stage.get('stage')}: quotas sum to {quota_total}, expected {stage.get('target_tokens')}"
            )
        stage_total += stage.get("target_tokens", 0)
        for source_id, tokens in quotas.items():
            if source_id in accumulated:
                accumulated[source_id] += tokens
    if stage_total != target:
        errors.append(f"stage targets sum to {stage_total}, expected {target}")
    for source_id, tokens in accumulated.items():
        expected = source_by_id.get(source_id, {}).get("target_tokens")
        if tokens != expected:
            errors.append(f"{source_id}: stage total {tokens}, source target {expected}")

    multilingual = source_by_id.get("fineweb2_multilingual", {})
    if sum(multilingual.get("language_quotas", {}).values()) != multilingual.get("target_tokens"):
        errors.append("FineWeb2 language quotas do not sum to its source target")
    for source_id in ("finemath", "olmoe_science_reference"):
        source = source_by_id.get(source_id, {})
        if sum(source.get("subsource_quotas", {}).values()) != source.get("target_tokens"):
            errors.append(f"{source_id}: subsource quotas do not sum to source target")

    tokenizer = plan.get("tokenizer", {})
    tokenizer_files = tokenizer.get("files", {})
    if tokenizer.get("vocabulary_size") != plan.get("model_contract", {}).get("vocabulary_size"):
        errors.append("tokenizer and model vocabulary sizes differ")
    if set(tokenizer_files) != {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }:
        errors.append("tokenizer file set is incomplete")
    for name, digest in tokenizer_files.items():
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            errors.append(f"{name}: invalid SHA-256")

    gates = set(plan.get("admission_gates", []))
    if not REQUIRED_GATES.issubset(gates):
        errors.append(f"missing admission gates: {sorted(REQUIRED_GATES - gates)}")
    if plan.get("status") != "FROZEN_CANDIDATE_PLAN_NOT_ADMITTED":
        errors.append("plan status must remain fail-closed until real admission receipts exist")
    promotion = plan.get("promotion_policy", {})
    if promotion.get("training_forbidden_until_all_admission_gates_pass") is not True:
        errors.append("training must be forbidden until every gate passes")
    if promotion.get("main_run_starts_from_fresh_initialization") is not True:
        errors.append("150B main run must not silently resume the one-source systems pilot")

    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {
        "status": "PASS" if not errors else "FAIL",
        "plan_id": plan.get("plan_id"),
        "plan_sha256": hashlib.sha256(canonical).hexdigest(),
        "target_exposed_tokens": target,
        "minimum_unique_tokens": unique_floor,
        "source_count": len(sources),
        "stage_count": len(stages),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", nargs="?", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = validate(plan)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
