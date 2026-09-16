#!/usr/bin/env python3
"""Admit and freeze seven Base continuations over the latest screened data pool."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


TARGET_BLOCKS = 262_144
SEQUENCE_LENGTH = 2_048
TARGET_TOKENS = TARGET_BLOCKS * SEQUENCE_LENGTH
BASE_SHA256 = "13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf"
TOKENIZER_SHA256 = "30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6"
VALIDATION_SHA256 = "429c1ab33bdb8a92214ffdd2327462a05275b29fbd60abecc1d98b9fad634af3"
RESEARCH_PLAN_SHA256 = "58d56059111eb92dea1c0f9213a1e7ebbacabe7dea55c24cf31e366401dcc1a7"
EXPANSION_PLAN_SHA256 = "ca16deedb115a04a33b7b49889317866fb806f64014b2161a04b7803e224d0a6"
DCLM_TOPUP_PLAN_SHA256 = "60193bad768d705ea79c7c79bcb1dbe5a8e9c20f2773fda6959202671beb5b7b"
RIGHTS_SNAPSHOT_SHA256 = "07ae41dc8a4cf4d1597e79fed64ef035d75d57c961eaa5171a6464a2325eedf7"
PIPELINE_STATUS = "DCLM_TOPUP_FILTER_AND_PACK_COMPLETE_NOT_TRAINING_ADMITTED"
REPORT_STATUSES = {
    "raw": "COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED",
    "contamination": "SUPPLEMENTAL_EXACT_SPAN_AUDIT_COMPLETE_ADMISSION_PENDING",
    "legacy": "LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED",
    "old_source": "OLD_SOURCE_NORMALIZED_OVERLAP_COMPLETE_NOT_ADMITTED",
    "exclusions": "POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK",
    "quality": "QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED",
    "family": "FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED",
    "pack": "SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED",
}
REQUIRED_CHECKS = (
    "licenses",
    "content_quality",
    "cross_source_deduplication",
    "benchmark_decontamination",
    "family_disjoint_splits",
)
CODE_ALLOWLIST = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "CC0-1.0", "Unlicense",
}

# Basis points are expanded to exact leaf-source quotas with Hamilton allocation.
# The pool is new relative to the original 20B-token Base lineage. Cosmopedia is
# intentionally absent because response-lineage admission did not close.
RECIPES_BPS = {
    "N0_new_pool_broad": {
        "dclm": 2500, "pdf_en": 3500, "finemath4": 1875, "infiwebmath4": 625,
        "code_python": 500, "code_javascript": 200, "code_typescript": 100,
        "code_cpp": 100, "code_java": 100, "multilingual_arb_Arab": 50,
        "multilingual_cmn_Hani": 250, "multilingual_deu_Latn": 50,
        "multilingual_fra_Latn": 50, "multilingual_jpn_Jpan": 50,
        "multilingual_spa_Latn": 50,
    },
    "N1_new_pool_knowledge": {
        "dclm": 3000, "pdf_en": 4000, "finemath4": 1125, "infiwebmath4": 375,
        "code_python": 500, "code_javascript": 200, "code_typescript": 100,
        "code_cpp": 100, "code_java": 100, "multilingual_arb_Arab": 50,
        "multilingual_cmn_Hani": 250, "multilingual_deu_Latn": 50,
        "multilingual_fra_Latn": 50, "multilingual_jpn_Jpan": 50,
        "multilingual_spa_Latn": 50,
    },
    "N2_new_pool_reasoning_code": {
        "dclm": 2000, "pdf_en": 2500, "finemath4": 2625, "infiwebmath4": 875,
        "code_python": 750, "code_javascript": 300, "code_typescript": 150,
        "code_cpp": 150, "code_java": 150, "multilingual_arb_Arab": 50,
        "multilingual_cmn_Hani": 250, "multilingual_deu_Latn": 50,
        "multilingual_fra_Latn": 50, "multilingual_jpn_Jpan": 50,
        "multilingual_spa_Latn": 50,
    },
    "N3_new_pool_balanced": {
        "dclm": 2500, "pdf_en": 3000, "finemath4": 1875, "infiwebmath4": 625,
        "code_python": 750, "code_javascript": 300, "code_typescript": 150,
        "code_cpp": 150, "code_java": 150, "multilingual_arb_Arab": 50,
        "multilingual_cmn_Hani": 250, "multilingual_deu_Latn": 50,
        "multilingual_fra_Latn": 50, "multilingual_jpn_Jpan": 50,
        "multilingual_spa_Latn": 50,
    },
    "N4_new_pool_multilingual": {
        "dclm": 2000, "pdf_en": 2500, "finemath4": 1875, "infiwebmath4": 625,
        "code_python": 500, "code_javascript": 200, "code_typescript": 100,
        "code_cpp": 100, "code_java": 100, "multilingual_arb_Arab": 200,
        "multilingual_cmn_Hani": 1000, "multilingual_deu_Latn": 200,
        "multilingual_fra_Latn": 200, "multilingual_jpn_Jpan": 200,
        "multilingual_spa_Latn": 200,
    },
}
ARMS = (
    ("n0-lr6e5", "N0_new_pool_broad", "0.00006"),
    ("n1-lr6e5", "N1_new_pool_knowledge", "0.00006"),
    ("n2-lr6e5", "N2_new_pool_reasoning_code", "0.00006"),
    ("n3-lr6e5", "N3_new_pool_balanced", "0.00006"),
    ("n4-lr6e5", "N4_new_pool_multilingual", "0.00006"),
    ("n3-lr3e5", "N3_new_pool_balanced", "0.00003"),
    ("n3-lr1e4", "N3_new_pool_balanced", "0.0001"),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, status: str | None = None, expected_sha: str | None = None) -> tuple[dict, str]:
    digest = sha(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"input SHA-256 mismatch: {path}")
    value = json.loads(path.read_text())
    if status is not None and value.get("status") != status:
        raise ValueError(f"input status mismatch: {path}")
    return value, digest


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def exact_block_quotas(basis_points: dict[str, int], total_blocks: int = TARGET_BLOCKS) -> dict[str, int]:
    if set(basis_points) != set(RECIPES_BPS["N0_new_pool_broad"]):
        raise ValueError("recipe source set changed")
    if sum(basis_points.values()) != 10_000 or any(value <= 0 for value in basis_points.values()):
        raise ValueError("recipe basis points must be positive and sum to 10000")
    quotas = {source: total_blocks * value // 10_000 for source, value in basis_points.items()}
    missing = total_blocks - sum(quotas.values())
    ranked = sorted(basis_points, key=lambda source: (-(total_blocks * basis_points[source] % 10_000), source))
    for source in ranked[:missing]:
        quotas[source] += 1
    if sum(quotas.values()) != total_blocks:
        raise AssertionError("largest-remainder quota construction failed")
    return quotas


def validate_packed_array(path: Path, train: dict, source_id: str) -> None:
    expected_array_tokens = int(train["blocks"]) * (SEQUENCE_LENGTH + 1)
    if (
        train.get("array_tokens") != expected_array_tokens
        or train.get("prediction_tokens") != int(train["blocks"]) * SEQUENCE_LENGTH
        or train.get("dtype") != "uint16"
    ):
        raise ValueError(f"packed metadata mismatch: {source_id}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if array.dtype != np.uint16 or array.ndim != 1 or array.shape != (expected_array_tokens,):
            raise ValueError(f"packed array mismatch: {source_id}")
    finally:
        del array


def source_revisions(expansion_plan: dict, topup_plan: dict) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for segment in expansion_plan["segments"] + topup_plan["segments"]:
        source_id = segment["logical_source_id"]
        revision = segment["revision"]
        previous = revisions.setdefault(source_id, revision)
        if previous != revision:
            raise ValueError(f"multiple revisions for source: {source_id}")
    return revisions


def build(root: Path, output: Path, research_plan: Path, rights_snapshot: Path) -> dict:
    root = root.resolve(strict=True)
    research_plan = research_plan.resolve(strict=True)
    rights_snapshot = rights_snapshot.resolve(strict=True)
    if output.exists():
        raise ValueError("output already exists")
    _, research_sha = read_json(research_plan, expected_sha=RESEARCH_PLAN_SHA256)
    rights, rights_sha = read_json(
        rights_snapshot,
        status="PINNED_SOURCE_CARDS_AND_SELECTED_LICENSE_POLICY_REVIEWED",
        expected_sha=RIGHTS_SNAPSHOT_SHA256,
    )
    if set(rights.get("code_allowlist", [])) != CODE_ALLOWLIST or rights.get("training_admitted") is not False:
        raise ValueError("rights snapshot policy changed")

    reports = {
        "raw": root / "data/diverse-audit-dclm-topup-v1/report.json",
        "contamination": root / "data/contamination-dclm-topup-v1/report.json",
        "legacy": root / "data/legacy-screen-dclm-topup-v1/report.json",
        "old_source": root / "data/old-source-overlap-dclm-topup-v1/report.json",
        "exclusions": root / "receipts/exclusion-union-dclm-topup-v1.json",
        "quality": root / "data/quality-family-dclm-topup-v1/report.json",
        "family": root / "data/family-split-dclm-topup-v1/report.json",
        "pack": root / "data/selected-packs-dclm-topup-v1/report.json",
    }
    documents: dict[str, dict] = {}
    evidence: dict[str, str] = {
        "research_plan_sha256": research_sha,
        "rights_snapshot_sha256": rights_sha,
    }
    for key, path in reports.items():
        documents[key], evidence[f"{key}_report_sha256"] = read_json(path, status=REPORT_STATUSES[key])
    pipeline_path = root / "receipts/pipeline-dclm-topup-v1-complete.json"
    pipeline, evidence["pipeline_receipt_sha256"] = read_json(pipeline_path, status=PIPELINE_STATUS)
    for path in reports.values():
        if pipeline.get("artifacts", {}).get(str(path)) != sha(path):
            raise ValueError(f"pipeline receipt does not bind report: {path}")
    if any(document.get("training_admitted") for document in documents.values()):
        raise ValueError("an upstream not-admitted state changed unexpectedly")
    if documents["raw"].get("physical_group_disjointness") != "PASS" or documents["raw"].get("input_hashes_stable") != "PASS":
        raise ValueError("raw identity or disjointness gate failed")
    if documents["family"].get("independent_under_observed_family_edges") is not True:
        raise ValueError("family split independence gate failed")

    expansion_plan, evidence["expansion_plan_sha256"] = read_json(
        root / "source/intake-expansion-v1/plan.json", expected_sha=EXPANSION_PLAN_SHA256
    )
    topup_plan, evidence["dclm_topup_plan_sha256"] = read_json(
        root / "source/intake-dclm-topup-v1/plan.json", expected_sha=DCLM_TOPUP_PLAN_SHA256
    )
    revisions = source_revisions(expansion_plan, topup_plan)
    validation_path = root / "data/development-masked-v2/development-mixture.json"
    if sha(validation_path) != VALIDATION_SHA256:
        raise ValueError("development validation identity changed")
    base_path = root / "formal/run-v5/resume.pt"
    if sha(base_path) != BASE_SHA256:
        raise ValueError("Base checkpoint identity changed")

    packed: dict[str, dict] = {}
    for item in documents["pack"]["sources"]:
        source_id = item["source_id"]
        train = item["outputs"]["train"]
        path = Path(train["path"])
        if sha(path) != train["sha256"]:
            raise ValueError(f"packed source hash changed: {source_id}")
        validate_packed_array(path, train, source_id)
        packed[source_id] = train
    required_sources = set(RECIPES_BPS["N0_new_pool_broad"])
    if set(packed) != required_sources or set(revisions) != required_sources:
        raise ValueError("latest packed source or revision set changed")
    if documents["pack"].get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("packed tokenizer changed")

    output.mkdir()
    manifests_dir = output / "manifests"
    admissions_dir = output / "admissions"
    manifests_dir.mkdir()
    admissions_dir.mkdir()
    protocol = {
        "schema": "p529m-new-pool-mixture-pilots-v1",
        "status": "FROZEN_EXPLORATORY_NEW_POOL",
        "data_scope": "latest family-split, quality-filtered, decontaminated DCLM/PDF/math/code/multilingual pool; no Base-lineage FineWeb replay",
        "target_prediction_tokens_per_run": TARGET_TOKENS,
        "steps_per_run": 256,
        "world_size_per_run": 4,
        "seed": 20260916,
        "arms": [
            {"arm_id": arm_id, "recipe": recipe, "peak_learning_rate": lr}
            for arm_id, recipe, lr in ARMS
        ],
        "base_checkpoint": {"path": str(base_path), "sha256": BASE_SHA256, "step": 7629},
        "training": {
            "architecture": "deep", "microbatch_per_gpu": 4, "gradient_accumulation": 64,
            "global_prediction_tokens_per_step": 2_097_152,
            "warmup_prediction_tokens": 67_108_864,
            "schedule": "warmup-stable-cosine-decay", "deterministic": True,
            "checkpoint_mode": "model-only-final",
        },
        "mfu_gate": {
            "dense_bf16_tflops_per_rtx5090": 209.5, "rolling_window_steps": 10,
            "grace_steps": 5, "minimum_strictly_greater_than": 0.70,
        },
        "selection": {
            "development_signal": "frozen five-domain equal-domain masked next-token loss",
            "next_gate": "same seven-task capability screen, then two-seed confirmation for one selected successor",
            "automatic_promotion": False, "automatic_market_superiority_claim": False,
        },
        "excluded_source": {
            "cosmopedia": "response-lineage admission did not close",
            "base_lineage_fineweb": "excluded to isolate new-pool signal",
        },
    }
    protocol_path = output / "protocol.json"
    write_json(protocol_path, protocol)
    protocol_sha = sha(protocol_path)

    built: dict[str, dict] = {}
    for recipe, basis_points in RECIPES_BPS.items():
        quotas = exact_block_quotas(basis_points)
        capacity: dict[str, dict] = {}
        sources = []
        for source_id in sorted(quotas):
            train = packed[source_id]
            maximum = 2 * int(train["blocks"])
            requested = quotas[source_id]
            if requested > maximum:
                raise ValueError(f"{recipe}/{source_id} exceeds the two-epoch cap")
            capacity[source_id] = {
                "requested_blocks": requested,
                "unique_blocks": int(train["blocks"]),
                "maximum_blocks_at_two_epochs": maximum,
                "headroom_blocks": maximum - requested,
            }
            sources.append({
                "id": source_id,
                "revision": revisions[source_id],
                "prior_prediction_tokens": 0,
                "max_cumulative_epochs": "2.0",
                "shards": [{"path": train["path"], "sha256": train["sha256"], "blocks": train["blocks"]}],
            })
        manifest = {
            "schema": "p529m-packed-mixture-v3",
            "protocol_id": protocol["schema"],
            "recipe": recipe,
            "sequence_length": SEQUENCE_LENGTH,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "source_block_quotas": quotas,
            "sources": sources,
            "evidence": {**evidence, "development_manifest_sha256": VALIDATION_SHA256, "protocol_sha256": protocol_sha},
        }
        manifest_path = manifests_dir / f"{recipe}.json"
        write_json(manifest_path, manifest)
        manifest_sha = sha(manifest_path)
        admission = {
            "schema": "p529m-corpus-admission-v1",
            "status": "PASS",
            "recipe": recipe,
            "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "training_manifest_sha256": manifest_sha,
            "validation_manifest_sha256": VALIDATION_SHA256,
            "protocol_sha256": protocol_sha,
            "checks": {check: "PASS" for check in REQUIRED_CHECKS},
            "evidence": manifest["evidence"],
            "cumulative_capacity": capacity,
            "scope": {
                "licenses": "Revision-pinned source cards plus the frozen permissive-code filter are bound; publication still requires release review.",
                "content_quality": "Frozen structural, language, and quality filters were applied before packing.",
                "cross_source_deduplication": "One eligible representative was selected after combined family closure.",
                "benchmark_decontamination": "Frozen benchmark-prefix, exact-span, legacy-hash, and old-source overlap exclusions were applied.",
                "family_disjoint_splits": "Training and reserved evaluation families are disjoint under observed exact, metadata, and verified near edges.",
            },
            "limitations": [
                "Admission is bounded to these exact hashes, quotas, and protocol.",
                "Observed decontamination and family closure cannot detect every semantic paraphrase.",
                "Dataset-card review is not a blanket grant for every upstream web or PDF document.",
                "A successful pilot does not establish downstream or market superiority.",
            ],
        }
        admission_path = admissions_dir / f"{recipe}.admission.json"
        write_json(admission_path, admission)
        built[recipe] = {
            "basis_points": basis_points,
            "source_block_quotas": quotas,
            "manifest_sha256": manifest_sha,
            "admission_sha256": sha(admission_path),
            "capacity": capacity,
        }

    receipt = {
        "schema": "p529m-new-pool-mixture-pilot-inputs-v1",
        "status": "PASS_NEW_POOL_INPUTS_ADMITTED_FOR_EXPLORATORY_PILOTS",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_prediction_tokens_per_run": TARGET_TOKENS,
        "protocol_sha256": protocol_sha,
        "base_checkpoint_sha256": BASE_SHA256,
        "recipes": built,
        "arms": protocol["arms"],
        "training_launched": False,
        "claim_boundary": "hash-bound exploratory inputs admitted; training, capability gain, and superiority remain unverified",
    }
    write_json(output / "build-receipt.json", receipt)
    for recipe, item in built.items():
        if sha(manifests_dir / f"{recipe}.json") != item["manifest_sha256"]:
            raise ValueError(f"generated manifest identity changed: {recipe}")
        if sha(admissions_dir / f"{recipe}.admission.json") != item["admission_sha256"]:
            raise ValueError(f"generated admission identity changed: {recipe}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--research-plan", type=Path, required=True)
    parser.add_argument("--rights-snapshot", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.root, args.output, args.research_plan, args.rights_snapshot)
    print(json.dumps({"status": receipt["status"], "arms": receipt["arms"]}, sort_keys=True))


if __name__ == "__main__":
    main()
