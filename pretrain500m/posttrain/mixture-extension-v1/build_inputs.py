#!/usr/bin/env python3
"""Build the hash-bound 1.074B-token R4 extension input and protocol."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from decimal import Decimal
from pathlib import Path


SEQUENCE_LENGTH = 2048
TARGET_BLOCKS = 524_288
TARGET_TOKENS = TARGET_BLOCKS * SEQUENCE_LENGTH
BASE_SHA256 = "13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf"
PARENT_BUILD_SHA256 = "8c61ddb49378ca7359cec8b623439ef62ed8840acea17a0eea4b5121243dced7"
PARENT_MANIFEST_SHA256 = "16fe10c43470a95e5bc8eff2e41875abec7c5a58991ca768ca27ec3f424b3bb0"
PARENT_ADMISSION_SHA256 = "4cea8abc46218f679eac02f31a0b8603a8e6b56bfdf331b7d0ded23bcb08e70e"
CAPABILITY_RESULT_SHA256 = "f14e4bead964ec758fcea8b1f42ae9294c9f37a9bb069dace95b096cee2b8d62"
VALIDATION_SHA256 = "429c1ab33bdb8a92214ffdd2327462a05275b29fbd60abecc1d98b9fad634af3"
RECIPE = "R4_current_code_guard"
REQUIRED_CHECKS = (
    "licenses",
    "content_quality",
    "cross_source_deduplication",
    "benchmark_decontamination",
    "family_disjoint_splits",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scaled_quotas(parent_quotas: dict[str, int]) -> dict[str, int]:
    result = {source: int(blocks) * 2 for source, blocks in parent_quotas.items()}
    if sum(result.values()) != TARGET_BLOCKS:
        raise ValueError("scaled source quotas do not sum to the exact target")
    return result


def remaining_capacity(source: dict) -> int:
    unique_blocks = sum(int(shard["blocks"]) for shard in source["shards"])
    cap = Decimal(str(source["max_cumulative_epochs"]))
    prior_blocks = Decimal(int(source.get("prior_prediction_tokens", 0))) / Decimal(SEQUENCE_LENGTH)
    return int(cap * unique_blocks - prior_blocks)


def build(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    if output.exists():
        raise ValueError("extension input output already exists")
    parent = root / "source/mixture-pilot-inputs-v1"
    parent_build_path = parent / "build-receipt.json"
    parent_manifest_path = parent / "manifests" / f"{RECIPE}.json"
    parent_admission_path = parent / "admissions" / f"{RECIPE}.admission.json"
    capability_path = root / "posttrain/mixture-pilot-eval-v1/r4-lr6e5-seed20260916-1591424/result.json"
    validation_path = root / "data/development-masked-v2/development-mixture.json"
    base_path = root / "formal/run-v5/resume.pt"
    expected = (
        (parent_build_path, PARENT_BUILD_SHA256),
        (parent_manifest_path, PARENT_MANIFEST_SHA256),
        (parent_admission_path, PARENT_ADMISSION_SHA256),
        (capability_path, CAPABILITY_RESULT_SHA256),
        (validation_path, VALIDATION_SHA256),
        (base_path, BASE_SHA256),
    )
    for path, expected_sha in expected:
        if digest(path) != expected_sha:
            raise ValueError(f"input identity mismatch: {path}")

    parent_build = json.loads(parent_build_path.read_text())
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_admission = json.loads(parent_admission_path.read_text())
    capability = json.loads(capability_path.read_text())
    if parent_build.get("status") != "PASS_INPUTS_ADMITTED_FOR_EXPLORATORY_PILOTS":
        raise ValueError("parent input build did not pass")
    if parent_admission.get("status") != "PASS" or any(
        parent_admission.get("checks", {}).get(check) != "PASS" for check in REQUIRED_CHECKS
    ):
        raise ValueError("parent admission is incomplete")
    if capability.get("status") != "PASS_ADAPTIVE_SEVEN_TASK_CAPABILITY_SCREEN":
        raise ValueError("R4 capability screen did not pass")
    if capability.get("candidate", {}).get("id") != "r4-lr6e5-seed20260916":
        raise ValueError("capability result is not the frozen R4 candidate")
    if float(capability.get("aggregate_delta_vs_base", 0.0)) <= 0.0:
        raise ValueError("R4 did not retain a positive aggregate delta over Base")

    quotas = scaled_quotas(parent_manifest["source_block_quotas"])
    sources = {source["id"]: source for source in parent_manifest["sources"]}
    if set(quotas) != set(sources):
        raise ValueError("parent source set differs from quota set")
    capacity = {}
    for source_id, blocks in quotas.items():
        available = remaining_capacity(sources[source_id])
        capacity[source_id] = {
            "requested_blocks": blocks,
            "remaining_allowed_blocks": available,
            "headroom_blocks": available - blocks,
        }
        if blocks > available:
            raise ValueError(f"{source_id} exceeds cumulative repeat cap")

    output.mkdir()
    manifest = {
        "schema": "p529m-packed-mixture-v3",
        "protocol_id": "p529m-r4-two-seed-extension-v1",
        "recipe": RECIPE,
        "sequence_length": SEQUENCE_LENGTH,
        "tokenizer_sha256": parent_manifest["tokenizer_sha256"],
        "source_block_quotas": quotas,
        "sources": [sources[source_id] for source_id in sorted(sources)],
        "evidence": {
            "parent_build_receipt_sha256": PARENT_BUILD_SHA256,
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "parent_admission_sha256": PARENT_ADMISSION_SHA256,
            "capability_result_sha256": CAPABILITY_RESULT_SHA256,
            "development_manifest_sha256": VALIDATION_SHA256,
        },
    }
    manifest_path = output / f"{RECIPE}.json"
    write_json(manifest_path, manifest)
    protocol = {
        "schema": "p529m-r4-two-seed-extension-v1",
        "status": "FROZEN_ADAPTIVE_EXTENSION",
        "data_scope": "reweighted current admitted corpus; not fresh corpus",
        "candidate": "r4-lr6e5",
        "recipe": RECIPE,
        "seeds": [20260916, 20260917],
        "target_prediction_tokens_per_run": TARGET_TOKENS,
        "steps_per_run": 512,
        "world_size_per_run": 4,
        "base_checkpoint": {"path": str(base_path), "sha256": BASE_SHA256, "step": 7629},
        "training": {
            "architecture": "deep",
            "microbatch_per_gpu": 4,
            "gradient_accumulation": 64,
            "global_prediction_tokens_per_step": 2_097_152,
            "peak_learning_rate": 0.00006,
            "warmup_prediction_tokens": 67_108_864,
            "schedule": "warmup-stable-cosine-decay",
            "deterministic": True,
            "checkpoint_mode": "model-only-final",
        },
        "mfu_gate": {
            "dense_bf16_tflops_per_rtx5090": 209.5,
            "rolling_window_steps": 10,
            "grace_steps": 5,
            "minimum_strictly_greater_than": 0.70,
        },
        "selection_evidence": {
            "r4_capability_result_sha256": CAPABILITY_RESULT_SHA256,
            "adaptive_seven_task_delta_vs_base": capability["aggregate_delta_vs_base"],
            "formal_promotion": False,
        },
    }
    protocol_path = output / "protocol.json"
    write_json(protocol_path, protocol)
    admission = {
        "schema": "p529m-corpus-admission-v1",
        "status": "PASS",
        "recipe": RECIPE,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_manifest_sha256": digest(manifest_path),
        "validation_manifest_sha256": VALIDATION_SHA256,
        "protocol_sha256": digest(protocol_path),
        "checks": {check: "PASS" for check in REQUIRED_CHECKS},
        "cumulative_capacity": capacity,
        "limitations": [
            "Admission is inherited for the exact parent shards and checked repeat caps.",
            "This is current-corpus reweighting, not fresh-corpus training.",
            "The capability result is adaptive and cannot establish market superiority.",
        ],
    }
    admission_path = output / f"{RECIPE}.admission.json"
    write_json(admission_path, admission)
    receipt = {
        "schema": "p529m-r4-two-seed-extension-inputs-v1",
        "status": "PASS_EXTENSION_INPUTS_ADMITTED",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_prediction_tokens_per_run": TARGET_TOKENS,
        "seeds": [20260916, 20260917],
        "manifest_sha256": digest(manifest_path),
        "admission_sha256": digest(admission_path),
        "protocol_sha256": digest(protocol_path),
        "base_checkpoint_sha256": BASE_SHA256,
        "capability_result_sha256": CAPABILITY_RESULT_SHA256,
        "cumulative_capacity_passed": True,
        "training_launched": False,
        "formal_promotion": False,
        "claim_boundary": "R4 current-corpus extension inputs admitted; training and capability improvement remain unverified",
    }
    write_json(output / "build-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
