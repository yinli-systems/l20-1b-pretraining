#!/usr/bin/env python3
"""Fail-closed validation for the tuned plain-CE binding screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


def expected_optimizer_steps(rows: int, micro_batch: int, accumulation: int) -> int:
    if min(rows, micro_batch, accumulation) < 1:
        raise ValueError("row and batch counts must be positive")
    if rows % (micro_batch * accumulation):
        raise ValueError("training rows must divide the effective batch exactly")
    return rows // (micro_batch * accumulation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "authorized_binding_answer_only_development_v1":
        raise SystemExit("tuned-CE protocol is not authorized")
    if protocol.get("experiment_id") != "binding_tuned_plain_ce_v1":
        raise SystemExit("unexpected tuned-CE experiment id")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    root = Path(__file__).parent
    for filename, key in (
        ("validate_binding_tuned_ce_protocol.py", "validator_sha256"),
        ("train_stage_c_counterfactual.py", "trainer_sha256"),
        ("evaluate_stage_a_visual_floor.py", "evaluator_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("scoring_contract.py", "scoring_contract_sha256"),
    ):
        if sha256_file(root / filename) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {filename}")
    prerequisite_statuses = {
        "e1_evaluation": "complete_development_only",
        "representation_probe": "complete_diagnostic_only",
        "text_oracle": "complete_diagnostic_only",
    }
    prerequisite_summaries = {}
    for name, expected_status in prerequisite_statuses.items():
        item = protocol["prerequisites"][name]
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise SystemExit(f"prerequisite hash mismatch: {name}")
        summary = json.loads(path.read_text())
        if summary.get("status") != expected_status or summary.get("final_test_used") is not False:
            raise SystemExit(f"prerequisite status mismatch: {name}")
        prerequisite_summaries[name] = {
            "status": summary["status"],
            "final_test_used": summary["final_test_used"],
        }
    data = protocol["data"]
    manifest = Path(data["manifest"])
    if sha256_file(manifest) != data["manifest_sha256"]:
        raise SystemExit("manifest hash mismatch")
    for name, item in data["evidence"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"data evidence hash mismatch: {name}")
    parents = protocol["parents"]
    if sha256_file(Path(parents["bridge"])) != parents["bridge_sha256"]:
        raise SystemExit("parent bridge hash mismatch")
    adapter = Path(parents["language_adapter"])
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(adapter / name) != expected:
            raise SystemExit(f"parent adapter hash mismatch: {name}")
    optimization = protocol["optimization"]
    calculated_steps = expected_optimizer_steps(
        data["scene_families"],
        optimization["micro_batch_families"],
        optimization["gradient_accumulation_steps"],
    )
    if calculated_steps != optimization["max_optimizer_steps"]:
        raise SystemExit("optimizer-step count mismatch")
    if protocol["architecture"].get("output_visual_tokens") != 49:
        raise SystemExit("tuned-CE screen must preserve 49 visual tokens")
    write_json_atomic(args.output, {
        "schema_version": "2026-09-14-v1",
        "status": "passed_fail_closed_preflight",
        "completed_at": utc_now(),
        "protocol_sha256": sha256_file(args.protocol),
        "source_hashes_verified": True,
        "parent_hashes_verified": True,
        "data_evidence_verified": True,
        "prerequisites": prerequisite_summaries,
        "optimizer_steps": calculated_steps,
        "final_test_use_authorized": False,
    })


if __name__ == "__main__":
    main()
