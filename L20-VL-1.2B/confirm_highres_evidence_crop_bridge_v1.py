#!/usr/bin/env python3
"""One-shot sealed confirmation of the selected isolated crop bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from evaluate_highres_evidence_crop_bridge_v1 import score_condition
from evaluate_highres_evidence_responder_floor_v1 import (
    cache_features,
    condition_metrics,
    mismatched_crop_mapping,
    paired_bootstrap,
)
from modeling import bridge_from_architecture, freeze
from train_stage_a_full_token import release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_highres_evidence_crop_bridge_one_shot_sealed_confirmation_v1"


def load_sealed_rows(manifest: Path):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == "sealed_test"]
    if not selected or any(row["split"] != "sealed_test" for row in selected):
        raise RuntimeError("sealed confirmation split isolation failed")
    selected.sort(key=lambda row: (row["family_id"], row["variant"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS or protocol.get("one_shot") is not True:
        raise SystemExit("one-shot sealed confirmation is not authorized")
    if protocol.get("parameter_update_authorized") is not False:
        raise SystemExit("parameter updates must remain forbidden")
    for path, key in (
        (Path(__file__), "confirmer_sha256"),
        (ROOT / "test_confirm_highres_evidence_crop_bridge_v1.py", "test_sha256"),
        (ROOT / "evaluate_highres_evidence_crop_bridge_v1.py", "shared_crop_evaluator_sha256"),
        (ROOT / "evaluate_highres_evidence_responder_floor_v1.py", "shared_floor_evaluator_sha256"),
        (ROOT / "train_highres_evidence_crop_bridge_v1.py", "shared_trainer_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"prerequisite hash mismatch: {name}")
    receipt_path = Path(protocol["output"]["receipt"])
    predictions_path = Path(protocol["output"]["predictions"])
    for path in (receipt_path, predictions_path):
        if path.exists():
            raise FileExistsError(path)

    parents = protocol["parents"]
    language_root = Path(parents["language"])
    if sha256_file(language_root / "release-manifest.json") != parents["language_release_manifest_sha256"]:
        raise SystemExit("language release manifest mismatch")
    language_files = release_records(language_root)
    vision_root = Path(parents["vision"])
    for name, expected in parents["vision_artifacts"].items():
        if sha256_file(vision_root / name) != expected:
            raise SystemExit(f"vision artifact mismatch: {name}")
    parent_bridge_path = Path(parents["bridge"])
    if sha256_file(parent_bridge_path) != parents["bridge_sha256"]:
        raise SystemExit("parent bridge mismatch")
    adapter_root = Path(parents["language_adapter"])
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(adapter_root / name) != expected:
            raise SystemExit(f"language adapter mismatch: {name}")
    selected_checkpoint = protocol["selected_checkpoint"]
    if sha256_file(Path(selected_checkpoint["path"])) != selected_checkpoint["sha256"]:
        raise SystemExit("selected crop bridge mismatch")

    from peft import PeftModel
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(language_root, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_root, local_files_only=True, use_fast=False)
    vision = SiglipVisionModel.from_pretrained(vision_root, dtype=dtype, local_files_only=True).cuda()
    freeze(vision)
    language = AutoModelForCausalLM.from_pretrained(
        language_root, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(language, adapter_root, is_trainable=False, local_files_only=True)
    freeze(language)
    parent_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    parent_bridge.load_state_dict(load_file(parent_bridge_path, device="cpu"), strict=True)
    freeze(parent_bridge)
    crop_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    crop_bridge.load_state_dict(load_file(selected_checkpoint["path"], device="cpu"), strict=True)
    freeze(crop_bridge)

    rows = load_sealed_rows(Path(protocol["data"]["manifest"]))
    if (
        len(rows) != int(protocol["data"]["expected_rows"])
        or len({row["family_id"] for row in rows}) != int(protocol["data"]["expected_families"])
    ):
        raise RuntimeError("unexpected sealed-test corpus size")
    local_rows = [row for row in rows if row["task"] == "read_local_digit"]
    global_rows = [row for row in rows if row["task"] == "read_global_border"]
    mismatch = mismatched_crop_mapping(rows)
    protocol["runtime_checkpoint_step"] = selected_checkpoint["step"]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    caches = cache_features(rows, protocol, processor, vision, dtype)
    condition_rows = {
        "global": rows,
        "no_image": rows,
        "crop": local_rows,
        "global_crop": local_rows,
        "global_mismatched_crop": local_rows,
    }
    predictions, records, invariance = {}, [], {}
    with torch.inference_mode():
        for condition, subset in condition_rows.items():
            normal, normal_records = score_condition(
                subset, condition, False, protocol, language, parent_bridge, crop_bridge,
                tokenizer, caches, mismatch,
            )
            reverse, reverse_records = score_condition(
                subset, condition, True, protocol, language, parent_bridge, crop_bridge,
                tokenizer, caches, mismatch,
            )
            predictions[condition] = normal
            records.extend(normal_records + reverse_records)
            invariance[condition] = {
                "rows": len(subset),
                "prediction_mismatches": sum(a != b for a, b in zip(normal, reverse)),
                "passes": normal == reverse,
            }
    global_by_id = {
        (row["family_id"], row["variant"]): prediction
        for row, prediction in zip(rows, predictions["global"])
    }
    local_global = [global_by_id[(row["family_id"], row["variant"])] for row in local_rows]
    global_task_predictions = [global_by_id[(row["family_id"], row["variant"])] for row in global_rows]
    policy_by_id = {
        **{(row["family_id"], row["variant"]): prediction for row, prediction in zip(local_rows, predictions["global_crop"])},
        **{(row["family_id"], row["variant"]): prediction for row, prediction in zip(global_rows, global_task_predictions)},
    }
    policy_predictions = [policy_by_id[(row["family_id"], row["variant"])] for row in rows]
    metrics = {
        condition: condition_metrics(condition_rows[condition], values)
        for condition, values in predictions.items()
    }
    metrics["global_task_only"] = condition_metrics(global_rows, global_task_predictions)
    metrics["oracle_action_policy_all_tasks"] = condition_metrics(rows, policy_predictions)
    statistics = protocol["statistics"]
    deltas = {
        "crop_minus_global": paired_bootstrap(
            local_rows, predictions["crop"], local_global,
            int(statistics["bootstrap_samples"]), int(statistics["seed"]),
        ),
        "global_crop_minus_global": paired_bootstrap(
            local_rows, predictions["global_crop"], local_global,
            int(statistics["bootstrap_samples"]), int(statistics["seed"]) + 1,
        ),
        "global_crop_minus_mismatched_crop": paired_bootstrap(
            local_rows, predictions["global_crop"], predictions["global_mismatched_crop"],
            int(statistics["bootstrap_samples"]), int(statistics["seed"]) + 2,
        ),
    }
    gate = protocol["gate"]
    gate_passed = (
        metrics["global_crop"]["row_accuracy"] >= float(gate["minimum_global_crop_row_accuracy"])
        and metrics["global_crop"]["family_joint_accuracy"] >= float(gate["minimum_global_crop_family_joint_accuracy"])
        and deltas["global_crop_minus_global"]["ci95_low"] > float(gate["minimum_global_crop_minus_global_ci95_low"])
        and deltas["global_crop_minus_mismatched_crop"]["ci95_low"] > float(gate["minimum_true_minus_mismatched_ci95_low"])
        and all(item["passes"] for item in invariance.values())
    )
    with predictions_path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_highres_evidence_crop_bridge_one_shot_sealed_confirmation",
        "completed_at": utc_now(),
        "decision": "confirm_crop_responder_mechanism" if gate_passed else "reject_crop_responder_mechanism",
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "selected_checkpoint": selected_checkpoint,
        "verified_parent_records": {
            "language": language_files,
            "vision_artifacts": parents["vision_artifacts"],
            "bridge_sha256": parents["bridge_sha256"],
            "language_adapter_files": parents["language_adapter_files"],
        },
        "rows": {
            "all_sealed_test": len(rows),
            "local_digit": len(local_rows),
            "global_border": len(global_rows),
        },
        "metrics": metrics,
        "paired_deltas": deltas,
        "candidate_order_invariance": invariance,
        "gate_passed": gate_passed,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "bytes": predictions_path.stat().st_size,
        },
        "parameter_updates": 0,
        "sealed_test_access_count": 1,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({
        key: receipt[key] for key in (
            "decision", "rows", "metrics", "paired_deltas",
            "candidate_order_invariance", "elapsed_seconds", "peak_cuda_memory_bytes",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
