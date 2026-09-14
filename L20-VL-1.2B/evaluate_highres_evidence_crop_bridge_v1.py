#!/usr/bin/env python3
"""Select an isolated crop bridge on development without touching sealed test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from safetensors.torch import load_file

from evaluate_highres_evidence_responder_floor_v1 import (
    answer_scores,
    cache_features,
    collate_candidates,
    condition_metrics,
    load_development_rows,
    mismatched_crop_mapping,
    paired_bootstrap,
)
from modeling import bridge_from_architecture, freeze
from train_highres_evidence_crop_bridge_v1 import inject_global_and_crop
from train_stage_a_full_token import release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_highres_evidence_crop_bridge_development_selection_only_v1"


def score_condition(rows, condition, reverse, protocol, language, parent_bridge, crop_bridge, tokenizer, caches, mismatch):
    global_cache, crop_cache = caches
    predictions, records = [], []
    batch_rows = int(protocol["evaluation"]["batch_rows"])
    for start in range(0, len(rows), batch_rows):
        batch = rows[start:start + batch_rows]
        input_ids, attention, labels, identities = collate_candidates(
            tokenizer, batch, reverse, int(protocol["evaluation"]["max_text_tokens"])
        )
        input_ids = input_ids.cuda(non_blocking=True)
        attention = attention.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        text = language.get_input_embeddings()(input_ids)
        row_indices = [row_index for row_index, _ in identities]
        if condition == "no_image":
            inputs, mask, targets = text, attention, labels
        else:
            global_features = torch.stack([
                global_cache[(batch[index]["family_id"], batch[index]["variant"])]
                for index in row_indices
            ]).to("cuda", non_blocking=True)
            if condition == "global":
                inputs, mask, targets = parent_bridge.inject(text, attention, labels, global_features)
            else:
                crop_keys = []
                for index in row_indices:
                    key = (batch[index]["family_id"], batch[index]["variant"])
                    crop_keys.append(key if condition in {"crop", "global_crop"} else mismatch[key])
                crop_features = torch.stack([crop_cache[key] for key in crop_keys]).to("cuda", non_blocking=True)
                if condition == "crop":
                    inputs, mask, targets = crop_bridge.inject(text, attention, labels, crop_features)
                elif condition in {"global_crop", "global_mismatched_crop"}:
                    inputs, mask, targets = inject_global_and_crop(
                        parent_bridge, crop_bridge, text, attention, labels, global_features, crop_features
                    )
                else:
                    raise ValueError(f"unknown condition: {condition}")
        logits = language(inputs_embeds=inputs, attention_mask=mask).logits
        scores = answer_scores(logits, targets, tokenizer.eos_token_id).detach().cpu().tolist()
        grouped = [dict() for _ in batch]
        for score, (row_index, candidate) in zip(scores, identities):
            grouped[row_index][candidate] = float(score)
        for row, candidate_scores in zip(batch, grouped):
            maximum = max(candidate_scores.values())
            winners = sorted(candidate for candidate, value in candidate_scores.items() if value == maximum)
            prediction = winners[0] if len(winners) == 1 else "__TIE__"
            predictions.append(prediction)
            records.append({
                "checkpoint_step": None if crop_bridge is None else protocol["runtime_checkpoint_step"],
                "family_id": row["family_id"],
                "variant": row["variant"],
                "task": row["task"],
                "answer": row["answer"],
                "condition": condition,
                "reverse_candidate_order": reverse,
                "prediction": prediction,
                "candidate_mean_answer_token_logprobs": candidate_scores,
            })
    return predictions, records


def select_checkpoint(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [result for result in results if result["gate_passed"]]
    if not eligible:
        return None
    return max(eligible, key=lambda result: (
        result["metrics"]["global_crop"]["family_joint_accuracy"],
        result["metrics"]["global_crop"]["row_accuracy"],
        result["paired_deltas"]["global_crop_minus_mismatched_crop"]["point_estimate"],
        -result["step"],
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise SystemExit("crop-bridge development selection is not authorized")
    for field in ("parameter_update_authorized", "sealed_test_access_authorized"):
        if protocol.get(field) is not False:
            raise SystemExit(f"{field} must remain false")
    for path, key in (
        (Path(__file__), "evaluator_sha256"),
        (ROOT / "test_evaluate_highres_evidence_crop_bridge_v1.py", "test_sha256"),
        (ROOT / "evaluate_highres_evidence_responder_floor_v1.py", "shared_evaluator_sha256"),
        (ROOT / "train_highres_evidence_crop_bridge_v1.py", "shared_trainer_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"prerequisite hash mismatch: {name}")
    output_receipt = Path(protocol["output"]["receipt"])
    output_predictions = Path(protocol["output"]["predictions"])
    for path in (output_receipt, output_predictions):
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
    for checkpoint in protocol["checkpoints"]:
        if sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise SystemExit(f"crop checkpoint mismatch: {checkpoint['step']}")

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

    rows = load_development_rows(Path(protocol["data"]["manifest"]))
    local_rows = [row for row in rows if row["task"] == "read_local_digit"]
    mismatch = mismatched_crop_mapping(rows)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    caches = cache_features(rows, protocol, processor, vision, dtype)
    all_records, results = [], []
    with torch.inference_mode():
        baseline_predictions = {}
        baseline_invariance = {}
        protocol["runtime_checkpoint_step"] = None
        for condition in ("global", "no_image"):
            normal, records = score_condition(
                rows, condition, False, protocol, language, parent_bridge, None, tokenizer, caches, mismatch
            )
            reverse, reverse_records = score_condition(
                rows, condition, True, protocol, language, parent_bridge, None, tokenizer, caches, mismatch
            )
            baseline_predictions[condition] = normal
            baseline_invariance[condition] = normal == reverse
            all_records.extend(records + reverse_records)
        global_by_id = {
            (row["family_id"], row["variant"]): prediction
            for row, prediction in zip(rows, baseline_predictions["global"])
        }
        local_global = [global_by_id[(row["family_id"], row["variant"])] for row in local_rows]
        for checkpoint in protocol["checkpoints"]:
            protocol["runtime_checkpoint_step"] = checkpoint["step"]
            crop_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
            crop_bridge.load_state_dict(load_file(checkpoint["path"], device="cpu"), strict=True)
            freeze(crop_bridge)
            predictions, invariance = {}, dict(baseline_invariance)
            for condition in ("crop", "global_crop", "global_mismatched_crop"):
                normal, records = score_condition(
                    local_rows, condition, False, protocol, language, parent_bridge, crop_bridge, tokenizer, caches, mismatch
                )
                reverse, reverse_records = score_condition(
                    local_rows, condition, True, protocol, language, parent_bridge, crop_bridge, tokenizer, caches, mismatch
                )
                predictions[condition] = normal
                invariance[condition] = normal == reverse
                all_records.extend(records + reverse_records)
            metrics = {
                condition: condition_metrics(local_rows, prediction)
                for condition, prediction in predictions.items()
            }
            deltas = {
                "crop_minus_global": paired_bootstrap(
                    local_rows, predictions["crop"], local_global,
                    int(protocol["statistics"]["bootstrap_samples"]), int(protocol["statistics"]["seed"]),
                ),
                "global_crop_minus_global": paired_bootstrap(
                    local_rows, predictions["global_crop"], local_global,
                    int(protocol["statistics"]["bootstrap_samples"]), int(protocol["statistics"]["seed"]) + 1,
                ),
                "global_crop_minus_mismatched_crop": paired_bootstrap(
                    local_rows, predictions["global_crop"], predictions["global_mismatched_crop"],
                    int(protocol["statistics"]["bootstrap_samples"]), int(protocol["statistics"]["seed"]) + 2,
                ),
            }
            gate = protocol["gate"]
            gate_passed = (
                metrics["global_crop"]["row_accuracy"] >= float(gate["minimum_global_crop_row_accuracy"])
                and metrics["global_crop"]["family_joint_accuracy"] >= float(gate["minimum_global_crop_family_joint_accuracy"])
                and deltas["global_crop_minus_global"]["ci95_low"] > float(gate["minimum_global_crop_minus_global_ci95_low"])
                and deltas["global_crop_minus_mismatched_crop"]["ci95_low"] > float(gate["minimum_true_minus_mismatched_ci95_low"])
                and all(invariance.values())
            )
            results.append({
                "step": checkpoint["step"],
                "path": checkpoint["path"],
                "sha256": checkpoint["sha256"],
                "metrics": metrics,
                "paired_deltas": deltas,
                "candidate_order_invariance": invariance,
                "gate_passed": gate_passed,
            })
            del crop_bridge
    selected = select_checkpoint(results)
    with output_predictions.open("x") as handle:
        for record in all_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_highres_evidence_crop_bridge_development_selection",
        "completed_at": utc_now(),
        "decision": "select_crop_bridge_checkpoint" if selected else "no_crop_bridge_checkpoint_passed",
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "verified_parent_records": {
            "language": language_files,
            "vision_artifacts": parents["vision_artifacts"],
            "bridge_sha256": parents["bridge_sha256"],
            "language_adapter_files": parents["language_adapter_files"],
        },
        "rows": {"all_development": len(rows), "local_digit": len(local_rows)},
        "baseline_metrics": {
            "global_all_tasks": condition_metrics(rows, baseline_predictions["global"]),
            "no_image_all_tasks": condition_metrics(rows, baseline_predictions["no_image"]),
        },
        "checkpoint_results": results,
        "selected": selected,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "predictions": {
            "path": str(output_predictions),
            "sha256": sha256_file(output_predictions),
            "bytes": output_predictions.stat().st_size,
        },
        "parameter_updates": 0,
        "sealed_test_rows_accessed": 0,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(output_receipt, receipt)
    print(json.dumps({
        "decision": receipt["decision"],
        "selected": selected,
        "checkpoint_results": results,
        "elapsed_seconds": receipt["elapsed_seconds"],
        "peak_cuda_memory_bytes": receipt["peak_cuda_memory_bytes"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
