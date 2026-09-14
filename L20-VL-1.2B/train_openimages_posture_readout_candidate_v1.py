#!/usr/bin/env python3
"""Fit one deployable posture readout after frozen cross-fit selection.

This script deliberately performs no development-set benchmark evaluation.  It
uses the selected query-CE recipe unchanged, fits a fresh query readout on all
77 audited development pairs, and emits a composition receipt.  The result is
only a development-selected candidate until a disjoint sealed confirmation is
run.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_openimages_posture_readout_all_development_fit_v1"
SELECTED_ARM = "query_ce"
FROZEN_OPTIMIZATION = {
    "seed": 20261005,
    "training_view": "marked",
    "epochs": 12,
    "batch_pairs": 4,
    "readout_learning_rate": 0.0003,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "minimum_learning_rate_ratio": 0.1,
    "gradient_clip_norm": 1.0,
    "pair_margin": 1.0,
    "max_text_tokens": 64,
    "checkpoint_selection": "none; use the final prespecified epoch",
    "hyperparameter_selection_on_this_dataset": "none",
}


def validate_selection_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("status") != "complete_audited_method_development_only":
        raise RuntimeError("method-development analysis is not complete")
    decision = receipt.get("decision", {})
    if decision.get("selected_for_new_sealed_confirmation") != SELECTED_ARM:
        raise RuntimeError("query_ce was not the frozen selected arm")
    if decision.get("do_not_select") != "grounded_query":
        raise RuntimeError("grounded_query rejection is missing")


def validate_frozen_recipe(protocol: dict[str, Any]) -> None:
    if protocol.get("selected_arm") != SELECTED_ARM:
        raise RuntimeError("selected arm changed")
    if protocol.get("loss_weights") != {
        "answer": 1.0,
        "address": 0.0,
        "pair_rank": 0.0,
    }:
        raise RuntimeError("selected query-CE objective changed")
    if protocol.get("optimization") != FROZEN_OPTIMIZATION:
        raise RuntimeError("selected optimization recipe changed")


def parent_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    from train_stage_a_full_token import sha256_file

    parents = protocol["parents"]
    values = {"bridge.safetensors": sha256_file(Path(parents["bridge"]))}
    for name in sorted(parents["language_adapter_files"]):
        values[f"language_adapter/{name}"] = sha256_file(
            Path(parents["language_adapter"]) / name
        )
    return values


def validate_protocol(path: Path) -> dict[str, Any]:
    from train_stage_a_full_token import sha256_file

    protocol = json.loads(path.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("all-development candidate fit is not authorized")
    if protocol.get("model_parameter_updates_authorized") is not True:
        raise RuntimeError("candidate parameter update is not authorized")
    for field in (
        "formal_test_present",
        "confirmation_data_present",
        "model_evaluation_authorized",
        "development_benchmark_scoring_authorized",
    ):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    validate_frozen_recipe(protocol)

    source = protocol["source_code"]
    checks = (
        (Path(__file__), "trainer_sha256"),
        (ROOT / "test_train_openimages_posture_readout_candidate_v1.py", "test_sha256"),
        (ROOT / "train_openimages_posture_readout_crossfit_v1.py", "crossfit_trainer_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
    )
    for file_path, key in checks:
        if sha256_file(file_path) != source[key]:
            raise RuntimeError(f"source hash mismatch: {file_path.name}")

    data = protocol["data"]
    for key in ("manifest", "selection_analysis", "crossfit_run"):
        item = data[key]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen evidence hash mismatch: {key}")
    analysis = json.loads(Path(data["selection_analysis"]["path"]).read_text())
    validate_selection_receipt(analysis)
    if analysis["run"]["sha256"] != data["crossfit_run"]["sha256"]:
        raise RuntimeError("selection analysis does not name the frozen cross-fit run")

    actual_parents = parent_hashes(protocol)
    expected_parents = {"bridge.safetensors": protocol["parents"]["bridge_sha256"]}
    expected_parents.update(
        {
            f"language_adapter/{name}": digest
            for name, digest in protocol["parents"]["language_adapter_files"].items()
        }
    )
    if actual_parents != expected_parents:
        raise RuntimeError("parent model hash mismatch")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, help="Smoke only; marks output non-candidate.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_epochs is not None and not 1 <= args.max_epochs < FROZEN_OPTIMIZATION["epochs"]:
        raise ValueError("max epochs must be a strict positive smoke cap")

    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(protocol_path)

    from train_openimages_posture_readout_crossfit_v1 import (
        cache_vision_features,
        flatten_pairs,
        load_jsonl,
        train_fold,
        verify_images,
    )

    pairs = load_jsonl(Path(protocol["data"]["manifest"]["path"]))
    if len(pairs) != int(protocol["data"]["pairs"]):
        raise RuntimeError("development pair count changed")
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise RuntimeError("duplicate development pair id")
    targets = flatten_pairs(pairs)
    if len(targets) != int(protocol["data"]["targets"]):
        raise RuntimeError("development target count changed")
    image_audit = verify_images(targets)
    if not image_audit["all_match_manifest"]:
        raise RuntimeError("image integrity audit failed")

    import torch
    from peft import PeftModel
    from transformers import (
        AutoImageProcessor,
        AutoModelForCausalLM,
        AutoTokenizer,
        SiglipVisionModel,
    )

    from modeling import freeze
    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16
    before_parent_hashes = parent_hashes(protocol)
    started = time.monotonic()
    args.output.mkdir(parents=True)
    run_path = args.output / "run.json"
    run = {
        "schema_version": "2026-09-14-v1",
        "status": "running",
        "started_at": utc_now(),
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "selection_provenance": protocol["data"]["selection_analysis"],
        "device": torch.cuda.get_device_name(),
        "pairs": len(pairs),
        "targets": len(targets),
        "image_integrity_audit": image_audit,
        "parent_hashes_before": before_parent_hashes,
        "training_scope": "fresh rank-64 query readout only",
        "development_benchmark_scoring_performed": False,
        "confirmation_evaluation_performed": False,
    }
    write_json_atomic(run_path, run)

    tokenizer = AutoTokenizer.from_pretrained(
        protocol["parents"]["language"], local_files_only=True
    )
    processor = AutoImageProcessor.from_pretrained(
        protocol["parents"]["vision"], local_files_only=True, use_fast=False
    )
    vision = SiglipVisionModel.from_pretrained(
        protocol["parents"]["vision"], dtype=dtype, local_files_only=True
    ).cuda()
    freeze(vision)
    feature_cache, _ = cache_vision_features(
        targets, ("marked",), protocol, processor, vision, dtype
    )
    del vision, processor
    gc.collect()
    torch.cuda.empty_cache()

    language = AutoModelForCausalLM.from_pretrained(
        protocol["parents"]["language"],
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).cuda()
    language = PeftModel.from_pretrained(
        language,
        protocol["parents"]["language_adapter"],
        is_trainable=False,
        local_files_only=True,
    )
    freeze(language)

    torch.cuda.reset_peak_memory_stats()
    bridge, fit_receipt = train_fold(
        protocol,
        SELECTED_ARM,
        0,
        pairs,
        feature_cache,
        language,
        tokenizer,
        dtype,
        args.output,
        args.max_epochs,
    )
    checkpoint = Path(fit_receipt["checkpoint"])
    log_path = checkpoint.parent / "train.jsonl"
    after_parent_hashes = parent_hashes(protocol)
    if after_parent_hashes != before_parent_hashes:
        raise RuntimeError("immutable parent changed during candidate fit")

    smoke = args.max_epochs is not None
    fit_receipt.update(
        {
            "status": "complete_smoke_only" if smoke else "complete_all_development_candidate_fit",
            "role": "engineering_smoke_only" if smoke else "development_selected_candidate_pending_sealed_confirmation",
            "training_pairs": fit_receipt.pop("train_pairs"),
            "training_pair_ids": fit_receipt.pop("train_pair_ids"),
            "initialization_seed_offset": fit_receipt.pop("fold"),
            "development_benchmark_scoring_performed": False,
            "confirmation_evaluation_performed": False,
            "checkpoint_sha256": sha256_file(checkpoint),
            "train_log_sha256": sha256_file(log_path),
        }
    )
    write_json_atomic(checkpoint.parent / "receipt.json", fit_receipt)

    del bridge, language
    gc.collect()
    torch.cuda.empty_cache()
    run.update(
        {
            "status": "complete_smoke_only" if smoke else "complete_candidate_pending_sealed_confirmation",
            "completed_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_gib": fit_receipt["final_event"]["peak_allocated_gib"],
            "optimizer_steps": fit_receipt["optimizer_steps"],
            "trainable_parameters": fit_receipt["trainable_parameters"],
            "candidate": {
                "role": fit_receipt["role"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": fit_receipt["checkpoint_sha256"],
                "fit_receipt": str(checkpoint.parent / "receipt.json"),
                "fit_receipt_sha256": sha256_file(checkpoint.parent / "receipt.json"),
                "train_log": str(log_path),
                "train_log_sha256": fit_receipt["train_log_sha256"],
                "composition": {
                    "language": protocol["parents"]["language"],
                    "vision": protocol["parents"]["vision"],
                    "bridge_parent": protocol["parents"]["bridge"],
                    "language_adapter": protocol["parents"]["language_adapter"],
                    "query_readout": str(checkpoint),
                },
            },
            "parent_hashes_after": after_parent_hashes,
            "claim_boundary": protocol["claim_boundary"],
        }
    )
    write_json_atomic(run_path, run)
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
