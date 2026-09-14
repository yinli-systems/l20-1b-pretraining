#!/usr/bin/env python3
"""Frozen development selection for natural-router adaptation checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from safetensors.torch import load_file

from evidence_acquisition import EvidenceAcquisitionRouter, choose_evidence_action
from evaluate_openimages_evidence_router_zero_shot_v1 import (
    bbox_targets,
    localization_flags,
    wilson_interval,
)
from modeling import freeze
from train_openimages_evidence_router_adaptation_v1 import (
    cache_vision_rows,
    load_split,
    prompt_only_rows,
)
from train_highres_evidence_router_v1 import collate_true_answers
from train_stage_a_full_token import release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_openimages_evidence_router_adaptation_development_selection_v1"
STOP = 196


def proportion(successes: int, total: int) -> dict[str, Any]:
    return wilson_interval(successes, total)


def localization_summary(rows: list[dict[str, Any]], actions: list[int]) -> dict[str, Any]:
    point = [(row, action) for row, action in zip(rows, actions) if row["expected_action"] == "POINT"]
    stop = [(row, action) for row, action in zip(rows, actions) if row["expected_action"] == "STOP"]
    if not point or not stop:
        raise RuntimeError("evaluation requires both POINT and STOP rows")
    flags = []
    for row, action in point:
        center, region = bbox_targets(row["source_bbox_xmin_xmax_ymin_ymax"], 14)
        flags.append(localization_flags(action, center, region, 14) if action != STOP else {
            "exact_center": False,
            "within_one_center": False,
            "inside_bbox": False,
            "chebyshev_center_error": 14,
        })
    action_successes = sum(action != STOP for _, action in point) + sum(action == STOP for _, action in stop)
    return {
        "rows": len(rows),
        "point_rows": len(point),
        "stop_rows": len(stop),
        "action_accuracy": proportion(action_successes, len(rows)),
        "point_recall": proportion(sum(action != STOP for _, action in point), len(point)),
        "stop_accuracy": proportion(sum(action == STOP for _, action in stop), len(stop)),
        "exact_center": proportion(sum(flag["exact_center"] for flag in flags), len(flags)),
        "within_one_center": proportion(sum(flag["within_one_center"] for flag in flags), len(flags)),
        "inside_bbox": proportion(sum(flag["inside_bbox"] for flag in flags), len(flags)),
        "mean_chebyshev_center_error": sum(flag["chebyshev_center_error"] for flag in flags) / len(flags),
    }


def synthetic_summary(rows: list[dict[str, Any]], actions: list[int]) -> dict[str, Any]:
    point = [(row, action) for row, action in zip(rows, actions) if row["expected_action"] == "POINT"]
    stop = [(row, action) for row, action in zip(rows, actions) if row["expected_action"] == "STOP"]
    if not point or not stop:
        raise RuntimeError("synthetic evaluation requires both POINT and STOP rows")
    exact, within = 0, 0
    for row, action in point:
        target = int(row["target_patch_index_14x14"])
        if action != STOP:
            exact += action == target
            ay, ax = divmod(action, 14)
            ty, tx = divmod(target, 14)
            within += max(abs(ay - ty), abs(ax - tx)) <= 1
    action_successes = sum(action != STOP for _, action in point) + sum(action == STOP for _, action in stop)
    return {
        "rows": len(rows),
        "point_rows": len(point),
        "stop_rows": len(stop),
        "action_accuracy": proportion(action_successes, len(rows)),
        "point_recall": proportion(sum(action != STOP for _, action in point), len(point)),
        "stop_accuracy": proportion(sum(action == STOP for _, action in stop), len(stop)),
        "exact_patch": proportion(exact, len(point)),
        "within_one_patch": proportion(within, len(point)),
    }


def passes_gate(result: dict[str, Any], gate: dict[str, float]) -> bool:
    natural, synthetic = result["natural"], result["synthetic_retention"]
    return (
        natural["action_accuracy"]["estimate"] >= gate["natural_minimum_action_accuracy"]
        and natural["point_recall"]["estimate"] >= gate["natural_minimum_point_recall"]
        and natural["stop_accuracy"]["estimate"] >= gate["natural_minimum_stop_accuracy"]
        and natural["inside_bbox"]["estimate"] >= gate["natural_minimum_inside_bbox"]
        and natural["within_one_center"]["estimate"] >= gate["natural_minimum_within_one_center"]
        and synthetic["exact_patch"]["estimate"] >= gate["synthetic_minimum_exact_patch"]
        and synthetic["within_one_patch"]["estimate"] >= gate["synthetic_minimum_within_one_patch"]
        and synthetic["stop_accuracy"]["estimate"] >= gate["synthetic_minimum_stop_accuracy"]
    )


def select_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [result for result in results if result["gate_passed"]]
    if not eligible:
        return None
    return max(eligible, key=lambda result: (
        result["natural"]["inside_bbox"]["estimate"],
        result["natural"]["within_one_center"]["estimate"],
        result["natural"]["action_accuracy"]["estimate"],
        result["synthetic_retention"]["exact_patch"]["estimate"],
        -int(result["step"]),
    ))


def router_actions(router, vision, text, attention, labels, protocol):
    actions, raw_actions, gains, confidences = [], [], [], []
    batch_size = int(protocol["evaluation"]["router_batch_rows"])
    with torch.inference_mode():
        for start in range(0, len(vision), batch_size):
            stop = min(len(vision), start + batch_size)
            logits, gain = router(
                vision[start:stop].cuda(),
                text[start:stop].cuda(),
                attention[start:stop].cuda(),
                labels[start:stop].cuda(),
            )
            deployed = choose_evidence_action(
                logits,
                gain,
                float(protocol["router"]["minimum_predicted_gain"]),
                float(protocol["router"]["minimum_action_probability"]),
            )
            probability = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
            actions.extend(deployed.cpu().tolist())
            raw_actions.extend(logits.argmax(dim=-1).cpu().tolist())
            gains.extend(gain.cpu().tolist())
            confidences.extend(probability.cpu().tolist())
    return actions, raw_actions, gains, confidences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS or protocol.get("parameter_update_authorized") is not False:
        raise SystemExit("development-only frozen evaluation is required")
    if protocol.get("allowed_splits") != {"natural": "development", "synthetic": "development"}:
        raise SystemExit("unexpected split authorization")
    for path, key in (
        (Path(__file__), "evaluator_sha256"),
        (ROOT / "test_evaluate_openimages_evidence_router_adaptation_v1.py", "test_sha256"),
        (ROOT / "train_openimages_evidence_router_adaptation_v1.py", "trainer_sha256"),
        (ROOT / "evidence_acquisition.py", "router_module_sha256"),
        (ROOT / "modeling.py", "modeling_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"prerequisite hash mismatch: {name}")
    output = Path(protocol["output"]["receipt"])
    predictions = Path(protocol["output"]["predictions"])
    if output.exists() or predictions.exists():
        raise FileExistsError("development output already exists")

    natural_rows = load_split(Path(protocol["data"]["natural_manifest"]), "development")
    synthetic_rows = load_split(Path(protocol["data"]["synthetic_manifest"]), "development")
    if len(natural_rows) != int(protocol["data"]["natural_expected_rows"]):
        raise RuntimeError("unexpected natural development rows")
    if len(synthetic_rows) != int(protocol["data"]["synthetic_expected_rows"]):
        raise RuntimeError("unexpected synthetic development rows")

    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel
    parents = protocol["parents"]
    language_root = Path(parents["language"])
    if sha256_file(language_root / "release-manifest.json") != parents["language_release_manifest_sha256"]:
        raise SystemExit("language release mismatch")
    language_before = release_records(language_root)
    vision_root = Path(parents["vision"])
    for name, expected in parents["vision_artifacts"].items():
        if sha256_file(vision_root / name) != expected:
            raise SystemExit(f"vision parent mismatch: {name}")
    adapter_root = Path(parents["language_adapter"])
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(adapter_root / name) != expected:
            raise SystemExit(f"language adapter mismatch: {name}")

    dtype = torch.bfloat16
    processor = AutoImageProcessor.from_pretrained(vision_root, local_files_only=True, use_fast=False)
    vision_model = SiglipVisionModel.from_pretrained(vision_root, dtype=dtype, local_files_only=True).cuda()
    freeze(vision_model)
    natural_vision = cache_vision_rows(
        natural_rows, Path(protocol["data"]["natural_root"]), protocol, processor, vision_model, dtype
    )
    synthetic_vision = cache_vision_rows(
        synthetic_rows, Path(protocol["data"]["synthetic_root"]), protocol, processor, vision_model, dtype
    )
    del vision_model
    torch.cuda.empty_cache()

    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(language_root, local_files_only=True)
    language = AutoModelForCausalLM.from_pretrained(
        language_root, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(language, adapter_root, is_trainable=False, local_files_only=True)
    freeze(language)
    encoded = []
    text_cache = []
    for rows in (natural_rows, synthetic_rows):
        ids, attention, labels = collate_true_answers(
            tokenizer, prompt_only_rows(rows), int(protocol["features"]["max_text_tokens"])
        )
        with torch.inference_mode():
            text = language.get_input_embeddings()(ids.cuda()).cpu().to(dtype)
        encoded.append((attention, labels))
        text_cache.append(text)
    del language
    torch.cuda.empty_cache()

    all_predictions, results = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    checkpoints = [protocol["parent_router"], *protocol["checkpoints"]]
    for checkpoint in checkpoints:
        path = Path(checkpoint["path"])
        if sha256_file(path) != checkpoint["sha256"]:
            raise SystemExit(f"router checkpoint mismatch: {path}")
        router = EvidenceAcquisitionRouter(
            int(protocol["router"]["vision_dim"]), int(protocol["router"]["language_dim"]),
            int(protocol["router"]["rank"]),
        ).to("cuda", dtype=dtype)
        router.load_state_dict(load_file(path, device="cpu"), strict=True)
        freeze(router)
        natural_actions = router_actions(
            router, natural_vision, text_cache[0], encoded[0][0], encoded[0][1], protocol
        )
        synthetic_actions = router_actions(
            router, synthetic_vision, text_cache[1], encoded[1][0], encoded[1][1], protocol
        )
        result = {
            "name": checkpoint["name"],
            "step": checkpoint["step"],
            "path": str(path),
            "sha256": checkpoint["sha256"],
            "natural": localization_summary(natural_rows, natural_actions[0]),
            "synthetic_retention": synthetic_summary(synthetic_rows, synthetic_actions[0]),
        }
        result["gate_passed"] = passes_gate(result, protocol["gate"])
        results.append(result)
        for split_name, rows, outputs in (
            ("natural_development", natural_rows, natural_actions),
            ("synthetic_development", synthetic_rows, synthetic_actions),
        ):
            deployed, raw, gains, confidence = outputs
            for row, action, raw_action, gain, prob in zip(rows, deployed, raw, gains, confidence):
                all_predictions.append({
                    "checkpoint": checkpoint["name"],
                    "step": checkpoint["step"],
                    "split": split_name,
                    "family_id": row["family_id"],
                    "variant": row["variant"],
                    "expected_action": row["expected_action"],
                    "target_patch": row.get("target_patch_index_14x14"),
                    "raw_action": raw_action,
                    "deployed_action": action,
                    "predicted_gain": gain,
                    "max_action_probability": prob,
                })
        del router
        torch.cuda.empty_cache()

    selected = select_result(results)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("x") as handle:
        for record in all_predictions:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_openimages_evidence_router_adaptation_development_selection",
        "completed_at": utc_now(),
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "parameter_updates": 0,
        "confirmation_rows_accessed": 0,
        "natural_development_rows": len(natural_rows),
        "synthetic_development_rows": len(synthetic_rows),
        "results": results,
        "selection": selected,
        "successor_status": "authorize_fresh_natural_confirmation" if selected else "reject_adaptation_v1",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "predictions": {"path": str(predictions), "sha256": sha256_file(predictions), "rows": len(all_predictions)},
        "parents_verified_unchanged": release_records(language_root) == language_before,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
