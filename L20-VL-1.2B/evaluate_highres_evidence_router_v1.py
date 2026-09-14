#!/usr/bin/env python3
"""Development selection and predicted-crop evaluation for the evidence router."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file

from evaluate_highres_evidence_crop_bridge_v1 import score_condition
from evaluate_highres_evidence_responder_floor_v1 import condition_metrics, load_development_rows
from evidence_acquisition import EvidenceAcquisitionRouter, choose_evidence_action, patch_index_to_crop_box
from modeling import bridge_from_architecture, freeze, vision_features
from train_highres_evidence_router_v1 import cache_vision, collate_true_answers, responder_gains
from train_stage_a_full_token import release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_highres_evidence_router_development_selection_only_v1"


def router_metrics(rows, actions, predicted_gains, observed_gains, patch_count: int) -> dict[str, Any]:
    point = torch.tensor([row["expected_action"] == "POINT" for row in rows])
    targets = torch.tensor([
        int(row["target_patch_index_14x14"]) if row["expected_action"] == "POINT" else patch_count
        for row in rows
    ])
    actions = torch.as_tensor(actions)
    predicted_gains = torch.as_tensor(predicted_gains, dtype=torch.float32)
    observed_gains = torch.as_tensor(observed_gains, dtype=torch.float32)
    action_correct = torch.where(point, actions.ne(patch_count), actions.eq(patch_count))
    exact = actions[point].eq(targets[point])
    predicted_point = actions[point].clamp(max=patch_count - 1)
    predicted_y, predicted_x = predicted_point // 14, predicted_point % 14
    target_y, target_x = targets[point] // 14, targets[point] % 14
    chebyshev = torch.maximum((predicted_y - target_y).abs(), (predicted_x - target_x).abs())
    acquired_point = actions[point].ne(patch_count)
    bounded_error = torch.where(acquired_point, chebyshev, torch.full_like(chebyshev, 14))
    return {
        "rows": len(rows),
        "point_rows": int(point.sum()),
        "stop_rows": int((~point).sum()),
        "action_accuracy": float(action_correct.float().mean()),
        "point_recall": float(actions[point].ne(patch_count).float().mean()),
        "stop_accuracy": float(actions[~point].eq(patch_count).float().mean()),
        "exact_patch_accuracy": float(exact.float().mean()),
        "within_one_patch_accuracy": float((acquired_point & (chebyshev <= 1)).float().mean()),
        "mean_chebyshev_patch_error": float(bounded_error.float().mean()),
        "acquisition_rate": float(actions.ne(patch_count).float().mean()),
        "gain_mae": float((predicted_gains - observed_gains).abs().mean()),
        "point_gain_mae": float((predicted_gains[point] - observed_gains[point]).abs().mean()),
        "stop_gain_mae": float((predicted_gains[~point] - observed_gains[~point]).abs().mean()),
    }


def select_checkpoint(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [result for result in results if result["gate_passed"]]
    if not eligible:
        return None
    return max(eligible, key=lambda result: (
        result["metrics"]["action_accuracy"],
        result["metrics"]["within_one_patch_accuracy"],
        result["metrics"]["exact_patch_accuracy"],
        -result["metrics"]["gain_mae"],
        -result["step"],
    ))


def cache_predicted_crops(rows, actions, protocol, processor, vision, dtype):
    root = Path(protocol["data"]["root"])
    selected = [(row, int(action)) for row, action in zip(rows, actions) if int(action) < 196]
    result = {}
    batch_size = int(protocol["evaluation"]["feature_batch_images"])
    expansion = float(protocol["router"]["crop_expansion_cells"])
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            images = []
            for row, action in batch:
                with Image.open(root / row["image_path"]) as source:
                    image = source.convert("RGB")
                box = patch_index_to_crop_box(action, 14, expansion)
                pixels = tuple(round(value * image.width) for value in box)
                images.append(image.crop(pixels))
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(
                "cuda", dtype=dtype
            )
            features = vision_features(
                vision, pixel_values, int(protocol["architecture"]["vision_feature_layer"])
            ).to("cpu", dtype=dtype)
            for (row, _), feature in zip(batch, features):
                result[(row["family_id"], row["variant"])] = feature.contiguous()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise SystemExit("router development selection is not authorized")
    for field in ("parameter_update_authorized", "sealed_test_access_authorized"):
        if protocol.get(field) is not False:
            raise SystemExit(f"{field} must remain false")
    for path, key in (
        (Path(__file__), "evaluator_sha256"),
        (ROOT / "test_evaluate_highres_evidence_router_v1.py", "test_sha256"),
        (ROOT / "evidence_acquisition.py", "router_module_sha256"),
        (ROOT / "train_highres_evidence_router_v1.py", "router_trainer_sha256"),
        (ROOT / "evaluate_highres_evidence_crop_bridge_v1.py", "crop_evaluator_sha256"),
        (ROOT / "evaluate_highres_evidence_responder_floor_v1.py", "floor_evaluator_sha256"),
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
    for key in ("bridge", "crop_bridge"):
        if sha256_file(Path(parents[key])) != parents[f"{key}_sha256"]:
            raise SystemExit(f"{key} mismatch")
    adapter_root = Path(parents["language_adapter"])
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(adapter_root / name) != expected:
            raise SystemExit(f"language adapter mismatch: {name}")
    for checkpoint in protocol["checkpoints"]:
        if sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise SystemExit(f"router checkpoint mismatch: {checkpoint['step']}")

    from peft import PeftModel
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

    dtype = torch.bfloat16
    rows = load_development_rows(Path(protocol["data"]["manifest"]))
    if len(rows) != int(protocol["data"]["expected_rows"]):
        raise RuntimeError("unexpected development row count")
    tokenizer = AutoTokenizer.from_pretrained(language_root, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_root, local_files_only=True, use_fast=False)
    vision = SiglipVisionModel.from_pretrained(vision_root, dtype=dtype, local_files_only=True).cuda()
    freeze(vision)
    caches = cache_vision(rows, protocol, processor, vision, dtype)
    language = AutoModelForCausalLM.from_pretrained(
        language_root, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(language, adapter_root, is_trainable=False, local_files_only=True)
    freeze(language)
    parent_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    parent_bridge.load_state_dict(load_file(parents["bridge"], device="cpu"), strict=True)
    freeze(parent_bridge)
    crop_bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    crop_bridge.load_state_dict(load_file(parents["crop_bridge"], device="cpu"), strict=True)
    freeze(crop_bridge)
    local_rows = [row for row in rows if row["expected_action"] == "POINT"]
    gains, _, _ = responder_gains(
        local_rows, protocol, tokenizer, language, parent_bridge, crop_bridge, caches
    )
    observed_gains = [gains.get((row["family_id"], row["variant"]), 0.0) for row in rows]
    input_ids, attention, labels = collate_true_answers(
        tokenizer, rows, int(protocol["teacher"]["max_text_tokens"])
    )
    with torch.inference_mode():
        text_embeddings = language.get_input_embeddings()(input_ids.cuda()).to("cpu", dtype=dtype)
    global_features = torch.stack([
        caches[0][(row["family_id"], row["variant"])] for row in rows
    ]).contiguous()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results, prediction_records = [], []
    selected_actions_by_step = {}
    batch_size = int(protocol["evaluation"]["router_batch_rows"])
    for checkpoint in protocol["checkpoints"]:
        router = EvidenceAcquisitionRouter(
            vision_dim=int(protocol["router"]["vision_dim"]),
            language_dim=int(protocol["router"]["language_dim"]),
            rank=int(protocol["router"]["rank"]),
        ).to("cuda", dtype=dtype)
        router.load_state_dict(load_file(checkpoint["path"], device="cpu"), strict=True)
        freeze(router)
        actions, predicted_gains, probabilities = [], [], []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                stop = min(len(rows), start + batch_size)
                logits, predicted_gain = router(
                    global_features[start:stop].to("cuda"),
                    text_embeddings[start:stop].to("cuda"),
                    attention[start:stop].to("cuda"),
                    labels[start:stop].to("cuda"),
                )
                action = choose_evidence_action(
                    logits, predicted_gain,
                    float(protocol["router"]["minimum_predicted_gain"]),
                    float(protocol["router"]["minimum_action_probability"]),
                )
                probability = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
                actions.extend(action.cpu().tolist())
                predicted_gains.extend(predicted_gain.cpu().tolist())
                probabilities.extend(probability.cpu().tolist())
        metrics = router_metrics(
            rows, actions, predicted_gains, observed_gains, int(protocol["router"]["patch_count"])
        )
        gate = protocol["gate"]
        gate_passed = (
            metrics["action_accuracy"] >= float(gate["minimum_action_accuracy"])
            and metrics["point_recall"] >= float(gate["minimum_point_recall"])
            and metrics["stop_accuracy"] >= float(gate["minimum_stop_accuracy"])
            and metrics["exact_patch_accuracy"] >= float(gate["minimum_exact_patch_accuracy"])
            and metrics["within_one_patch_accuracy"] >= float(gate["minimum_within_one_patch_accuracy"])
            and metrics["gain_mae"] <= float(gate["maximum_gain_mae"])
        )
        results.append({
            "step": checkpoint["step"],
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "metrics": metrics,
            "gate_passed": gate_passed,
        })
        selected_actions_by_step[checkpoint["step"]] = actions
        for row, action, predicted_gain, probability, observed_gain in zip(
            rows, actions, predicted_gains, probabilities, observed_gains
        ):
            prediction_records.append({
                "checkpoint_step": checkpoint["step"],
                "family_id": row["family_id"],
                "variant": row["variant"],
                "expected_action": row["expected_action"],
                "target_patch_index": row["target_patch_index_14x14"],
                "selected_action": action,
                "selected_probability": probability,
                "predicted_gain": predicted_gain,
                "observed_gain": observed_gain,
            })
        del router
    selected = select_checkpoint(results)
    policy = None
    if selected is not None:
        actions = selected_actions_by_step[selected["step"]]
        predicted_crop_cache = cache_predicted_crops(rows, actions, protocol, processor, vision, dtype)
        point_rows = [row for row, action in zip(rows, actions) if action < int(protocol["router"]["patch_count"])]
        stop_rows = [row for row, action in zip(rows, actions) if action == int(protocol["router"]["patch_count"])]
        policy_predictions, policy_invariance = {}, {}
        protocol["runtime_checkpoint_step"] = selected["step"]
        with torch.inference_mode():
            if point_rows:
                normal, normal_records = score_condition(
                    point_rows, "global_crop", False, protocol, language, parent_bridge,
                    crop_bridge, tokenizer, (caches[0], predicted_crop_cache), {},
                )
                reverse, reverse_records = score_condition(
                    point_rows, "global_crop", True, protocol, language, parent_bridge,
                    crop_bridge, tokenizer, (caches[0], predicted_crop_cache), {},
                )
                prediction_records.extend(normal_records + reverse_records)
                policy_invariance["POINT"] = normal == reverse
                policy_predictions.update({
                    (row["family_id"], row["variant"]): prediction
                    for row, prediction in zip(point_rows, normal)
                })
            if stop_rows:
                normal, normal_records = score_condition(
                    stop_rows, "global", False, protocol, language, parent_bridge,
                    crop_bridge, tokenizer, (caches[0], predicted_crop_cache), {},
                )
                reverse, reverse_records = score_condition(
                    stop_rows, "global", True, protocol, language, parent_bridge,
                    crop_bridge, tokenizer, (caches[0], predicted_crop_cache), {},
                )
                prediction_records.extend(normal_records + reverse_records)
                policy_invariance["STOP"] = normal == reverse
                policy_predictions.update({
                    (row["family_id"], row["variant"]): prediction
                    for row, prediction in zip(stop_rows, normal)
                })
        ordered_predictions = [policy_predictions[(row["family_id"], row["variant"])] for row in rows]
        local_predictions = [
            policy_predictions[(row["family_id"], row["variant"])] for row in local_rows
        ]
        global_rows = [row for row in rows if row["expected_action"] == "STOP"]
        global_predictions = [
            policy_predictions[(row["family_id"], row["variant"])] for row in global_rows
        ]
        policy = {
            "all_tasks": condition_metrics(rows, ordered_predictions),
            "local_evidence": condition_metrics(local_rows, local_predictions),
            "global_stop": condition_metrics(global_rows, global_predictions),
            "candidate_order_invariance": policy_invariance,
            "acquired_rows": len(point_rows),
            "stopped_rows": len(stop_rows),
        }
    with predictions_path.open("x") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_highres_evidence_router_development_selection",
        "completed_at": utc_now(),
        "decision": "select_router_checkpoint" if selected else "no_router_checkpoint_passed",
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "verified_parent_records": {
            "language": language_files,
            "vision_artifacts": parents["vision_artifacts"],
            "bridge_sha256": parents["bridge_sha256"],
            "crop_bridge_sha256": parents["crop_bridge_sha256"],
            "language_adapter_files": parents["language_adapter_files"],
        },
        "rows": len(rows),
        "checkpoint_results": results,
        "selected": selected,
        "selected_predicted_crop_policy": policy,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "bytes": predictions_path.stat().st_size,
        },
        "parameter_updates": 0,
        "sealed_test_rows_accessed": 0,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({
        "decision": receipt["decision"],
        "checkpoint_results": results,
        "selected": selected,
        "selected_predicted_crop_policy": policy,
        "elapsed_seconds": receipt["elapsed_seconds"],
        "peak_cuda_memory_bytes": receipt["peak_cuda_memory_bytes"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
