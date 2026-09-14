#!/usr/bin/env python3
"""One-shot confirmation of one frozen high-resolution evidence router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from safetensors.torch import load_file

from evaluate_highres_evidence_crop_bridge_v1 import score_condition
from evaluate_highres_evidence_responder_floor_v1 import condition_metrics
from evaluate_highres_evidence_router_v1 import cache_predicted_crops, router_metrics
from evidence_acquisition import EvidenceAcquisitionRouter, choose_evidence_action
from modeling import bridge_from_architecture, freeze
from train_highres_evidence_router_v1 import cache_vision, collate_true_answers, responder_gains
from train_stage_a_full_token import release_records, sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_highres_evidence_router_confirmation_once_v1"


def load_confirmation_rows(manifest: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == split]
    selected.sort(key=lambda row: (row["family_id"], row["variant"]))
    if any(row["split"] != split for row in selected):
        raise RuntimeError("non-confirmation row reached confirmation evaluator")
    return selected


def confirmation_gate(metrics: dict[str, Any], policy: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        metrics["action_accuracy"] >= float(gate["minimum_action_accuracy"])
        and metrics["point_recall"] >= float(gate["minimum_point_recall"])
        and metrics["stop_accuracy"] >= float(gate["minimum_stop_accuracy"])
        and metrics["exact_patch_accuracy"] >= float(gate["minimum_exact_patch_accuracy"])
        and metrics["within_one_patch_accuracy"] >= float(gate["minimum_within_one_patch_accuracy"])
        and metrics["gain_mae"] <= float(gate["maximum_gain_mae"])
        and policy["local_evidence"]["row_accuracy"] >= float(gate["minimum_local_policy_row_accuracy"])
        and policy["local_evidence"]["family_joint_accuracy"] >= float(gate["minimum_local_policy_family_accuracy"])
        and all(policy["candidate_order_invariance"].values())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise SystemExit("one-shot router confirmation is not authorized")
    for field in ("parameter_update_authorized", "threshold_update_authorized", "checkpoint_selection_authorized"):
        if protocol.get(field) is not False:
            raise SystemExit(f"{field} must remain false")
    for path, key in (
        (Path(__file__), "evaluator_sha256"),
        (ROOT / "test_evaluate_highres_evidence_router_confirmation_v1.py", "test_sha256"),
        (ROOT / "evaluate_highres_evidence_router_v1.py", "development_evaluator_sha256"),
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
    checkpoint = protocol["checkpoint"]
    if sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise SystemExit("frozen router checkpoint mismatch")

    from peft import PeftModel
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

    data = protocol["data"]
    rows = load_confirmation_rows(Path(data["manifest"]), data["allowed_split"])
    if len(rows) != int(data["expected_rows"]):
        raise RuntimeError("unexpected confirmation row count")
    if len({row["family_id"] for row in rows}) != int(data["expected_families"]):
        raise RuntimeError("unexpected confirmation family count")
    counts = {action: sum(row["expected_action"] == action for row in rows) for action in ("POINT", "STOP")}
    if counts != data["expected_action_counts"]:
        raise RuntimeError(f"unexpected confirmation action counts: {counts}")

    dtype = torch.bfloat16
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

    point_rows = [row for row in rows if row["expected_action"] == "POINT"]
    gains, _, _ = responder_gains(
        point_rows, protocol, tokenizer, language, parent_bridge, crop_bridge, caches
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
    router = EvidenceAcquisitionRouter(
        vision_dim=int(protocol["router"]["vision_dim"]),
        language_dim=int(protocol["router"]["language_dim"]),
        rank=int(protocol["router"]["rank"]),
    ).to("cuda", dtype=dtype)
    router.load_state_dict(load_file(checkpoint["path"], device="cpu"), strict=True)
    freeze(router)
    actions, predicted_gains, probabilities = [], [], []
    batch_size = int(protocol["evaluation"]["router_batch_rows"])
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
                logits,
                predicted_gain,
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

    predicted_crop_cache = cache_predicted_crops(rows, actions, protocol, processor, vision, dtype)
    acquired_rows = [row for row, action in zip(rows, actions) if action < int(protocol["router"]["patch_count"])]
    stopped_rows = [row for row, action in zip(rows, actions) if action == int(protocol["router"]["patch_count"])]
    prediction_records = [
        {
            "record_type": "router",
            "family_id": row["family_id"],
            "variant": row["variant"],
            "expected_action": row["expected_action"],
            "target_patch_index": row["target_patch_index_14x14"],
            "selected_action": action,
            "selected_probability": probability,
            "predicted_gain": predicted_gain,
            "observed_gain": observed_gain,
        }
        for row, action, probability, predicted_gain, observed_gain in zip(
            rows, actions, probabilities, predicted_gains, observed_gains
        )
    ]
    policy_predictions, policy_invariance = {}, {}
    protocol["runtime_checkpoint_step"] = checkpoint["step"]
    with torch.inference_mode():
        if acquired_rows:
            normal, normal_records = score_condition(
                acquired_rows, "global_crop", False, protocol, language, parent_bridge,
                crop_bridge, tokenizer, (caches[0], predicted_crop_cache), {},
            )
            reverse, reverse_records = score_condition(
                acquired_rows, "global_crop", True, protocol, language, parent_bridge,
                crop_bridge, tokenizer, (caches[0], predicted_crop_cache), {},
            )
            prediction_records.extend(normal_records + reverse_records)
            policy_invariance["POINT_ACTION"] = normal == reverse
            policy_predictions.update({
                (row["family_id"], row["variant"]): prediction
                for row, prediction in zip(acquired_rows, normal)
            })
        if stopped_rows:
            normal, normal_records = score_condition(
                stopped_rows, "global", False, protocol, language, parent_bridge,
                crop_bridge, tokenizer, (caches[0], predicted_crop_cache), {},
            )
            reverse, reverse_records = score_condition(
                stopped_rows, "global", True, protocol, language, parent_bridge,
                crop_bridge, tokenizer, (caches[0], predicted_crop_cache), {},
            )
            prediction_records.extend(normal_records + reverse_records)
            policy_invariance["STOP_ACTION"] = normal == reverse
            policy_predictions.update({
                (row["family_id"], row["variant"]): prediction
                for row, prediction in zip(stopped_rows, normal)
            })
    ordered_predictions = [policy_predictions[(row["family_id"], row["variant"])] for row in rows]
    local_predictions = [policy_predictions[(row["family_id"], row["variant"])] for row in point_rows]
    stop_target_rows = [row for row in rows if row["expected_action"] == "STOP"]
    stop_predictions = [policy_predictions[(row["family_id"], row["variant"])] for row in stop_target_rows]
    policy = {
        "all_tasks": condition_metrics(rows, ordered_predictions),
        "local_evidence": condition_metrics(point_rows, local_predictions),
        "global_stop": condition_metrics(stop_target_rows, stop_predictions),
        "candidate_order_invariance": policy_invariance,
        "acquired_rows": len(acquired_rows),
        "stopped_rows": len(stopped_rows),
    }
    passed = confirmation_gate(metrics, policy, protocol["gate"])

    with predictions_path.open("x") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_highres_evidence_router_confirmation",
        "completed_at": utc_now(),
        "decision": "confirm_router_checkpoint" if passed else "reject_router_checkpoint",
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "checkpoint": checkpoint,
        "verified_parent_records": {
            "language": language_files,
            "vision_artifacts": parents["vision_artifacts"],
            "bridge_sha256": parents["bridge_sha256"],
            "crop_bridge_sha256": parents["crop_bridge_sha256"],
            "language_adapter_files": parents["language_adapter_files"],
        },
        "rows": len(rows),
        "families": len({row["family_id"] for row in rows}),
        "action_counts": counts,
        "router_metrics": metrics,
        "predicted_crop_policy": policy,
        "gate_passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "bytes": predictions_path.stat().st_size,
        },
        "parameter_updates": 0,
        "threshold_updates": 0,
        "checkpoint_selections": 0,
        "confirmation_runs": 1,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({
        "decision": receipt["decision"],
        "checkpoint": checkpoint,
        "router_metrics": metrics,
        "predicted_crop_policy": policy,
        "elapsed_seconds": receipt["elapsed_seconds"],
        "peak_cuda_memory_bytes": receipt["peak_cuda_memory_bytes"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
