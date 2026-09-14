#!/usr/bin/env python3
"""Evaluate address/value interchange on held-out binding development splits."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from counterfactual_losses import masked_sequence_logprob, paired_cluster_bootstrap
from evaluate_stage_a_visual_floor import directory_records, image_integrity_audit, select_binding_scene_pairs
from modeling import bridge_from_architecture, freeze, vision_features
from train_counterfactual_address_value_router import (
    RoutingBatchBuilder,
    VARIANTS,
    counterfactual_value_features,
    grid_index,
)
from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic
from train_stage_c_counterfactual import FamilyDataset


def bootstrap_percent(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    selected = [record for record in records if record.get("affected", False)]
    interval = paired_cluster_bootstrap(
        [100.0 * float(record[key]) for record in selected],
        [record["scene_pair_id"] for record in selected],
        resamples=10_000,
        seed=20260914,
    )
    return {
        "estimate_percent": interval.estimate,
        "lower_95_ci_percent": interval.lower,
        "upper_95_ci_percent": interval.upper,
        "scene_pair_clusters": interval.clusters,
        "question_families": interval.samples,
        "bootstrap_resamples": interval.resamples,
    }


def candidate_attention_gap(attention: torch.Tensor) -> torch.Tensor:
    if attention.ndim != 4 or attention.shape[2] != 2:
        raise ValueError("attention must be [families, variants, 2, visual_tokens]")
    return (attention[:, :, 0] - attention[:, :, 1]).abs().amax(dim=(1, 2))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    affected = [record for record in records if record["affected"]]
    if not affected:
        raise ValueError("mechanism evaluation requires affected question families")

    def mean(key: str, selected: list[dict[str, Any]] = affected) -> float:
        return sum(float(record[key]) for record in selected) / len(selected)

    return {
        "question_families": len(records),
        "affected_question_families": len(affected),
        "scene_pair_clusters": len({record["scene_pair_id"] for record in records}),
        "candidate_attention_max_gap": max(record["candidate_attention_max_gap"] for record in records),
        "affected_address_variant_accuracy_percent": 100.0 * mean("address_variant_accuracy"),
        "affected_address_all_variants_percent": 100.0 * mean("address_all_variants"),
        "affected_address_target_probability_percent": 100.0 * mean("address_target_probability"),
        "affected_address_base_edit_js": mean("address_base_edit_js"),
        "affected_normal_base_edit_joint_percent": 100.0 * mean("normal_base_edit_joint"),
        "affected_swapped_base_edit_joint_percent": 100.0 * mean("swapped_base_edit_joint"),
        "affected_invariant_normal_correct_percent": 100.0 * mean("invariant_normal_correct"),
        "affected_invariant_swapped_correct_percent": 100.0 * mean("invariant_swapped_correct"),
        "affected_invariant_prediction_stable_percent": 100.0 * mean("invariant_prediction_stable"),
        "affected_causal_six_way_percent": 100.0 * mean("causal_six_way"),
        "cluster_bootstrap_95_ci": {
            key: bootstrap_percent(records, key)
            for key in (
                "address_all_variants",
                "normal_base_edit_joint",
                "swapped_base_edit_joint",
                "causal_six_way",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=("query_ce", "cavr"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--split", choices=("mechanism_dev", "selection_dev"), required=True)
    parser.add_argument("--scene-pairs", type=int)
    parser.add_argument("--batch-families", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.step <= 0 or args.batch_families <= 0:
        raise SystemExit("step and batch-families must be positive")

    root = Path(__file__).resolve().parent
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "authorized_cavr_mechanism_development_v1":
        raise SystemExit("CAVR mechanism protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    for filename, key in (
        ("evaluate_cavr_mechanism.py", "mechanism_evaluator_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("train_counterfactual_address_value_router.py", "trainer_sha256"),
        ("counterfactual_losses.py", "counterfactual_losses_sha256"),
    ):
        if sha256_file(root / filename) != protocol["source_code"][key]:
            raise SystemExit(f"mechanism source hash mismatch: {filename}")

    training_path = Path(protocol["training_protocol"]["path"])
    if sha256_file(training_path) != protocol["training_protocol"]["sha256"]:
        raise SystemExit("mechanism training protocol hash mismatch")
    training = json.loads(training_path.read_text())
    architecture = training["arms"][args.arm]["architecture"]
    arm = protocol["arms"][args.arm]
    run_root = Path(arm["run"])
    receipt = json.loads((run_root / "run.json").read_text())
    if (
        receipt.get("status") != "complete"
        or receipt.get("arm") != args.arm
        or receipt.get("protocol_sha256") != protocol["training_protocol"]["sha256"]
        or receipt.get("parent_before") != receipt.get("parent_after")
    ):
        raise SystemExit("mechanism training receipt mismatch")
    checkpoint = run_root / f"step-{args.step:06d}"
    if not checkpoint.is_dir():
        raise SystemExit("declared checkpoint does not exist")

    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("mechanism manifest hash mismatch")
    all_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    rows = select_binding_scene_pairs(
        [row for row in all_rows if row["split"] == args.split], args.scene_pairs
    )
    image_audit = image_integrity_audit(rows)
    if image_audit["all_match_manifest"] is not True:
        raise SystemExit("mechanism image integrity audit failed")

    started = time.monotonic()
    parents = protocol["parents"]
    tokenizer = AutoTokenizer.from_pretrained(parents["language"], local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(
        parents["vision"], local_files_only=True, use_fast=False
    )
    language = AutoModelForCausalLM.from_pretrained(
        parents["language"], dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    from peft import PeftModel

    language = PeftModel.from_pretrained(
        language,
        checkpoint / "language_adapter",
        is_trainable=False,
        local_files_only=True,
    )
    freeze(language)
    vision = SiglipVisionModel.from_pretrained(
        parents["vision"], dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    bridge = bridge_from_architecture(architecture).to(device="cuda", dtype=torch.bfloat16)
    bridge.load_state_dict(load_file(checkpoint / "bridge.safetensors", device="cpu"), strict=True)
    freeze(bridge)

    row_by_id = {row["scene_family_id"]: row for row in rows}
    loader = DataLoader(
        FamilyDataset(rows),
        batch_size=args.batch_families,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=RoutingBatchBuilder(
            tokenizer=tokenizer,
            processor=processor,
            max_tokens=args.max_tokens,
            teacher=None,
            deduplicate_images=True,
            output_grid=7,
        ),
    )
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            family_ids = batch["scene_family_ids"]
            families = len(family_ids)
            pixels = batch["pixel_values"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            targets = batch["target_indices"].to("cuda", non_blocking=True)
            unique_features = vision_features(
                vision, pixels, int(architecture.get("vision_feature_layer", -1))
            )
            indices = batch["image_feature_indices"].to("cuda", non_blocking=True)
            features = unique_features[indices.reshape(-1)].repeat_interleave(2, dim=0)
            text = language.get_input_embeddings()(input_ids)
            inputs, mask, injected_targets, readout_attention = bridge.inject_with_readout(
                text, attention_mask, labels, features
            )
            assert injected_targets is not None and readout_attention is not None
            logits = language(inputs_embeds=inputs, attention_mask=mask).logits[:, :-1]
            shifted = injected_targets[:, 1:]
            token_mask = shifted != -100
            normal_scores = masked_sequence_logprob(
                logits, shifted.masked_fill(~token_mask, 0), token_mask
            ).reshape(families, len(VARIANTS), 2)
            swapped_features = counterfactual_value_features(features, families)
            swapped_inputs, swapped_mask, swapped_targets, _ = bridge.inject_with_readout(
                text,
                attention_mask,
                labels,
                features,
                query_readout_value_features=swapped_features,
            )
            assert swapped_targets is not None
            swapped_logits = language(
                inputs_embeds=swapped_inputs, attention_mask=swapped_mask
            ).logits[:, :-1]
            shifted_swapped = swapped_targets[:, 1:]
            swapped_token_mask = shifted_swapped != -100
            swapped_scores = masked_sequence_logprob(
                swapped_logits,
                shifted_swapped.masked_fill(~swapped_token_mask, 0),
                swapped_token_mask,
            ).reshape(families, len(VARIANTS), 2)

            attention = readout_attention.reshape(families, len(VARIANTS), 2, -1).float()
            candidate_gap = candidate_attention_gap(attention)
            mean_attention = attention.mean(dim=2)
            normal_predictions = normal_scores.argmax(dim=-1)
            swapped_predictions = swapped_scores.argmax(dim=-1)
            expected_swapped = torch.stack((targets[:, 1], targets[:, 0], targets[:, 2]), dim=1)
            for index, family_id in enumerate(family_ids):
                row = row_by_id[family_id]
                target_object = next(
                    item for item in row["scene_state"]["objects"] if item["id"] == "target"
                )
                address_target = grid_index(target_object["y"], 7) * 7 + grid_index(target_object["x"], 7)
                address_correct = mean_attention[index].argmax(dim=-1).eq(address_target)
                address_probability = mean_attention[index, :, address_target]
                base = mean_attention[index, 0].clamp_min(1e-8)
                edited = mean_attention[index, 1].clamp_min(1e-8)
                midpoint = 0.5 * (base + edited)
                js = 0.5 * (
                    (base * (base.log() - midpoint.log())).sum()
                    + (edited * (edited.log() - midpoint.log())).sum()
                )
                normal_correct = normal_predictions[index].eq(targets[index])
                swapped_correct = swapped_predictions[index].eq(expected_swapped[index])
                records.append({
                    "scene_family_id": family_id,
                    "scene_pair_id": row["scene_pair_id"],
                    "question_role": row["question_role"],
                    "affected": row["question_role"].startswith("affected_"),
                    "address_target": address_target,
                    "address_predictions": mean_attention[index].argmax(dim=-1).cpu().tolist(),
                    "address_variant_accuracy": float(address_correct.float().mean()),
                    "address_all_variants": bool(address_correct.all()),
                    "address_target_probability": float(address_probability.mean()),
                    "address_base_edit_js": float(js),
                    "candidate_attention_max_gap": float(candidate_gap[index]),
                    "normal_predictions": normal_predictions[index].cpu().tolist(),
                    "normal_targets": targets[index].cpu().tolist(),
                    "swapped_predictions": swapped_predictions[index].cpu().tolist(),
                    "swapped_targets": expected_swapped[index].cpu().tolist(),
                    "normal_base_edit_joint": bool(normal_correct[:2].all()),
                    "swapped_base_edit_joint": bool(swapped_correct[:2].all()),
                    "invariant_normal_correct": bool(normal_correct[2]),
                    "invariant_swapped_correct": bool(swapped_correct[2]),
                    "invariant_prediction_stable": bool(
                        normal_predictions[index, 2] == swapped_predictions[index, 2]
                    ),
                    "causal_six_way": bool(normal_correct.all() and swapped_correct.all()),
                })
            if batch_index % 10 == 0:
                print(
                    f"MECHANISM_PROGRESS arm={args.arm} split={args.split} families={len(records)} "+
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )

    if not math.isfinite(max(record["candidate_attention_max_gap"] for record in records)):
        raise RuntimeError("non-finite candidate attention audit")
    result = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_development_only",
        "arm": args.arm,
        "split": args.split,
        "step": args.step,
        "protocol_sha256": sha256_file(protocol_path),
        "training_protocol_sha256": protocol["training_protocol"]["sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_bridge_sha256": sha256_file(checkpoint / "bridge.safetensors"),
        "checkpoint_adapter_records": directory_records(checkpoint / "language_adapter"),
        "image_integrity_audit": image_audit,
        "metrics": summarize_records(records),
        "records": records,
        "wall_seconds": time.monotonic() - started,
        "completed_at": utc_now(),
        "final_test_used": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "arm", "split", "step", "metrics", "wall_seconds", "final_test_used")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
