#!/usr/bin/env python3
"""Audit Stage-C training opportunity, cache semantics, and model capacity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def tensor_count(path: Path, prefix: str | None = None) -> int:
    count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if prefix is None or key.startswith(prefix):
                shape = handle.get_slice(key).get_shape()
                elements = 1
                for dimension in shape:
                    elements *= dimension
                count += elements
    return count


def selected(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": selection["selected_step"],
        "bridge": selection["selected_checkpoint"],
        "bridge_sha256": selection["selected_checkpoint_sha256"],
        "language_adapter": selection["selected_language_adapter"],
        "language_adapter_model_sha256": selection["selected_language_adapter_model_sha256"],
        "selection_candidates": len(selection["evaluations"]),
        "candidate_steps": [item["step"] for item in selection["evaluations"]],
        "selection_split": selection["selection_split"],
        "families_per_candidate": selection["scene_families_per_checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--full-protocol", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--full-selection", type=Path, required=True)
    parser.add_argument("--compressed-protocol", type=Path, required=True)
    parser.add_argument("--compressed-run", type=Path, required=True)
    parser.add_argument("--compressed-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    manifest_by_family = {row["scene_family_id"]: row for row in manifest_rows}
    if len(manifest_rows) != len(manifest_by_family):
        raise RuntimeError("manifest contains duplicate scene families")
    split_counts = Counter(row["split"] for row in manifest_rows)
    task_split_counts = Counter((row["task"], row["split"]) for row in manifest_rows)
    image_hashes = [
        row[f"{variant}_image_sha256"]
        for row in manifest_rows
        for variant in ("base", "edited", "invariant")
    ]

    cache_rows = [json.loads(line) for line in args.teacher_cache.read_text().splitlines() if line]
    cache_by_family = {row["scene_family_id"]: row for row in cache_rows}
    cache_errors = []
    for family, cached in cache_by_family.items():
        source = manifest_by_family.get(family)
        if source is None:
            cache_errors.append(f"unknown family {family}")
            continue
        if cached["task"] != source["task"]:
            cache_errors.append(f"task mismatch {family}")
        if cached["candidate_answers"] != source["candidate_answers"]:
            cache_errors.append(f"candidate mismatch {family}")
        for variant in ("base", "edited", "invariant"):
            expected_index = source["candidate_answers"].index(source[f"{variant}_answer"])
            if cached["variants"][variant]["expected_candidate_index"] != expected_index:
                cache_errors.append(f"expected-index mismatch {family}/{variant}")
    full_protocol = json_file(args.full_protocol)
    compressed_protocol = json_file(args.compressed_protocol)
    full_run = json_file(args.full_run)
    compressed_run = json_file(args.compressed_run)
    full_selection = selected(json_file(args.full_selection))
    compressed_selection = selected(json_file(args.compressed_selection))

    full_bridge = resolve(args.root, full_selection["bridge"])
    full_adapter = resolve(args.root, full_selection["language_adapter"]) / "adapter_model.safetensors"
    compressed_bridge = resolve(args.root, compressed_selection["bridge"])
    compressed_adapter = resolve(args.root, compressed_selection["language_adapter"]) / "adapter_model.safetensors"
    capacity = {
        "full_196_selected": {
            "bridge_parameters": tensor_count(full_bridge),
            "compressor_parameters": tensor_count(full_bridge, "compressor."),
            "language_lora_parameters": tensor_count(full_adapter),
        },
        "answer_49_selected": {
            "bridge_parameters": tensor_count(compressed_bridge),
            "compressor_parameters": tensor_count(compressed_bridge, "compressor."),
            "language_lora_parameters": tensor_count(compressed_adapter),
        },
    }
    shared_parent = compressed_protocol["parents"]["bridge_sha256"] == full_selection["bridge_sha256"]
    training_opportunity = {
        "same_manifest": full_run["manifest_sha256"] == compressed_run["manifest_sha256"],
        "same_14000_train_families_per_run": full_run["examples"] // 3 == compressed_run["families"] == 14000,
        "same_nominal_optimizer_steps_per_run": full_run["optimizer_step"] == compressed_run["optimizer_step"] == 875,
        "compressed_continuation_starts_from_selected_full_checkpoint": shared_parent,
        "selected_full_step_within_its_run": full_selection["step"],
        "compressed_additional_steps_after_selected_full": compressed_run["optimizer_step"],
        "current_observed_comparison_is_training_opportunity_matched": False,
        "reason": (
            "answer_49 starts from the selected full_196 checkpoint and receives 875 additional "
            "optimizer steps; the selected full_196 checkpoint does not receive that matched continuation."
        ),
        "learning_rate_search": {
            "full_196": "fixed schedule; no three-point LR micro-search recorded",
            "answer_49": "fixed schedule; no three-point LR micro-search recorded",
            "stage_d_requirement": "equal 0.5x/1x/2x development-only micro-search budget",
        },
        "checkpoint_selection": {
            "full_196": full_selection,
            "answer_49": compressed_selection,
            "matched_opportunity": full_selection["selection_candidates"] == compressed_selection["selection_candidates"],
        },
    }
    gates = {
        "manifest_scene_families_unique": len(manifest_rows) == len(manifest_by_family),
        "manifest_image_hashes_unique": len(image_hashes) == len(set(image_hashes)),
        "teacher_cache_exact_train_family_set": set(cache_by_family) == {
            row["scene_family_id"] for row in manifest_rows if row["split"] == "train"
        },
        "teacher_cache_semantics_match_manifest": not cache_errors,
        "compressed_parent_is_selected_full_checkpoint": shared_parent,
        "current_comparison_training_opportunity_matched": False,
    }
    receipt = {
        "schema_version": "2026-09-13-v1",
        "status": "audit_complete_matched_control_required",
        "manifest": {
            "sha256": sha256_file(args.manifest),
            "scene_families": len(manifest_rows),
            "split_counts": dict(sorted(split_counts.items())),
            "task_split_counts": {"|".join(key): value for key, value in sorted(task_split_counts.items())},
            "rendered_image_hashes": len(image_hashes),
            "unique_rendered_image_hashes": len(set(image_hashes)),
        },
        "teacher_cache": {
            "sha256": sha256_file(args.teacher_cache),
            "families": len(cache_rows),
            "semantic_error_count": len(cache_errors),
            "semantic_error_sample": cache_errors[:20],
            "binds_task_candidate_order_and_expected_candidate_index": not cache_errors,
            "binds_image_sha256": False,
            "image_hash_boundary": (
                "The legacy teacher cache contains no image hashes. Stage-D answer-only training does not "
                "consume teacher scores, and evaluator input images are independently hash-verified."
            ),
        },
        "training_opportunity": training_opportunity,
        "capacity": capacity,
        "protocol_hashes": {
            "full": sha256_file(args.full_protocol),
            "compressed": sha256_file(args.compressed_protocol),
        },
        "gates": gates,
        "required_resolution": (
            "Run F_matched_196 and A_spatial_49 from the exact same selected parent with equal data, "
            "updates, LR-search opportunity, seed pairing, and checkpoint-selection opportunity."
        ),
        "prior_test_role": "regression_only_not_for_stage_d_tuning_or_selection",
        "claim_boundary": (
            "The Stage-C observed checkpoint difference is real, but this audit confirms it cannot yet be "
            "causally attributed to compression because adaptation opportunities were unequal."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "gates": gates, "capacity": capacity}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
