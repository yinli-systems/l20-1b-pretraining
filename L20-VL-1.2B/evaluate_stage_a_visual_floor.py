#!/usr/bin/env python3
"""Evaluate a frozen visual bridge on controlled development pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel
from PIL import Image

from counterfactual_losses import paired_cluster_bootstrap
from modeling import bridge_from_architecture, freeze, vision_features
from scoring_contract import (
    RANDOM_CONTROL_POLICIES,
    VARIANTS,
    metric_contract,
    random_control_assignment,
)
from train_stage_a_full_token import encode_prompt_response, release_records, vision_records

BINDING_DEVELOPMENT_STATUSES = {
    "authorized_binding_answer_only_development_v1",
    "authorized_binding_answer_only_development_v2",
    "authorized_query_ce_gpu_time_matched_evaluation_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_records(path: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def collate_candidates(tokenizer, prompts: list[str], answers: list[str], max_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [encode_prompt_response(tokenizer, prompt, answer, max_tokens) for prompt, answer in zip(prompts, answers)]
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    for index, (ids, targets) in enumerate(encoded):
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids)
        labels[index, :length] = torch.tensor(targets)
        attention[index, :length] = 1
    return input_ids, attention, labels


def score_variant(
    rows: list[dict[str, Any]],
    image_paths: list[str] | None,
    language,
    vision,
    bridge,
    tokenizer,
    processor,
    batch_families: int,
    max_tokens: int,
    vision_feature_layer: int,
    candidate_order: str,
    scoring_dtype: torch.dtype,
) -> tuple[list[str], list[dict[str, Any]]]:
    predictions: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_families):
            batch_rows = rows[start : start + batch_families]
            prompts = []
            answers = []
            unique_paths = []
            path_indices = []
            path_to_index = {}
            for local_index, row in enumerate(batch_rows):
                manifest_answers = row["candidate_answers"]
                if len(manifest_answers) != 2 or len(set(manifest_answers)) != 2:
                    raise RuntimeError("evaluator requires exactly two distinct candidate answers")
                ordered_answers = (
                    manifest_answers if candidate_order == "manifest" else list(reversed(manifest_answers))
                )
                for answer in ordered_answers:
                    prompts.append(row["question"])
                    answers.append(answer)
                    if image_paths is not None:
                        path = image_paths[start + local_index]
                        if path not in path_to_index:
                            path_to_index[path] = len(unique_paths)
                            unique_paths.append(path)
                        path_indices.append(path_to_index[path])
            input_ids, attention, labels = collate_candidates(tokenizer, prompts, answers, max_tokens)
            input_ids = input_ids.cuda(non_blocking=True)
            attention = attention.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            text_embeddings = language.get_input_embeddings()(input_ids)
            if image_paths is None:
                inputs, mask, targets = text_embeddings, attention, labels
            else:
                images = []
                for path in unique_paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB").copy())
                pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
                    "cuda", dtype=scoring_dtype
                )
                unique_visual_features = vision_features(vision, pixels, vision_feature_layer)
                visual_features = unique_visual_features[
                    torch.tensor(path_indices, dtype=torch.long, device=unique_visual_features.device)
                ]
                inputs, mask, targets = bridge.inject(
                    text_embeddings, attention, labels, visual_features
                )
            logits = language(inputs_embeds=inputs, attention_mask=mask).logits[:, :-1].float()
            shifted = targets[:, 1:]
            target_mask = shifted != -100
            if not target_mask.any(dim=1).all():
                raise RuntimeError("candidate scoring encountered an empty answer span")
            safe_targets = shifted.masked_fill(~target_mask, 0)
            token_logprob = torch.log_softmax(logits, dim=-1).gather(
                -1, safe_targets.unsqueeze(-1)
            ).squeeze(-1)
            sequence_logprob = (token_logprob * target_mask).sum(dim=-1)
            for index, row in enumerate(batch_rows):
                offset = 2 * index
                candidate_index = int(torch.argmax(sequence_logprob[offset : offset + 2]))
                ordered_answers = (
                    row["candidate_answers"]
                    if candidate_order == "manifest"
                    else list(reversed(row["candidate_answers"]))
                )
                predictions.append(ordered_answers[candidate_index])
                score_values = sequence_logprob[offset : offset + 2].detach().cpu().tolist()
                diagnostics.append({
                    "candidate_sequence_logprobs": {
                        answer: float(score)
                        for answer, score in zip(ordered_answers, score_values)
                    },
                    "absolute_margin": abs(float(score_values[0] - score_values[1])),
                })
    return predictions, diagnostics


def candidate_span_audit(tokenizer, rows: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    lengths = []
    for row in rows:
        for answer in row["candidate_answers"]:
            _, labels = encode_prompt_response(tokenizer, row["question"], answer, max_tokens)
            lengths.append(sum(label != -100 for label in labels))
    return {
        "candidate_sequences": len(lengths),
        "nonempty_answer_spans": sum(length > 0 for length in lengths),
        "minimum_answer_tokens": min(lengths),
        "maximum_answer_tokens": max(lengths),
        "all_nonempty": all(length > 0 for length in lengths),
    }


def image_integrity_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    mismatches = []
    verified = {}
    for row in rows:
        for variant in VARIANTS:
            path = Path(row[f"{variant}_image_path"])
            expected = row[f"{variant}_image_sha256"]
            key = (str(path), expected)
            if key not in verified:
                verified[key] = sha256_file(path)
                checked += 1
            actual = verified[key]
            if actual != expected:
                mismatches.append({
                    "scene_family_id": row["scene_family_id"],
                    "variant": variant,
                    "path": str(path),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                })
    return {
        "checked_images": checked,
        "mismatch_count": len(mismatches),
        "all_match_manifest": not mismatches,
        "mismatches": mismatches,
    }


def interval(differences: list[float], families: list[str]) -> dict[str, float | int]:
    result = paired_cluster_bootstrap(differences, families, resamples=10_000, seed=20260913)
    return {
        "estimate_pp": 100 * result.estimate,
        "lower_95_ci_pp": 100 * result.lower,
        "upper_95_ci_pp": 100 * result.upper,
        "clusters": result.clusters,
        "resamples": result.resamples,
    }


def binding_selective_metrics(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Separate binding questions from invariant controls.

    Each scene pair has two questions whose answers must change after the swap
    and four controls whose answers must remain unchanged. Pooling all six lets
    the controls dominate, so affected-only joint accuracy is the primary
    metric and all-six correctness is the strict selective-intervention metric.
    """
    if not prediction_rows or not all(row.get("question_role") for row in prediction_rows):
        return None
    roles = {row["question_role"] for row in prediction_rows}
    if not all(role.startswith(("affected_", "invariant_")) for role in roles):
        return None
    conditions = sorted(
        set.intersection(*(set(row["predictions"]) for row in prediction_rows))
    )
    if not conditions:
        raise RuntimeError("binding metrics require a common evaluation condition")
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        pair_id = row.get("scene_pair_id")
        if not pair_id:
            raise RuntimeError("binding row is missing scene_pair_id")
        pairs[pair_id].append(row)
    for pair_id, pair_rows in pairs.items():
        affected = [row for row in pair_rows if row["question_role"].startswith("affected_")]
        invariant = [row for row in pair_rows if row["question_role"].startswith("invariant_")]
        if len(pair_rows) != 6 or len(affected) != 2 or len(invariant) != 4:
            raise RuntimeError(
                f"binding scene pair {pair_id} must contain 2 affected and 4 invariant rows"
            )

    def row_joint(row: dict[str, Any], condition: str) -> float:
        expected = row["expected"]
        predicted = row["predictions"][condition]
        return float(
            predicted["base"] == expected["base"]
            and predicted["edited"] == expected["edited"]
        )

    affected_rows = [
        row for row in prediction_rows if row["question_role"].startswith("affected_")
    ]
    invariant_rows = [
        row for row in prediction_rows if row["question_role"].startswith("invariant_")
    ]
    per_condition: dict[str, dict[str, float]] = {}
    for condition in conditions:
        affected_values = [row_joint(row, condition) for row in affected_rows]
        invariant_values = [row_joint(row, condition) for row in invariant_rows]
        pair_affected = []
        pair_invariant = []
        pair_all = []
        for pair_rows in pairs.values():
            affected = [
                row_joint(row, condition)
                for row in pair_rows
                if row["question_role"].startswith("affected_")
            ]
            invariant = [
                row_joint(row, condition)
                for row in pair_rows
                if row["question_role"].startswith("invariant_")
            ]
            pair_affected.append(float(all(affected)))
            pair_invariant.append(float(all(invariant)))
            pair_all.append(float(all(affected + invariant)))
        per_condition[condition] = {
            "affected_question_joint_accuracy_percent": 100 * sum(affected_values) / len(affected_values),
            "invariant_question_joint_accuracy_percent": 100 * sum(invariant_values) / len(invariant_values),
            "scene_pair_all_affected_correct_percent": 100 * sum(pair_affected) / len(pair_affected),
            "scene_pair_all_invariant_correct_percent": 100 * sum(pair_invariant) / len(pair_invariant),
            "scene_pair_selective_all_six_correct_percent": 100 * sum(pair_all) / len(pair_all),
        }
    affected_clusters = [row["statistical_cluster_id"] for row in affected_rows]
    affected_by_condition = {
        condition: [row_joint(row, condition) for row in affected_rows]
        for condition in conditions
    }
    true_vs_no = None
    true_vs_random = None
    if "true_image" in conditions and "no_image" in conditions:
        true_vs_no = interval(
            [
                true - control
                for true, control in zip(
                    affected_by_condition["true_image"], affected_by_condition["no_image"]
                )
            ],
            affected_clusters,
        )
    if "true_image" in conditions and "random_image" in conditions:
        true_vs_random = interval(
            [
                true - control
                for true, control in zip(
                    affected_by_condition["true_image"], affected_by_condition["random_image"]
                )
            ],
            affected_clusters,
        )
    return {
        "primary_metric": "affected_question_joint_accuracy_percent",
        "scene_pairs": len(pairs),
        "affected_questions": len(affected_rows),
        "invariant_control_questions": len(invariant_rows),
        "per_condition": per_condition,
        "affected_true_minus_no_image": true_vs_no,
        "affected_true_minus_random_image": true_vs_random,
        "claim_boundary": (
            "Affected-only accuracy is primary. The all-six metric additionally requires every "
            "affected and invariant-control question in a synthetic scene pair to be correct."
        ),
    }


def select_evaluation_rows(
    rows: list[dict[str, Any]],
    families_per_task: int | None,
    *,
    balance_base_answer: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("statistical_cluster_id", row["scene_family_id"]),
            row.get("question_index", 0),
            row["scene_family_id"],
        ),
    )
    if families_per_task is None:
        return ordered
    if families_per_task < 1:
        raise ValueError("families-per-task must be positive")
    if balance_base_answer and families_per_task % 2:
        raise ValueError("answer-balanced selection requires an even families-per-task")
    selected_rows = []
    for task in sorted({row["task"] for row in ordered}):
        task_rows = [row for row in ordered if row["task"] == task]
        if balance_base_answer:
            per_answer = families_per_task // 2
            for answer in ("no", "yes"):
                stratum = [row for row in task_rows if row["base_answer"] == answer]
                if len(stratum) < per_answer:
                    raise RuntimeError(f"insufficient {task}/{answer} rows for selection")
                selected_rows.extend(stratum[:per_answer])
        else:
            if len(task_rows) < families_per_task:
                raise RuntimeError(f"insufficient {task} rows for checkpoint selection")
            selected_rows.extend(task_rows[:families_per_task])
    return sorted(selected_rows, key=lambda row: row["scene_family_id"])


def select_binding_scene_pairs(
    rows: list[dict[str, Any]], scene_pairs: int | None
) -> list[dict[str, Any]]:
    if scene_pairs is None:
        return rows
    if scene_pairs < 1:
        raise ValueError("scene-pairs must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_id = row.get("scene_pair_id")
        if not pair_id:
            raise ValueError("binding row is missing scene_pair_id")
        grouped[pair_id].append(row)
    if len(grouped) < scene_pairs:
        raise ValueError("insufficient binding scene pairs for selection")
    selected_ids = sorted(grouped)[:scene_pairs]
    if any(len(grouped[pair_id]) != 6 for pair_id in selected_ids):
        raise ValueError("selected binding scene pair is incomplete")
    return sorted(
        [row for pair_id in selected_ids for row in grouped[pair_id]],
        key=lambda row: (row["scene_pair_id"], row.get("question_index", 0)),
    )


def main() -> None:
    process_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--language-adapter", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--batch-families", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--families-per-task", type=int)
    parser.add_argument("--scene-pairs", type=int)
    parser.add_argument(
        "--random-control",
        choices=RANDOM_CONTROL_POLICIES,
        default="global_shift",
    )
    parser.add_argument("--candidate-order", choices=("manifest", "reversed"), default="manifest")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("true_image", "random_image", "no_image"),
        default=("true_image", "random_image", "no_image"),
    )
    parser.add_argument("--verify-image-hashes", action="store_true")
    parser.add_argument("--scoring-precision", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    scoring_dtype = torch.bfloat16 if args.scoring_precision == "bfloat16" else torch.float32
    torch.set_float32_matmul_precision("high" if scoring_dtype == torch.bfloat16 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = scoring_dtype == torch.bfloat16
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    protocol_status = protocol.get("status")
    protocol_arm = None
    if protocol.get("status") in {
        "authorized_visual_token_scaling_sweep",
        "authorized_stage_d_lr_microsearch",
        "authorized_stage_d_two_arm_five_seed_replication",
    }:
        if args.arm is None or args.arm not in protocol.get("arms", {}):
            raise SystemExit("a declared --arm is required for a visual-token sweep protocol")
        protocol_arm = args.arm
        protocol = {**protocol, "architecture": protocol["arms"][args.arm]["architecture"]}
    elif protocol.get("status") in BINDING_DEVELOPMENT_STATUSES:
        if args.arm is not None:
            raise SystemExit("binding development protocol is a single-arm protocol")
        if args.split not in {"mechanism_dev", "selection_dev"}:
            raise SystemExit("binding development evaluator may only use development splits")
        if sha256_file(args.manifest) != protocol["data"]["manifest_sha256"]:
            raise SystemExit("binding evaluation manifest hash mismatch")
        if args.families_per_task is not None:
            raise SystemExit("binding evaluation selects complete scene pairs, not individual families")
    elif args.arm is not None:
        raise SystemExit("--arm is only valid for a declared multi-arm protocol")
    vision_feature_layer = int(protocol["architecture"].get("vision_feature_layer", -1))
    if protocol["architecture"]["visual_tokens"] != 196:
        raise SystemExit("visual-floor evaluator requires 196 input vision patches")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    balance_base_answer = protocol_status in {
        "authorized_stage_d_lr_microsearch",
        "authorized_stage_d_two_arm_five_seed_replication",
    }
    try:
        rows = select_evaluation_rows(
            [row for row in rows if row["split"] == args.split],
            args.families_per_task,
            balance_base_answer=balance_base_answer,
        )
        if protocol_status in BINDING_DEVELOPMENT_STATUSES:
            rows = select_binding_scene_pairs(rows, args.scene_pairs)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if len(rows) < 2:
        raise RuntimeError("need at least two scene families for random-image control")
    base = Path(protocol["parents"]["language"])
    vision_path = Path(protocol["parents"]["vision"])
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    span_audit = candidate_span_audit(tokenizer, rows, args.max_tokens)
    if not span_audit["all_nonempty"]:
        raise RuntimeError("one or more candidate answer spans are empty")
    image_audit = image_integrity_audit(rows) if args.verify_image_hashes else {
        "checked_images": 0,
        "mismatch_count": None,
        "all_match_manifest": None,
        "mismatches": [],
    }
    if image_audit["all_match_manifest"] is False:
        raise RuntimeError("one or more image files do not match the manifest hash")
    processor = AutoImageProcessor.from_pretrained(
        vision_path, local_files_only=True, use_fast=False
    )
    language = AutoModelForCausalLM.from_pretrained(
        base, dtype=scoring_dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    adapter_records = None
    if args.language_adapter is not None:
        from peft import PeftModel

        adapter_records = directory_records(args.language_adapter)
        language = PeftModel.from_pretrained(
            language, args.language_adapter, is_trainable=False, local_files_only=True
        )
    freeze(language)
    vision = SiglipVisionModel.from_pretrained(
        vision_path, dtype=scoring_dtype, local_files_only=True
    ).cuda()
    freeze(vision)
    bridge = bridge_from_architecture(protocol["architecture"]).to(device="cuda", dtype=scoring_dtype)
    state = load_file(args.checkpoint, device="cpu")
    bridge.load_state_dict(state, strict=True)
    freeze(bridge)
    conditions: dict[str, dict[str, list[str]]] = defaultdict(dict)
    score_diagnostics: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    random_control_audit = {}
    started = time.monotonic()
    for variant in VARIANTS:
        true_paths = [row[f"{variant}_image_path"] for row in rows]
        if "true_image" in args.conditions:
            conditions["true_image"][variant], score_diagnostics["true_image"][variant] = score_variant(
                rows, true_paths, language, vision, bridge, tokenizer, processor, args.batch_families, args.max_tokens, vision_feature_layer, args.candidate_order, scoring_dtype
            )
            print(f"EVAL_PROGRESS condition=true_image variant={variant} families={len(rows)} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
        if "random_image" in args.conditions:
            donors, donor_audit = random_control_assignment(rows, variant, args.random_control)
            random_paths = [rows[index][f"{variant}_image_path"] for index in donors]
            random_control_audit[variant] = donor_audit
            conditions["random_image"][variant], score_diagnostics["random_image"][variant] = score_variant(
                rows, random_paths, language, vision, bridge, tokenizer, processor, args.batch_families, args.max_tokens, vision_feature_layer, args.candidate_order, scoring_dtype
            )
            print(f"EVAL_PROGRESS condition=random_image variant={variant} families={len(rows)} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
        if "no_image" in args.conditions:
            conditions["no_image"][variant], score_diagnostics["no_image"][variant] = score_variant(
                rows, None, language, vision, bridge, tokenizer, processor, args.batch_families, args.max_tokens, vision_feature_layer, args.candidate_order, scoring_dtype
            )
            print(f"EVAL_PROGRESS condition=no_image variant={variant} families={len(rows)} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
    prediction_rows = []
    families = [row.get("statistical_cluster_id", row["scene_family_id"]) for row in rows]
    joint_by_condition: dict[str, list[float]] = {}
    invariant_joint_by_condition: dict[str, list[float]] = {}
    for condition, variants in conditions.items():
        base_correct = [prediction == row["base_answer"] for prediction, row in zip(variants["base"], rows)]
        edited_correct = [prediction == row["edited_answer"] for prediction, row in zip(variants["edited"], rows)]
        invariant_correct = [prediction == row["invariant_answer"] for prediction, row in zip(variants["invariant"], rows)]
        joint_by_condition[condition] = [float(left and right) for left, right in zip(base_correct, edited_correct)]
        invariant_joint_by_condition[condition] = [float(left and right) for left, right in zip(base_correct, invariant_correct)]
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "scene_family_id": row["scene_family_id"],
                "scene_pair_id": row.get("scene_pair_id"),
                "statistical_cluster_id": row.get("statistical_cluster_id", row["scene_family_id"]),
                "task": row["task"],
                "challenge": row.get("challenge"),
                "question_role": row.get("question_role"),
                "split": row["split"],
                "candidate_answers": row["candidate_answers"],
                "expected": {
                    "base": row["base_answer"],
                    "edited": row["edited_answer"],
                    "invariant": row["invariant_answer"],
                },
                "predictions": {
                    condition: {variant: values[variant][index] for variant in VARIANTS}
                    for condition, values in conditions.items()
                },
                "candidate_score_diagnostics": {
                    condition: {variant: values[variant][index] for variant in VARIANTS}
                    for condition, values in score_diagnostics.items()
                },
            }
        )
    binding_metrics = binding_selective_metrics(prediction_rows)
    true_joint = joint_by_condition.get("true_image")
    no_joint = joint_by_condition.get("no_image")
    random_joint = joint_by_condition.get("random_image")
    true_vs_no = None if true_joint is None or no_joint is None else interval(
        [a - b for a, b in zip(true_joint, no_joint)], families
    )
    true_vs_random = None if true_joint is None or random_joint is None else interval(
        [a - b for a, b in zip(true_joint, random_joint)], families
    )
    task_metrics = {}
    for task in sorted({row["task"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["task"] == task]
        task_metrics[task] = {
            condition: 100 * sum(joint_by_condition[condition][index] for index in indices) / len(indices)
            for condition in joint_by_condition
        }
    role_metrics = {}
    for role in sorted({row.get("question_role") for row in rows if row.get("question_role") is not None}):
        indices = [index for index, row in enumerate(rows) if row.get("question_role") == role]
        role_metrics[role] = {
            condition: 100 * sum(joint_by_condition[condition][index] for index in indices) / len(indices)
            for condition in joint_by_condition
        }
    challenge_metrics = {}
    for challenge in sorted({row.get("challenge") for row in rows if row.get("challenge") is not None}):
        indices = [index for index, row in enumerate(rows) if row.get("challenge") == challenge]
        challenge_metrics[challenge] = {
            condition: 100 * sum(joint_by_condition[condition][index] for index in indices) / len(indices)
            for condition in joint_by_condition
        }
    result = {
        "schema_version": "2026-09-13-v1",
        "scope": (
            "binding_intervention_development"
            if protocol_status in BINDING_DEVELOPMENT_STATUSES
            else (
                "stage_a_full_token_controlled_development_visual_floor"
                if bridge.spec.output_tokens == bridge.spec.input_tokens
                else "stage_b_compressed_controlled_development_visual_floor"
            )
        ),
        "split": args.split,
        "selection": {
            "families_per_task": args.families_per_task,
            "scene_pairs": args.scene_pairs,
            "stratified_by": (
                ["scene_pair_id"]
                if protocol_status in BINDING_DEVELOPMENT_STATUSES
                else (["task", "base_answer"] if balance_base_answer else ["task"])
            ),
            "purpose": (
                "checkpoint_screen"
                if args.scene_pairs is not None or args.families_per_task is not None
                else "full_split_gate"
            ),
        },
        "scene_families": len(rows),
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "language_adapter": None if args.language_adapter is None else {
            "path": str(args.language_adapter),
            "files": adapter_records,
        },
        "protocol_sha256": sha256_file(args.protocol),
        "protocol_arm": protocol_arm,
        "candidate_order": args.candidate_order,
        "scoring_precision": args.scoring_precision,
        "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
        "conditions": list(args.conditions),
        "random_control": {
            "policy": args.random_control,
            "audit": random_control_audit,
            "claim_boundary": (
                "within_task_answer_matched controls task/render and donor-label strata while breaking exact-family pairing; "
                "it is a conservative nuisance control, not an estimate of natural-image performance."
            ),
        },
        "candidate_answer_span_audit": span_audit,
        "image_integrity_audit": image_audit,
        "vision_feature_layer": vision_feature_layer,
        "input_visual_tokens": bridge.spec.input_tokens,
        "output_visual_tokens": bridge.spec.output_tokens,
        "achieved_compression_ratio": bridge.spec.achieved_ratio,
        "parent_records": {"language": release_records(base), "vision": vision_records(vision_path)},
        "paired_joint_accuracy_percent": {
            condition: 100 * sum(values) / len(values) for condition, values in joint_by_condition.items()
        },
        "base_plus_invariant_joint_accuracy_percent": {
            condition: 100 * sum(values) / len(values)
            for condition, values in invariant_joint_by_condition.items()
        },
        "true_minus_no_image": true_vs_no,
        "true_minus_random_image": true_vs_random,
        "per_task_paired_joint_accuracy_percent": task_metrics,
        "per_question_role_paired_joint_accuracy_percent": role_metrics,
        "per_challenge_paired_joint_accuracy_percent": challenge_metrics,
        "binding_selective_metrics": binding_metrics,
        "metric_contract": metric_contract(prediction_rows),
        "predictions": prediction_rows,
        "training_prediction_tokens": 0,
        "total_wall_seconds": time.monotonic() - process_started,
        "claim_boundary": (
            "Development-set visual-use evidence is not test-set confirmation, real-image transfer, or publication evidence. "
            "For compressed protocols, it measures only the declared controlled-domain compression setting."
        ),
    }
    if true_joint is not None and true_vs_no is not None and true_vs_random is not None:
        if binding_metrics is None:
            floor_accuracy = 100 * sum(true_joint) / len(true_joint)
            floor_no = true_vs_no
            floor_random = true_vs_random
            accuracy_gate_name = "true_pair_joint_accuracy_gte_70"
        else:
            floor_accuracy = binding_metrics["per_condition"]["true_image"][
                "affected_question_joint_accuracy_percent"
            ]
            floor_no = binding_metrics["affected_true_minus_no_image"]
            floor_random = binding_metrics["affected_true_minus_random_image"]
            accuracy_gate_name = "true_affected_question_joint_accuracy_gte_70"
        result["visual_floor"] = {
            accuracy_gate_name: floor_accuracy >= 70.0,
            "true_vs_no_lower_95_ci_gt_0": floor_no["lower_95_ci_pp"] > 0,
            "true_vs_random_lower_95_ci_gt_0": floor_random["lower_95_ci_pp"] > 0,
        }
        result["visual_floor"]["passes_all"] = all(result["visual_floor"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"paired_joint_accuracy_percent": result["paired_joint_accuracy_percent"], "visual_floor": result.get("visual_floor")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
