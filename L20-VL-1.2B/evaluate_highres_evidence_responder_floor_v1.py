#!/usr/bin/env python3
"""Forward-only language-path floor for the high-resolution evidence corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image


STATUS = "authorized_highres_evidence_responder_development_floor_only_v1"
CANDIDATES = {
    "read_local_digit": [str(value) for value in range(10)],
    "read_global_border": ["purple", "teal", "gold", "navy"],
}


def load_development_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == "development"]
    if any(row["split"] == "sealed_test" for row in selected):
        raise RuntimeError("sealed test row reached development evaluator")
    selected.sort(key=lambda row: (row["family_id"], row["variant"]))
    return selected


def collate_candidates(tokenizer, rows: list[dict[str, Any]], reverse: bool, max_tokens: int):
    import torch
    from train_stage_a_full_token import encode_prompt_response

    encoded, identities = [], []
    for row_index, row in enumerate(rows):
        candidates = list(CANDIDATES[row["task"]])
        if reverse:
            candidates.reverse()
        for candidate in candidates:
            encoded.append(encode_prompt_response(tokenizer, row["question"], candidate, max_tokens))
            identities.append((row_index, candidate))
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    for index, (ids, targets) in enumerate(encoded):
        input_ids[index, :len(ids)] = torch.tensor(ids)
        labels[index, :len(ids)] = torch.tensor(targets)
        attention[index, :len(ids)] = 1
    return input_ids, attention, labels, identities


def inject_two_views(bridge, text_embeddings, text_mask, labels, global_features, crop_features):
    import torch

    if global_features.shape != crop_features.shape:
        raise ValueError("global/crop feature shape mismatch")
    batch = text_embeddings.shape[0]
    global_visual = bridge.visual_tokens(global_features)
    crop_visual = bridge.visual_tokens(crop_features)
    start = bridge.image_start.expand(batch, -1, -1)
    end = bridge.image_end.expand(batch, -1, -1)
    inputs = torch.cat((start, global_visual, end, start, crop_visual, end, text_embeddings), dim=1)
    prefix_length = 2 * (global_visual.shape[1] + 2)
    prefix_mask = torch.ones(batch, prefix_length, dtype=text_mask.dtype, device=text_mask.device)
    mask = torch.cat((prefix_mask, text_mask), dim=1)
    ignored = torch.full((batch, prefix_length), -100, dtype=labels.dtype, device=labels.device)
    return inputs, mask, torch.cat((ignored, labels), dim=1)


def answer_scores(logits, targets, eos_token_id: int):
    import torch

    shifted = targets[:, 1:]
    token_mask = shifted.ne(-100) & shifted.ne(eos_token_id)
    if not token_mask.any(dim=1).all():
        raise RuntimeError("candidate has no non-EOS answer token")
    safe = shifted.masked_fill(~token_mask, 0)
    logprob = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (logprob * token_mask).sum(dim=-1) / token_mask.sum(dim=-1)


def condition_metrics(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    correct = [prediction == row["answer"] for prediction, row in zip(predictions, rows)]
    by_family: dict[str, list[bool]] = defaultdict(list)
    for row, value in zip(rows, correct):
        by_family[row["family_id"]].append(value)
    if any(len(values) != 2 for values in by_family.values()):
        raise RuntimeError("each family must contain two variants")
    return {
        "rows": len(rows),
        "families": len(by_family),
        "row_accuracy": sum(correct) / len(correct),
        "family_joint_accuracy": sum(all(values) for values in by_family.values()) / len(by_family),
        "prediction_distribution": dict(sorted(Counter(predictions).items())),
    }


def paired_bootstrap(rows, left, right, samples: int, seed: int) -> dict[str, Any]:
    families = sorted({row["family_id"] for row in rows})
    indices = {family: [index for index, row in enumerate(rows) if row["family_id"] == family] for family in families}
    left_correct = np.asarray([prediction == row["answer"] for prediction, row in zip(left, rows)], dtype=float)
    right_correct = np.asarray([prediction == row["answer"] for prediction, row in zip(right, rows)], dtype=float)
    deltas = np.asarray([(left_correct[indices[family]] - right_correct[indices[family]]).mean() for family in families])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(families), size=(samples, len(families)))
    estimates = deltas[draws].mean(axis=1)
    return {
        "point_estimate": float(deltas.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": len(families),
        "bootstrap_samples": samples,
    }


def cache_features(rows, protocol, processor, vision, dtype):
    import torch
    from modeling import vision_features

    root = Path(protocol["data"]["root"])
    global_cache, crop_cache = {}, {}
    batch_size = int(protocol["evaluation"]["feature_batch_images"])
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images, crop_rows, crops = [], [], []
            for row in batch:
                with Image.open(root / row["image_path"]) as source:
                    image = source.convert("RGB")
                images.append(image)
                box = row["target_crop_box_xyxy_normalized"]
                if box is not None:
                    pixels = tuple(round(value * image.width) for value in box)
                    crop_rows.append(row)
                    crops.append(image.crop(pixels))
            all_images = images + crops
            pixel_values = processor(images=all_images, return_tensors="pt")["pixel_values"].to("cuda", dtype=dtype)
            features = vision_features(vision, pixel_values, int(protocol["architecture"]["vision_feature_layer"])).to("cpu", dtype=dtype)
            for row, feature in zip(batch, features[:len(batch)]):
                global_cache[(row["family_id"], row["variant"])] = feature.contiguous()
            for row, feature in zip(crop_rows, features[len(batch):]):
                crop_cache[(row["family_id"], row["variant"])] = feature.contiguous()
    return global_cache, crop_cache


def mismatched_crop_mapping(rows: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[str, str]]:
    local_families = sorted({row["family_id"] for row in rows if row["task"] == "read_local_digit"})
    mapping = {}
    for index, family in enumerate(local_families):
        other = local_families[(index + 1) % len(local_families)]
        for variant in ("base", "answer_change"):
            mapping[(family, variant)] = (other, variant)
    return mapping


def score_condition(rows, condition, reverse, protocol, language, bridge, tokenizer, caches, mismatch):
    import torch

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
        feature_indices = [row_index for row_index, _ in identities]
        if condition == "no_image":
            inputs, mask, targets = text, attention, labels
        else:
            global_features = torch.stack([
                global_cache[(batch[index]["family_id"], batch[index]["variant"])]
                for index in feature_indices
            ]).to("cuda", non_blocking=True)
            if condition == "global":
                inputs, mask, targets = bridge.inject(text, attention, labels, global_features)
            else:
                crop_keys = []
                for index in feature_indices:
                    key = (batch[index]["family_id"], batch[index]["variant"])
                    crop_keys.append(key if condition in {"crop", "global_crop"} else mismatch[key])
                crop_features = torch.stack([crop_cache[key] for key in crop_keys]).to("cuda", non_blocking=True)
                if condition == "crop":
                    inputs, mask, targets = bridge.inject(text, attention, labels, crop_features)
                elif condition in {"global_crop", "global_mismatched_crop"}:
                    inputs, mask, targets = inject_two_views(
                        bridge, text, attention, labels, global_features, crop_features
                    )
                else:
                    raise ValueError(f"unknown condition: {condition}")
        with torch.inference_mode():
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel
    from modeling import bridge_from_architecture, freeze
    from train_stage_a_full_token import release_records, sha256_file, utc_now, write_json_atomic

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("responder development floor is not authorized")
    for field in ("parameter_update_authorized", "sealed_test_access_authorized"):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    root = Path(__file__).resolve().parent
    for path, key in ((Path(__file__), "evaluator_sha256"), (root / "test_evaluate_highres_evidence_responder_floor_v1.py", "test_sha256")):
        if sha256_file(path) != protocol["source_code"][key]:
            raise RuntimeError(f"source hash mismatch: {path.name}")
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"prerequisite hash mismatch: {name}")
    language_root = Path(protocol["parents"]["language"])
    if sha256_file(language_root / "release-manifest.json") != protocol["parents"]["language_release_manifest_sha256"]:
        raise RuntimeError("language release manifest changed")
    language_files = release_records(language_root)
    for name, expected in protocol["parents"]["vision_artifacts"].items():
        if sha256_file(Path(protocol["parents"]["vision"]) / name) != expected:
            raise RuntimeError(f"vision artifact hash mismatch: {name}")
    if sha256_file(Path(protocol["parents"]["bridge"])) != protocol["parents"]["bridge_sha256"]:
        raise RuntimeError("bridge hash mismatch")
    for name, expected in protocol["parents"]["language_adapter_files"].items():
        if sha256_file(Path(protocol["parents"]["language_adapter"]) / name) != expected:
            raise RuntimeError(f"language adapter hash mismatch: {name}")
    receipt_path = Path(protocol["output"]["receipt"])
    predictions_path = Path(protocol["output"]["predictions"])
    for path in (receipt_path, predictions_path):
        if path.exists():
            raise FileExistsError(path)

    rows = load_development_rows(Path(protocol["data"]["manifest"]))
    local_rows = [row for row in rows if row["task"] == "read_local_digit"]
    global_rows = [row for row in rows if row["task"] == "read_global_border"]
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(protocol["parents"]["language"], local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(protocol["parents"]["vision"], local_files_only=True, use_fast=False)
    vision = SiglipVisionModel.from_pretrained(protocol["parents"]["vision"], dtype=dtype, local_files_only=True).cuda()
    freeze(vision)
    language = AutoModelForCausalLM.from_pretrained(protocol["parents"]["language"], dtype=dtype, local_files_only=True, low_cpu_mem_usage=True).cuda()
    language = PeftModel.from_pretrained(language, protocol["parents"]["language_adapter"], is_trainable=False, local_files_only=True)
    freeze(language)
    bridge = bridge_from_architecture(protocol["architecture"]).to("cuda", dtype=dtype)
    bridge.load_state_dict(load_file(protocol["parents"]["bridge"], device="cpu"), strict=True)
    freeze(bridge)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    caches = cache_features(rows, protocol, processor, vision, dtype)
    mismatch = mismatched_crop_mapping(rows)
    condition_rows = {
        "global": rows,
        "no_image": rows,
        "crop": local_rows,
        "global_crop": local_rows,
        "global_mismatched_crop": local_rows,
    }
    predictions, all_records = {}, []
    order_invariance = {}
    with torch.inference_mode():
        for condition, subset in condition_rows.items():
            normal, records = score_condition(subset, condition, False, protocol, language, bridge, tokenizer, caches, mismatch)
            reversed_order, reverse_records = score_condition(subset, condition, True, protocol, language, bridge, tokenizer, caches, mismatch)
            predictions[condition] = normal
            all_records.extend(records)
            order_invariance[condition] = {
                "rows": len(subset),
                "prediction_mismatches": sum(a != b for a, b in zip(normal, reversed_order)),
                "passes": normal == reversed_order,
            }
            all_records.extend(reverse_records)
    elapsed = time.perf_counter() - started
    metrics_by_condition = {
        condition: condition_metrics(condition_rows[condition], values)
        for condition, values in predictions.items()
    }
    stats = protocol["statistics"]
    deltas = {
        "global_crop_minus_mismatched_crop": paired_bootstrap(local_rows, predictions["global_crop"], predictions["global_mismatched_crop"], int(stats["bootstrap_samples"]), int(stats["seed"]) + 2),
    }
    # Rows are sorted by family; local families precede/follow global families,
    # so create an explicit local global-view prediction rather than rely on slicing.
    global_by_id = {
        (row["family_id"], row["variant"]): prediction
        for row, prediction in zip(rows, predictions["global"])
    }
    local_global = [global_by_id[(row["family_id"], row["variant"])] for row in local_rows]
    deltas["crop_minus_global"] = paired_bootstrap(local_rows, predictions["crop"], local_global, int(stats["bootstrap_samples"]), int(stats["seed"]))
    deltas["global_crop_minus_global"] = paired_bootstrap(local_rows, predictions["global_crop"], local_global, int(stats["bootstrap_samples"]), int(stats["seed"]) + 1)
    with predictions_path.open("x") as handle:
        for record in all_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    gate_passed = (
        metrics_by_condition["global_crop"]["row_accuracy"] >= float(protocol["gate"]["minimum_global_crop_local_accuracy"])
        and deltas["global_crop_minus_global"]["ci95_low"] > float(protocol["gate"]["minimum_global_crop_minus_global_ci95_low"])
        and deltas["global_crop_minus_mismatched_crop"]["ci95_low"] > float(protocol["gate"]["minimum_true_minus_mismatched_ci95_low"])
        and all(item["passes"] for item in order_invariance.values())
    )
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_highres_evidence_responder_development_floor",
        "completed_at": utc_now(),
        "decision": "pass_existing_responder_uses_crop" if gate_passed else "existing_responder_requires_adaptation",
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "verified_parent_records": {
            "language": language_files,
            "vision_artifacts": protocol["parents"]["vision_artifacts"],
            "bridge_sha256": protocol["parents"]["bridge_sha256"],
            "language_adapter_files": protocol["parents"]["language_adapter_files"],
        },
        "rows": {"all_development": len(rows), "local_digit": len(local_rows), "global_border": len(global_rows)},
        "metrics": metrics_by_condition,
        "paired_deltas": deltas,
        "candidate_order_invariance": order_invariance,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path), "bytes": predictions_path.stat().st_size},
        "parameter_updates": 0,
        "sealed_test_rows_accessed": 0,
        "gate_passed": gate_passed,
        "next_allowed_step": (
            "Freeze router/responder baselines before training."
            if gate_passed
            else "Freeze a bounded oracle-crop responder adaptation protocol; do not train the value router until the responder can use crop evidence."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({key: receipt[key] for key in ("decision", "metrics", "paired_deltas", "candidate_order_invariance", "elapsed_seconds", "peak_cuda_memory_bytes")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
