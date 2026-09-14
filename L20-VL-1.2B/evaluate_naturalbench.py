#!/usr/bin/env python3
"""One-shot, zero-shot NaturalBench evaluation for frozen L20-VL arms."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable


OFFICIAL_COMBINATIONS = (
    (0, 0, "q0_i0"),
    (0, 1, "q0_i1"),
    (1, 0, "q1_i0"),
    (1, 1, "q1_i1"),
)
CANDIDATES = {
    "yes_no": ("Yes", "No"),
    "multiple_choice": ("A", "B"),
}
OFFICIAL_METRICS = ("Q_Acc", "I_Acc", "Acc", "G_Acc")
BALANCED_RANK_METRICS = (
    "Balanced_Q_Acc",
    "Balanced_I_Acc",
    "Balanced_G_Acc",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_components(correct: dict[str, bool]) -> dict[str, float]:
    """Return per-group contributions exactly equivalent to official NaturalBench metrics."""
    required = {name for _, _, name in OFFICIAL_COMBINATIONS}
    if set(correct) != required:
        raise ValueError(f"expected correctness keys {sorted(required)}, got {sorted(correct)}")
    question = (
        float(correct["q0_i0"] and correct["q0_i1"])
        + float(correct["q1_i0"] and correct["q1_i1"])
    ) / 2.0
    image = (
        float(correct["q0_i0"] and correct["q1_i0"])
        + float(correct["q0_i1"] and correct["q1_i1"])
    ) / 2.0
    binary = sum(float(value) for value in correct.values()) / 4.0
    group = float(all(correct.values()))
    return {"Q_Acc": question, "I_Acc": image, "Acc": binary, "G_Acc": group}


def balanced_rank_components(margins: dict[str, float]) -> dict[str, float]:
    """Sample-balanced likelihood ranking metrics from NaturalBench Section 5."""
    required = {name for _, _, name in OFFICIAL_COMBINATIONS}
    if set(margins) != required:
        raise ValueError(f"expected margin keys {sorted(required)}, got {sorted(margins)}")
    question = (
        float(margins["q0_i0"] > margins["q0_i1"])
        + float(margins["q1_i1"] > margins["q1_i0"])
    ) / 2.0
    image = (
        float(margins["q0_i0"] > margins["q1_i0"])
        + float(margins["q1_i1"] > margins["q0_i1"])
    ) / 2.0
    positive = (margins["q0_i0"], margins["q1_i1"])
    negative = (margins["q0_i1"], margins["q1_i0"])
    group = float(min(positive) > max(negative))
    return {
        "Balanced_Q_Acc": question,
        "Balanced_I_Acc": image,
        "Balanced_G_Acc": group,
    }


def mean_metrics(components: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(components)
    if not rows:
        raise ValueError("cannot score zero NaturalBench groups")
    return {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in OFFICIAL_METRICS
    }


def bootstrap_interval(
    values: list[float], *, resamples: int, seed: int
) -> dict[str, float | int]:
    import numpy as np

    if not values:
        raise ValueError("cannot bootstrap zero values")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    chunk = 512
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        estimates[start:stop] = array[indices].mean(axis=1)
    return {
        "estimate_percent": 100.0 * float(array.mean()),
        "lower_95_ci_percent": 100.0 * float(np.quantile(estimates, 0.025)),
        "upper_95_ci_percent": 100.0 * float(np.quantile(estimates, 0.975)),
        "groups": len(values),
        "resamples": resamples,
    }


def summarize_components(
    components: list[dict[str, float]], *, resamples: int, seed: int
) -> dict[str, dict[str, float | int]]:
    return {
        metric: bootstrap_interval(
            [row[metric] for row in components],
            resamples=resamples,
            seed=seed + index,
        )
        for index, metric in enumerate(OFFICIAL_METRICS)
    }


def summarize_named_components(
    components: list[dict[str, float]],
    metrics: tuple[str, ...],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    return {
        metric: bootstrap_interval(
            [row[metric] for row in components],
            resamples=resamples,
            seed=seed + index,
        )
        for index, metric in enumerate(metrics)
    }


def paired_comparison(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    if len(left) != len(right):
        raise ValueError("paired NaturalBench arms must contain the same groups")
    result = {}
    for index, metric in enumerate(OFFICIAL_METRICS + BALANCED_RANK_METRICS):
        interval = bootstrap_interval(
            [a[metric] - b[metric] for a, b in zip(left, right)],
            resamples=resamples,
            seed=seed + index,
        )
        result[metric] = {
            "estimate_pp": interval["estimate_percent"],
            "lower_95_ci_pp": interval["lower_95_ci_percent"],
            "upper_95_ci_pp": interval["upper_95_ci_percent"],
            "groups": interval["groups"],
            "resamples": interval["resamples"],
        }
    return result


def subgroup_metrics(predictions: list[dict[str, Any]], field: str) -> dict[str, Any]:
    result = {}
    for value in sorted({str(row[field]) for row in predictions}):
        rows = [row for row in predictions if str(row[field]) == value]
        components = [
            {**row["metric_components"], **row["balanced_rank_components"]}
            for row in rows
        ]
        result[value] = {
            "groups": len(rows),
            "official_metrics": mean_metrics(components),
            "balanced_rank_metrics": {
                metric: sum(row[metric] for row in components) / len(components)
                for metric in BALANCED_RANK_METRICS
            },
        }
    return result


def decode_image(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if isinstance(value, dict) and value.get("bytes") is not None:
        with Image.open(io.BytesIO(value["bytes"])) as image:
            return image.convert("RGB").copy()
    if isinstance(value, dict) and value.get("path"):
        with Image.open(value["path"]) as image:
            return image.convert("RGB").copy()
    raise TypeError(f"unsupported NaturalBench image value: {type(value)!r}")


class LazyDatasetRows:
    """List-like access without materializing decoded benchmark images."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        for index in range(len(self)):
            yield self.dataset[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.dataset[offset] for offset in range(*index.indices(len(self)))]
        return self.dataset[index]


def collate_candidates(tokenizer, prompts: list[str], answers: list[str], max_tokens: int):
    import torch

    from train_stage_a_full_token import encode_prompt_response

    encoded = [
        encode_prompt_response(tokenizer, prompt, answer, max_tokens)
        for prompt, answer in zip(prompts, answers)
    ]
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


def candidate_span_audit(tokenizer, rows, max_tokens: int) -> dict[str, Any]:
    from train_stage_a_full_token import encode_prompt_response

    lengths = []
    for row in rows:
        candidates = CANDIDATES[row["Question Type"]]
        for question_index in (0, 1):
            for answer in candidates:
                _, labels = encode_prompt_response(
                    tokenizer, row[f"Question_{question_index}"], answer, max_tokens
                )
                lengths.append(sum(label != -100 for label in labels) - 1)
    return {
        "candidate_sequences": len(lengths),
        "nonempty_answer_spans": sum(length > 0 for length in lengths),
        "minimum_answer_tokens": min(lengths),
        "maximum_answer_tokens": max(lengths),
        "all_nonempty": all(length > 0 for length in lengths),
        "eos_excluded_from_candidate_score": True,
    }


def answer_token_mask(targets, labels, text_attention):
    """Mask candidate answer tokens while excluding the terminal EOS token."""
    import torch

    target_mask = targets[:, 1:].ne(-100)
    prefix_tokens = targets.shape[1] - labels.shape[1]
    eos_positions = prefix_tokens + text_attention.sum(dim=1) - 2
    target_mask[
        torch.arange(target_mask.shape[0], device=target_mask.device), eos_positions
    ] = False
    return target_mask


def dataset_records(dataset_root: Path, expected: dict[str, str]) -> dict[str, dict[str, Any]]:
    records = {}
    for relative, expected_sha in sorted(expected.items()):
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError(f"NaturalBench file hash mismatch: {relative}")
        records[relative] = {"bytes": path.stat().st_size, "sha256": actual}
    return records


def validate_dataset(rows, expected_rows: int) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"NaturalBench row count {len(rows)} != {expected_rows}")
    type_counts = {kind: 0 for kind in CANDIDATES}
    source_counts: dict[str, int] = {}
    failures = []
    for offset, row in enumerate(rows):
        kind = row["Question Type"]
        if kind not in CANDIDATES:
            failures.append({"row": offset, "reason": f"unknown question type {kind}"})
            continue
        type_counts[kind] += 1
        source = str(row["Source"])
        source_counts[source] = source_counts.get(source, 0) + 1
        answers = CANDIDATES[kind]
        expected = {
            "q0_i0": row["Image_0_Question_0"],
            "q0_i1": row["Image_1_Question_0"],
            "q1_i0": row["Image_0_Question_1"],
            "q1_i1": row["Image_1_Question_1"],
        }
        if tuple(expected.values()) != (answers[0], answers[1], answers[1], answers[0]):
            failures.append({"row": offset, "reason": "official paired-answer invariant failed"})
    return {
        "rows": len(rows),
        "question_type_counts": type_counts,
        "source_counts": source_counts,
        "paired_answer_invariant_failures": failures,
        "passes": not failures,
    }


def score_arm(
    rows,
    *,
    arm_name: str,
    arm: dict[str, Any],
    protocol: dict[str, Any],
    tokenizer,
    processor,
    vision,
    dtype,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from modeling import bridge_from_architecture, freeze, vision_features

    base = Path(protocol["parents"]["language"])
    checkpoint = Path(arm["checkpoint"])
    adapter = Path(arm["language_adapter"])
    if sha256_file(checkpoint) != arm["checkpoint_sha256"]:
        raise RuntimeError(f"{arm_name} bridge checkpoint hash mismatch")
    for relative, expected_sha in arm["language_adapter_files"].items():
        if sha256_file(adapter / relative) != expected_sha:
            raise RuntimeError(f"{arm_name} adapter hash mismatch: {relative}")

    language = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(
        language, adapter, is_trainable=False, local_files_only=True
    )
    freeze(language)
    bridge = bridge_from_architecture(arm["architecture"]).to(device="cuda", dtype=dtype)
    bridge.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
    freeze(bridge)

    batch_groups = int(protocol["evaluation"]["batch_groups"])
    max_tokens = int(protocol["evaluation"]["max_text_tokens"])
    layer = int(arm["architecture"].get("vision_feature_layer", -1))
    predictions = []
    components = []
    tie_count = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_groups):
            batch = rows[start : start + batch_groups]
            images = []
            for row in batch:
                images.extend((decode_image(row["Image_0"]), decode_image(row["Image_1"])))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
                "cuda", dtype=dtype
            )
            unique_visual = vision_features(vision, pixels, layer)

            prompts: list[str] = []
            candidate_answers: list[str] = []
            visual_indices: list[int] = []
            identities: list[tuple[int, str, str, str]] = []
            for local_index, row in enumerate(batch):
                kind = row["Question Type"]
                candidates = CANDIDATES[kind]
                for question_index, image_index, combination in OFFICIAL_COMBINATIONS:
                    prompt = row[f"Question_{question_index}"]
                    expected = row[f"Image_{image_index}_Question_{question_index}"]
                    for candidate in candidates:
                        prompts.append(prompt)
                        candidate_answers.append(candidate)
                        visual_indices.append(2 * local_index + image_index)
                        identities.append((local_index, combination, expected, candidate))
            input_ids, attention, labels = collate_candidates(
                tokenizer, prompts, candidate_answers, max_tokens
            )
            input_ids = input_ids.cuda(non_blocking=True)
            attention = attention.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            text_embeddings = language.get_input_embeddings()(input_ids)
            visual = unique_visual[
                torch.tensor(visual_indices, dtype=torch.long, device=unique_visual.device)
            ]
            inputs, mask, targets = bridge.inject(
                text_embeddings, attention, labels, visual
            )
            logits = language(inputs_embeds=inputs, attention_mask=mask).logits[:, :-1].float()
            shifted = targets[:, 1:]
            target_mask = answer_token_mask(targets, labels, attention)
            if not target_mask.any(dim=1).all():
                raise RuntimeError("NaturalBench candidate has an empty answer span")
            safe_targets = shifted.masked_fill(~target_mask, 0)
            token_logprob = torch.log_softmax(logits, dim=-1).gather(
                -1, safe_targets.unsqueeze(-1)
            ).squeeze(-1)
            sequence_logprob = (token_logprob * target_mask).sum(dim=-1).cpu().tolist()

            batch_scores: list[dict[str, dict[str, Any]]] = [dict() for _ in batch]
            for sequence_index, (local_index, combination, expected, candidate) in enumerate(identities):
                slot = batch_scores[local_index].setdefault(
                    combination, {"expected": expected, "candidate_sequence_logprobs": {}}
                )
                slot["candidate_sequence_logprobs"][candidate] = float(sequence_logprob[sequence_index])
            for local_index, (row, scores) in enumerate(zip(batch, batch_scores)):
                group_prediction = {
                    "index": int(row["Index"]),
                    "source": str(row["Source"]),
                    "question_type": row["Question Type"],
                    "combinations": {},
                }
                correct = {}
                margins = {}
                for _, _, combination in OFFICIAL_COMBINATIONS:
                    slot = scores[combination]
                    candidates = CANDIDATES[row["Question Type"]]
                    left = slot["candidate_sequence_logprobs"][candidates[0]]
                    right = slot["candidate_sequence_logprobs"][candidates[1]]
                    if left == right:
                        tie_count += 1
                        predicted = None
                    else:
                        predicted = candidates[0] if left > right else candidates[1]
                    is_correct = predicted == slot["expected"]
                    correct[combination] = is_correct
                    margins[combination] = left - right
                    group_prediction["combinations"][combination] = {
                        **slot,
                        "predicted": predicted,
                        "correct": is_correct,
                        "absolute_margin": abs(left - right),
                    }
                group_prediction["metric_components"] = group_components(correct)
                group_prediction["balanced_rank_components"] = balanced_rank_components(margins)
                predictions.append(group_prediction)
                components.append({
                    **group_prediction["metric_components"],
                    **group_prediction["balanced_rank_components"],
                })
            completed = min(start + len(batch), len(rows))
            if completed == len(rows) or completed % 100 == 0:
                print(
                    f"NATURALBENCH_PROGRESS arm={arm_name} groups={completed}/{len(rows)}",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    return {
        "arm": arm_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "language_adapter": str(adapter),
        "architecture": arm["architecture"],
        "official_metrics": mean_metrics(components),
        "official_metrics_cluster_bootstrap_95_ci": summarize_components(
            components,
            resamples=int(protocol["statistics"]["bootstrap_resamples"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]),
        ),
        "balanced_rank_metrics": {
            metric: sum(row[metric] for row in components) / len(components)
            for metric in BALANCED_RANK_METRICS
        },
        "balanced_rank_metrics_cluster_bootstrap_95_ci": summarize_named_components(
            components,
            BALANCED_RANK_METRICS,
            resamples=int(protocol["statistics"]["bootstrap_resamples"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]) + 100,
        ),
        "descriptive_subgroups": {
            "source": subgroup_metrics(predictions, "source"),
            "question_type": subgroup_metrics(predictions, "question_type"),
        },
        "tie_count": tie_count,
        "groups": len(components),
        "candidate_sequences": 8 * len(components),
        "wall_seconds": elapsed,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "predictions": predictions,
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "authorized_naturalbench_external_zero_shot_v2":
        raise SystemExit("NaturalBench evaluation requires its frozen external-evaluation protocol")
    if protocol.get("training_allowed") is not False:
        raise SystemExit("NaturalBench protocol must explicitly prohibit training")
    root = Path(__file__).resolve().parent
    for filename, key in (
        ("evaluate_naturalbench.py", "evaluator_sha256"),
        ("test_evaluate_naturalbench.py", "evaluator_test_sha256"),
        (protocol["source_code"]["launcher_filename"], "launcher_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("train_stage_a_full_token.py", "prompt_encoder_sha256"),
    ):
        if sha256_file(root / filename) != protocol["source_code"][key]:
            raise SystemExit(f"NaturalBench source hash mismatch: {filename}")

    import torch
    from datasets import Image as DatasetImage, load_dataset
    from transformers import AutoImageProcessor, AutoTokenizer, SiglipVisionModel

    from modeling import freeze
    from train_stage_a_full_token import release_records, vision_records

    dtype = torch.bfloat16
    torch.manual_seed(int(protocol["statistics"]["bootstrap_seed"]))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset_root = Path(protocol["dataset"]["root"])
    records = dataset_records(dataset_root, protocol["dataset"]["files"])
    parquet_files = [
        str(dataset_root / relative)
        for relative in sorted(protocol["dataset"]["files"])
        if relative.endswith(".parquet")
    ]
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    dataset = dataset.cast_column("Image_0", DatasetImage(decode=False))
    dataset = dataset.cast_column("Image_1", DatasetImage(decode=False))
    rows = LazyDatasetRows(dataset)
    if protocol["evaluation"].get("dataset_access") != "lazy_rows_image_decode_false":
        raise RuntimeError("NaturalBench v2 requires frozen lazy image decoding")
    dataset_audit = validate_dataset(rows, int(protocol["dataset"]["expected_groups"]))
    if not dataset_audit["passes"]:
        raise RuntimeError("NaturalBench dataset audit failed")

    base = Path(protocol["parents"]["language"])
    vision_path = Path(protocol["parents"]["vision"])
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    span_audit = candidate_span_audit(
        tokenizer, rows, int(protocol["evaluation"]["max_text_tokens"])
    )
    if not span_audit["all_nonempty"]:
        raise RuntimeError("one or more NaturalBench candidate spans are empty")
    processor = AutoImageProcessor.from_pretrained(
        vision_path, local_files_only=True, use_fast=False
    )
    vision = SiglipVisionModel.from_pretrained(
        vision_path, dtype=dtype, local_files_only=True
    ).cuda()
    freeze(vision)
    arm_results = {}
    total_started = time.monotonic()
    for arm_name in protocol["evaluation"]["arm_order"]:
        arm_results[arm_name] = score_arm(
            rows,
            arm_name=arm_name,
            arm=protocol["arms"][arm_name],
            protocol=protocol,
            tokenizer=tokenizer,
            processor=processor,
            vision=vision,
            dtype=dtype,
        )
        gc.collect()
        torch.cuda.empty_cache()

    comparisons = {}
    for item in protocol["statistics"]["paired_comparisons"]:
        left, right = item["left"], item["right"]
        comparisons[f"{left}_minus_{right}"] = paired_comparison(
            arm_results[left]["components"],
            arm_results[right]["components"],
            resamples=int(protocol["statistics"]["bootstrap_resamples"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]),
        )
    all_ties_zero = all(result["tie_count"] == 0 for result in arm_results.values())
    primary = protocol["decision_rule"]["primary_comparison"]
    primary_result = comparisons[primary]["G_Acc"]
    decision = (
        "external_transfer_supported"
        if all_ties_zero
        and primary_result["estimate_pp"] >= protocol["decision_rule"]["minimum_g_acc_advantage_pp"]
        and primary_result["lower_95_ci_pp"] > 0
        else "external_transfer_not_demonstrated"
    )
    output = {
        "schema_version": "2026-09-14-v2",
        "scope": "one_shot_zero_shot_naturalbench_external_evaluation",
        "protocol_sha256": sha256_file(args.protocol),
        "dataset_revision": protocol["dataset"]["revision"],
        "dataset_files": records,
        "dataset_audit": dataset_audit,
        "dataset_access": protocol["evaluation"]["dataset_access"],
        "candidate_answer_span_audit": span_audit,
        "official_metric_source": protocol["dataset"]["official_metric_source"],
        "parent_records": {
            "language": release_records(base),
            "vision": vision_records(vision_path),
        },
        "gpu": torch.cuda.get_device_name(0),
        "precision": "bfloat16",
        "arms": arm_results,
        "paired_comparisons": comparisons,
        "candidate_score_ties_zero": all_ties_zero,
        "decision": decision,
        "training_examples": 0,
        "total_wall_seconds": time.monotonic() - total_started,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": decision,
        "metrics": {name: result["official_metrics"] for name, result in arm_results.items()},
        "primary_comparison": {primary: primary_result},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
