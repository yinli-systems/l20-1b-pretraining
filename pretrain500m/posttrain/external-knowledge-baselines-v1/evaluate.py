#!/usr/bin/env python3
"""Score a pinned Hugging Face base model on the frozen closed-book SciQ proxy."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def content_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_accuracy(correct: list[bool], *, seed: int, resamples: int) -> list[float]:
    rng = random.Random(seed)
    size = len(correct)
    estimates = [sum(correct[rng.randrange(size)] for _ in range(size)) / size
                 for _ in range(resamples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def load_rows(data_path: Path, split_path: Path) -> list[dict]:
    raw_lines = data_path.read_bytes().splitlines()
    split = json.loads(split_path.read_text())
    if len(raw_lines) != len(split["rows"]) or split["source_sha256"] != digest(data_path):
        raise ValueError("split/source binding mismatch")
    selected = []
    for ordinal, (assignment, line) in enumerate(zip(split["rows"], raw_lines)):
        if assignment["ordinal"] != ordinal or hashlib.sha256(line).hexdigest() != assignment["row_sha256"]:
            raise ValueError("non-canonical split row")
        if assignment.get("role") != "development":
            continue
        row = json.loads(line)
        source_options = [row["correct_answer"], row["distractor1"],
                          row["distractor2"], row["distractor3"]]
        by_hash = {hashlib.sha256(text.encode()).hexdigest(): text for text in source_options}
        if len(by_hash) != 4 or set(by_hash) != set(assignment["choice_text_sha256"]):
            raise ValueError(f"choice binding mismatch at ordinal {ordinal}")
        choices = [by_hash[value] for value in assignment["choice_text_sha256"]]
        gold = choices.index(row["correct_answer"])
        if gold != assignment["gold_position"]:
            raise ValueError(f"gold binding mismatch at ordinal {ordinal}")
        selected.append({
            "ordinal": ordinal,
            "row_sha256": assignment["row_sha256"],
            "question": row["question"],
            "choices": choices,
            "gold": gold,
        })
    if len(selected) != 489 or len(selected) != split["counts"]["development"]:
        raise ValueError("development row count mismatch")
    return selected


def validate_snapshot(snapshot: Path, expected_manifest_sha256: str) -> dict:
    manifest_path = snapshot / "snapshot-manifest.json"
    if digest(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"snapshot manifest digest mismatch: {snapshot}")
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        path = snapshot / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"snapshot file mismatch: {path}")
    return manifest


def encode_choices(tokenizer, rows: list[dict], max_length: int) -> list[dict]:
    encoded = []
    for row in rows:
        context = f"Question: {row['question']}\nAnswer:"
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        choices = []
        for answer in row["choices"]:
            completion = " " + answer
            full_ids = tokenizer.encode(context + completion, add_special_tokens=False)
            if full_ids[:len(context_ids)] != context_ids:
                raise ValueError(f"non-additive token boundary at row {row['ordinal']}")
            target_ids = full_ids[len(context_ids):]
            if not context_ids or not target_ids or len(full_ids) > max_length:
                raise ValueError(f"invalid sequence length at row {row['ordinal']}")
            choices.append({
                "ids": full_ids,
                "target_start": len(context_ids),
                "target_tokens": len(target_ids),
            })
        encoded.append({
            "ordinal": row["ordinal"],
            "row_sha256": row["row_sha256"],
            "gold": row["gold"],
            "choices": choices,
        })
    return encoded


def score(model, encoded: list[dict], *, batch_size: int, pad_id: int) -> list[dict]:
    flat = []
    for row_index, row in enumerate(encoded):
        for choice_index, choice in enumerate(row["choices"]):
            flat.append((row_index, choice_index, choice))
    totals = [[None] * 4 for _ in encoded]
    normalized = [[None] * 4 for _ in encoded]
    flat.sort(key=lambda item: len(item[2]["ids"]))
    with torch.inference_mode():
        for offset in range(0, len(flat), batch_size):
            batch = flat[offset:offset + batch_size]
            length = max(len(item[2]["ids"]) for item in batch)
            input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long, device="cuda")
            attention_mask = torch.zeros_like(input_ids)
            for batch_index, (_, _, choice) in enumerate(batch):
                ids = torch.tensor(choice["ids"], dtype=torch.long, device="cuda")
                input_ids[batch_index, :len(ids)] = ids
                attention_mask[batch_index, :len(ids)] = 1
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite model logits")
            for batch_index, (row_index, choice_index, choice) in enumerate(batch):
                start = choice["target_start"]
                stop = len(choice["ids"])
                prediction_logits = logits[batch_index, start - 1:stop - 1].float()
                targets = input_ids[batch_index, start:stop]
                selected = prediction_logits.gather(1, targets[:, None]).squeeze(1)
                token_log_probabilities = selected - torch.logsumexp(prediction_logits, dim=-1)
                total = float(token_log_probabilities.sum().cpu())
                totals[row_index][choice_index] = total
                normalized[row_index][choice_index] = total / choice["target_tokens"]
            del logits, input_ids, attention_mask
    results = []
    for row, raw, norm in zip(encoded, totals, normalized):
        if any(value is None for value in raw + norm):
            raise RuntimeError("missing choice score")
        prediction = max(range(4), key=lambda index: (norm[index], -index))
        raw_prediction = max(range(4), key=lambda index: (raw[index], -index))
        results.append({
            "ordinal": row["ordinal"],
            "row_sha256": row["row_sha256"],
            "gold": row["gold"],
            "prediction_acc_norm": prediction,
            "prediction_acc": raw_prediction,
            "correct_acc_norm": prediction == row["gold"],
            "correct_acc": raw_prediction == row["gold"],
            "choice_loglikelihood": raw,
            "choice_loglikelihood_per_token": norm,
        })
    return sorted(results, key=lambda item: item["ordinal"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    candidates = [item for item in plan["candidates"] if item["id"] == args.candidate_id]
    if len(candidates) != 1:
        raise ValueError("candidate is absent or duplicated")
    candidate = candidates[0]
    if digest(args.data) != plan["data_sha256"] or digest(args.splits) != plan["split_manifest_sha256"]:
        raise ValueError("frozen evaluation input digest mismatch")

    model_path = args.model_root / candidate["snapshot"]
    model_manifest = validate_snapshot(model_path, candidate["snapshot_manifest_sha256"])
    tokenizer_candidate = candidate.get("tokenizer_snapshot", candidate["snapshot"])
    tokenizer_path = args.model_root / tokenizer_candidate
    tokenizer_manifest = validate_snapshot(tokenizer_path, candidate.get(
        "tokenizer_snapshot_manifest_sha256", candidate["snapshot_manifest_sha256"]))

    rows = load_rows(args.data, args.splits)
    if candidate.get("tokenizer_loader") == "tokenizer_json":
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path / "tokenizer.json"),
            bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=True, use_fast=True)
    encoded = encode_choices(tokenizer, rows, plan["execution"]["max_length"])
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan["execution"]["seed"])
    torch.cuda.manual_seed_all(plan["execution"]["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda").eval()
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError(
            f"model/tokenizer vocabulary mismatch: {model.get_input_embeddings().num_embeddings} != {len(tokenizer)}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    results = score(model, encoded, batch_size=plan["execution"]["batch_size_per_gpu"],
                    pad_id=pad_id)
    model_class = type(model).__name__
    config_class = type(model.config).__name__
    del model
    torch.cuda.empty_cache()

    metrics = {}
    for metric in ["acc_norm", "acc"]:
        correct = [item[f"correct_{metric}"] for item in results]
        metrics[metric] = {
            "score": sum(correct) / len(correct),
            "correct": sum(correct),
            "samples": len(correct),
            "bootstrap_ci95": bootstrap_accuracy(
                correct, seed=plan["bootstrap"]["seed"],
                resamples=plan["bootstrap"]["resamples"]),
        }
    report = {
        "schema": "p529m-external-closed-book-sciq-result-v1",
        "status": "EXTERNAL_CLOSED_BOOK_SCIQ_DEVELOPMENT_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate": candidate,
        "model_manifest": model_manifest,
        "tokenizer_manifest": tokenizer_manifest,
        "model_class": model_class,
        "config_class": config_class,
        "parameters": parameters,
        "role": "development",
        "metrics": metrics,
        "primary_metric": "acc_norm",
        "plan_sha256": digest(args.plan),
        "data_sha256": digest(args.data),
        "split_manifest_sha256": digest(args.splits),
        "result_rows_sha256": content_digest(results),
        "results": results,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
