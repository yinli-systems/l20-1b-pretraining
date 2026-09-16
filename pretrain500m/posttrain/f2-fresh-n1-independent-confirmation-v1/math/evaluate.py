#!/usr/bin/env python3
"""Score one native 529M checkpoint on the frozen MGSM English proxy."""

from __future__ import annotations

import argparse
import csv
import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
from typing import Optional

import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

from model import Config


NUMBER = re.compile(r"-?(?:\$)?(?:[0-9][0-9,]*)(?:\.[0-9]+)?")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
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


def bootstrap_mean(values: list[float], *, seed: int, resamples: int) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    estimates = [sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(resamples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def number(text: str) -> Optional[str]:
    match = NUMBER.search(text)
    if not match:
        return None
    value = match.group(0).replace("$", "").replace(",", "")
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return None


def load_rows(data_path: Path, split_path: Path, role: str) -> list[dict]:
    raw = data_path.read_bytes()
    lines = raw.splitlines()
    records = list(csv.reader(io.StringIO(raw.decode()), delimiter="\t"))
    split = json.loads(split_path.read_text())
    if split["source_sha256"] != digest(data_path) or len(records) != len(split["rows"]):
        raise ValueError("split/source binding mismatch")
    selected = []
    for ordinal, (assignment, line, record) in enumerate(zip(split["rows"], lines, records)):
        if assignment["ordinal"] != ordinal or hashlib.sha256(line).hexdigest() != assignment["row_sha256"]:
            raise ValueError("split row identity mismatch")
        if len(record) != 2 or number(record[1]) is None:
            raise ValueError("invalid MGSM row")
        if assignment["role"] == role:
            selected.append({"ordinal": ordinal, "row_sha256": assignment["row_sha256"],
                             "question": record[0], "answer": record[1]})
    if len(selected) != split["counts"][role]:
        raise ValueError("split count mismatch")
    return selected


def gold_nll(model: LlamaForCausalLM, tokenizer: Tokenizer, prompt: str, answer: str) -> tuple[float, int]:
    context = tokenizer.encode(prompt).ids
    full = tokenizer.encode(prompt + " " + answer).ids
    if full[:len(context)] != context or len(full) <= len(context):
        raise ValueError("non-additive answer token boundary")
    ids = torch.tensor([full], device="cuda")
    with torch.inference_mode():
        logits = model(ids).logits[0, len(context) - 1:len(full) - 1].float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite gold logits")
    targets = ids[0, len(context):]
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    log_probabilities = selected - torch.logsumexp(logits, dim=-1)
    return float(-log_probabilities.mean().cpu()), len(targets)


def direct_generation(model: LlamaForCausalLM, tokenizer: Tokenizer, prompt: str, max_new_tokens: int,
                      eos_token_id: int) -> tuple[str, int]:
    prompt_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor([prompt_ids], device="cuda")
    generated = []
    newline = tokenizer.encode("\n").ids
    if len(newline) != 1:
        raise ValueError("newline must be one token for the frozen stopping rule")
    with torch.inference_mode():
        output = model(ids, use_cache=True)
        logits = output.logits[:, -1, :]
        past_key_values = output.past_key_values
        for _ in range(max_new_tokens):
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite generation logits")
            token = int(torch.argmax(logits, dim=-1))
            if token in {newline[0], eos_token_id}:
                break
            generated.append(token)
            output = model(torch.tensor([[token]], device="cuda"), past_key_values=past_key_values,
                           use_cache=True)
            logits = output.logits[:, -1, :]
            past_key_values = output.past_key_values
    return tokenizer.decode(generated), len(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--role", choices=["development", "confirmation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    candidates = [item for item in plan["candidates"] if item["id"] == args.candidate_id]
    if len(candidates) != 1:
        raise ValueError("candidate is absent or duplicated")
    candidate = candidates[0]
    if digest(args.data) != plan["data"]["sha256"] or digest(args.splits) != plan["split_manifest_sha256"]:
        raise ValueError("data or split digest mismatch")
    checkpoint_path = Path(candidate["checkpoint"])
    if digest(checkpoint_path) != candidate["checkpoint_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    tokenizer_path = args.tokenizer_dir / "tokenizer.json"
    if digest(tokenizer_path) != plan["tokenizer_json_sha256"]:
        raise ValueError("tokenizer digest mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = load_rows(args.data, args.splits, args.role)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != candidate["expected_step"]:
        raise ValueError("checkpoint step mismatch")
    native_config = Config(hidden_size=1280, intermediate_size=3584, num_hidden_layers=26,
                           num_attention_heads=20, num_key_value_heads=5)
    hf_config = LlamaConfig(**native_config.hf_dict())
    hf_config._attn_implementation = "eager"
    model = LlamaForCausalLM(hf_config)
    loaded = model.load_state_dict(checkpoint["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("checkpoint state mapping mismatch")
    del checkpoint
    torch.use_deterministic_algorithms(True)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()

    results = []
    for row in rows:
        prompt = f"Question: {row['question']}\nAnswer:"
        if len(tokenizer.encode(prompt).ids) + plan["execution"]["max_new_tokens"] > plan["execution"]["max_length"]:
            raise ValueError("prompt exceeds generation limit")
        generated, generated_tokens = direct_generation(model, tokenizer, prompt,
            plan["execution"]["max_new_tokens"], native_config.eos_token_id)
        nll, gold_tokens = gold_nll(model, tokenizer, prompt, row["answer"])
        predicted = number(generated)
        gold = number(row["answer"])
        results.append({"ordinal": row["ordinal"], "row_sha256": row["row_sha256"],
                        "generated_text": generated,
                        "generated_text_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                        "generated_tokens": generated_tokens, "predicted_number": predicted,
                        "gold_number": gold, "exact_numeric_match": predicted == gold,
                        "gold_answer_token_nll": nll, "gold_answer_tokens": gold_tokens})
    del model
    torch.cuda.empty_cache()
    accuracy = [float(row["exact_numeric_match"]) for row in results]
    nlls = [row["gold_answer_token_nll"] for row in results]
    report = {
        "schema": "p529m-independent-math-proxy-mgsm-result-v1",
        "status": "INDEPENDENT_MATH_PROXY_DEVELOPMENT_COMPLETE" if args.role == "development" else "INDEPENDENT_MATH_PROXY_CONFIRMATION_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate": candidate,
        "role": args.role,
        "metrics": {
            "exact_numeric_match": {"score": sum(accuracy) / len(accuracy), "correct": int(sum(accuracy)),
                                    "samples": len(accuracy), "bootstrap_ci95": bootstrap_mean(accuracy, seed=plan["bootstrap"]["seed"], resamples=plan["bootstrap"]["resamples"])},
            "gold_answer_token_nll": {"mean": sum(nlls) / len(nlls), "samples": len(nlls),
                                      "bootstrap_ci95": bootstrap_mean(nlls, seed=plan["bootstrap"]["seed"], resamples=plan["bootstrap"]["resamples"])},
        },
        "primary_metric": "exact_numeric_match",
        "secondary_metric": "gold_answer_token_nll",
        "plan_sha256": digest(args.plan),
        "data_sha256": digest(args.data),
        "split_manifest_sha256": digest(args.splits),
        "tokenizer_json_sha256": digest(tokenizer_path),
        "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "results": results,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
