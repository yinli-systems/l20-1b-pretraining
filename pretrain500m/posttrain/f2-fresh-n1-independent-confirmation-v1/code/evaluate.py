#!/usr/bin/env python3
"""Score one native 529M checkpoint on the frozen MBPP code proxy."""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import tempfile

import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

from model import Config


ALLOWED_MODULES = {
    "array", "bisect", "cmath", "collections", "copy", "datetime", "decimal",
    "fractions", "functools", "heapq", "itertools", "math", "operator",
    "re", "statistics", "string",
}
BLOCKED_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "exit", "globals",
    "help", "input", "locals", "open", "quit", "vars",
}
PROMPT_TEMPLATE = "# Task: {prompt}\n# Write only executable Python code.\n# Solution:\n"


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
    estimates = [sum(values[rng.randrange(size)] for _ in range(size)) / size
                 for _ in range(resamples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def load_rows(rows_path: Path, split_path: Path, role: str) -> list[dict]:
    records = [json.loads(line) for line in rows_path.read_text().splitlines()]
    split = json.loads(split_path.read_text())
    if split["rows_jsonl_sha256"] != digest(rows_path) or len(records) != len(split["rows"]):
        raise ValueError("split/row binding mismatch")
    selected = []
    for ordinal, (assignment, record) in enumerate(zip(split["rows"], records)):
        canonical = json.dumps(record, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode()
        if assignment["ordinal"] != ordinal or assignment["task_id"] != record["task_id"]:
            raise ValueError("split row identity mismatch")
        if hashlib.sha256(canonical).hexdigest() != assignment["row_sha256"]:
            raise ValueError("split row digest mismatch")
        if assignment["role"] == role:
            record = dict(record)
            record["ordinal"] = ordinal
            record["row_sha256"] = assignment["row_sha256"]
            selected.append(record)
    if len(selected) != split["counts"][role]:
        raise ValueError("split count mismatch")
    return selected


def gold_nll(model: LlamaForCausalLM, tokenizer: Tokenizer, prompt: str,
             code: str) -> tuple[float, int]:
    context = tokenizer.encode(prompt).ids
    full = tokenizer.encode(prompt + code).ids
    if full[:len(context)] != context or len(full) <= len(context):
        raise ValueError("non-additive code token boundary")
    ids = torch.tensor([full], device="cuda")
    with torch.inference_mode():
        logits = model(ids).logits[0, len(context) - 1:len(full) - 1].float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite gold logits")
    targets = ids[0, len(context):]
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    log_probabilities = selected - torch.logsumexp(logits, dim=-1)
    return float(-log_probabilities.mean().cpu()), len(targets)


def generate(model: LlamaForCausalLM, tokenizer: Tokenizer, prompt: str,
             max_new_tokens: int, eos_token_id: int) -> tuple[str, int]:
    prompt_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor([prompt_ids], device="cuda")
    generated: list[int] = []
    with torch.inference_mode():
        output = model(ids, use_cache=True)
        logits = output.logits[:, -1, :]
        past_key_values = output.past_key_values
        for _ in range(max_new_tokens):
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite generation logits")
            token = int(torch.argmax(logits, dim=-1))
            if token == eos_token_id:
                break
            generated.append(token)
            output = model(torch.tensor([[token]], device="cuda"),
                           past_key_values=past_key_values, use_cache=True)
            logits = output.logits[:, -1, :]
            past_key_values = output.past_key_values
    return tokenizer.decode(generated), len(generated)


def extract_code(text: str) -> str:
    if "```" not in text:
        return text.strip()
    chunks = text.split("```")
    if len(chunks) < 3:
        return text.strip()
    body = chunks[1]
    if body.startswith("python\n"):
        body = body[len("python\n"):]
    elif body.startswith("py\n"):
        body = body[len("py\n"):]
    return body.strip()


def validate_ast(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError) as error:
        return False, f"syntax:{type(error).__name__}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in ALLOWED_MODULES for alias in node.names):
                return False, "policy:blocked_import"
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_MODULES:
                return False, "policy:blocked_import"
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return False, "policy:blocked_name"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "policy:dunder_attribute"
    return True, "pass"


def limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024**2, 384 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def execute(code: str, test_imports: list[str], tests: list[str]) -> tuple[bool, str]:
    valid, reason = validate_ast(code)
    if not valid:
        return False, reason
    source = code + "\n" + "\n".join(test_imports) + "\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory(prefix="p529m-code-proxy-") as directory:
        script = Path(directory) / "candidate.py"
        script.write_text(source)
        environment = {
            "HOME": directory,
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
        }
        try:
            completed = subprocess.run(
                ["unshare", "-Urn", "--", "/usr/bin/python3", "-I", "-S", str(script)],
                cwd=directory, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                timeout=3, preexec_fn=limits, check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "runtime:timeout"
        except OSError as error:
            return False, f"runtime:{type(error).__name__}"
        if completed.returncode == 0:
            return True, "pass"
        last = completed.stderr.strip().splitlines()
        return False, "runtime:" + (last[-1][:160] if last else f"exit_{completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--rows", type=Path, required=True)
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
    if digest(args.rows) != plan["data"]["rows_jsonl_sha256"]:
        raise ValueError("row digest mismatch")
    if digest(args.splits) != plan["split_manifest_sha256"]:
        raise ValueError("split digest mismatch")
    checkpoint_path = Path(candidate["checkpoint"])
    if digest(checkpoint_path) != candidate["checkpoint_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    tokenizer_path = args.tokenizer_dir / "tokenizer.json"
    if digest(tokenizer_path) != plan["tokenizer_json_sha256"]:
        raise ValueError("tokenizer digest mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = load_rows(args.rows, args.splits, args.role)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != candidate["expected_step"]:
        raise ValueError("checkpoint step mismatch")
    native_config = Config(hidden_size=1280, intermediate_size=3584,
                           num_hidden_layers=26, num_attention_heads=20,
                           num_key_value_heads=5)
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
        prompt = PROMPT_TEMPLATE.format(prompt=row["prompt"])
        if len(tokenizer.encode(prompt).ids) + plan["execution"]["max_new_tokens"] > plan["execution"]["max_length"]:
            raise ValueError("prompt exceeds generation limit")
        generated, generated_tokens = generate(
            model, tokenizer, prompt, plan["execution"]["max_new_tokens"],
            native_config.eos_token_id,
        )
        code = extract_code(generated)
        ast_ok, ast_reason = validate_ast(code)
        passed, execution_reason = execute(code, row["test_imports"], row["test_list"])
        nll, gold_tokens = gold_nll(model, tokenizer, prompt, row["code"])
        results.append({
            "ordinal": row["ordinal"],
            "task_id": row["task_id"],
            "row_sha256": row["row_sha256"],
            "generated_text": generated,
            "generated_text_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "extracted_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "generated_tokens": generated_tokens,
            "syntax_and_policy_pass": ast_ok,
            "syntax_and_policy_reason": ast_reason,
            "execution_pass": passed,
            "execution_reason": execution_reason,
            "gold_code_token_nll": nll,
            "gold_code_tokens": gold_tokens,
        })
    del model
    torch.cuda.empty_cache()
    execution = [float(row["execution_pass"]) for row in results]
    syntax = [float(row["syntax_and_policy_pass"]) for row in results]
    nlls = [row["gold_code_token_nll"] for row in results]
    report = {
        "schema": "p529m-independent-code-proxy-mbpp-result-v1",
        "status": "INDEPENDENT_CODE_PROXY_DEVELOPMENT_COMPLETE" if args.role == "development" else "INDEPENDENT_CODE_PROXY_CONFIRMATION_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate": candidate,
        "role": args.role,
        "metrics": {
            "execution_pass_at_1": {
                "score": sum(execution) / len(execution), "correct": int(sum(execution)),
                "samples": len(execution), "bootstrap_ci95": bootstrap_mean(
                    execution, seed=plan["bootstrap"]["seed"],
                    resamples=plan["bootstrap"]["resamples"]),
            },
            "syntax_and_policy_rate": {
                "score": sum(syntax) / len(syntax), "correct": int(sum(syntax)),
                "samples": len(syntax), "bootstrap_ci95": bootstrap_mean(
                    syntax, seed=plan["bootstrap"]["seed"],
                    resamples=plan["bootstrap"]["resamples"]),
            },
            "gold_code_token_nll": {
                "mean": sum(nlls) / len(nlls), "samples": len(nlls),
                "bootstrap_ci95": bootstrap_mean(
                    nlls, seed=plan["bootstrap"]["seed"],
                    resamples=plan["bootstrap"]["resamples"]),
            },
        },
        "primary_metric": "execution_pass_at_1",
        "secondary_metrics": ["syntax_and_policy_rate", "gold_code_token_nll"],
        "plan_sha256": digest(args.plan),
        "rows_jsonl_sha256": digest(args.rows),
        "split_manifest_sha256": digest(args.splits),
        "tokenizer_json_sha256": digest(tokenizer_path),
        "results_sha256": hashlib.sha256(json.dumps(
            results, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "results": results,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
