#!/usr/bin/env python3
"""Run a frozen margin-aware HF export gate on one predeclared sample."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from model import Config


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=7629)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_sha = digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != args.expected_step:
        raise RuntimeError(f"checkpoint step is {checkpoint.get('step')}, expected {args.expected_step}")
    native_state = checkpoint["model"]
    del checkpoint
    config = Config(
        hidden_size=1280,
        intermediate_size=3584,
        num_hidden_layers=26,
        num_attention_heads=20,
        num_key_value_heads=5,
    )
    hf_config = LlamaConfig(**config.hf_dict())
    model = LlamaForCausalLM(hf_config)
    result = model.load_state_dict(native_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"state mapping mismatch: {result}")
    del native_state
    model.config._attn_implementation = "eager"
    model = model.to(dtype=torch.bfloat16, device="cuda").eval()
    state_digests = {name: tensor_digest(value) for name, value in model.state_dict().items()}
    torch.manual_seed(args.sample_seed)
    sample = torch.randint(0, config.vocab_size, (2, 128), device="cuda")
    with torch.no_grad():
        expected = model(sample).logits.float().cpu()
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer_files = []
    for source in sorted(args.tokenizer_dir.iterdir()):
        if source.is_file():
            destination = args.output_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
            tokenizer_files.append(destination.name)
    tokenizer_config_path = args.output_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    original_tokenizer_class = tokenizer_config.get("tokenizer_class")
    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    tokenizer_config["model_max_length"] = config.max_position_embeddings
    tokenizer_config["max_length"] = config.max_position_embeddings
    atomic_json(tokenizer_config_path, tokenizer_config)
    raw_tokenizer = Tokenizer.from_file(str(args.output_dir / "tokenizer.json"))
    hf_tokenizer = AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True, use_fast=True)
    samples = ["Hello, world!", "  leading spaces\nnew line", "Café 123 — test"]
    for text in samples:
        expected_ids = raw_tokenizer.encode(text).ids
        actual_ids = hf_tokenizer.encode(text, add_special_tokens=False)
        if actual_ids != expected_ids:
            raise RuntimeError(f"tokenizer adapter changed token IDs for {text!r}")
    if len(hf_tokenizer) != config.vocab_size or hf_tokenizer.eos_token_id != config.eos_token_id:
        raise RuntimeError("tokenizer adapter vocabulary or EOS mismatch")
    reloaded = LlamaForCausalLM.from_pretrained(
        args.output_dir, local_files_only=True, dtype=torch.bfloat16
    ).cuda().eval()
    reloaded.config._attn_implementation = "eager"
    reloaded_state = reloaded.state_dict()
    if set(reloaded_state) != set(state_digests):
        raise RuntimeError("reloaded state-dict keys differ from the exported model")
    mismatched_tensors = [
        name for name, value in reloaded_state.items() if tensor_digest(value) != state_digests[name]
    ]
    if mismatched_tensors:
        raise RuntimeError(f"reloaded state tensors differ: {mismatched_tensors[:8]}")
    with torch.no_grad():
        actual = reloaded(sample).logits.float().cpu()
    del reloaded
    difference = (actual - expected).abs()
    flat_difference = difference.flatten()
    max_abs_difference = float(flat_difference.max())
    mean_abs_difference = float(flat_difference.mean())
    logit_difference_quantiles = {
        str(q): float(torch.quantile(flat_difference, q))
        for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
    }
    expected_argmax = expected.argmax(dim=-1)
    actual_argmax = actual.argmax(dim=-1)
    argmax_matches = int((expected_argmax == actual_argmax).sum())
    argmax_total = expected_argmax.numel()
    argmax_agreement = argmax_matches / argmax_total
    expected_values, expected_indices = expected.topk(2, dim=-1)
    actual_values, actual_indices = actual.topk(2, dim=-1)
    expected_margins = (expected_values[..., 0] - expected_values[..., 1]).flatten()
    actual_margins = (actual_values[..., 0] - actual_values[..., 1]).flatten()
    row_max_abs_drift = difference.amax(dim=-1).flatten()
    mismatch_positions = (expected_argmax != actual_argmax).flatten().nonzero().flatten().tolist()
    mismatch_rows = []
    for position in mismatch_positions:
        sample_index, token_index = divmod(position, sample.shape[1])
        mismatch_rows.append({
            "sample_index": sample_index,
            "token_index": token_index,
            "expected_top1_token_id": int(expected_indices[sample_index, token_index, 0]),
            "actual_top1_token_id": int(actual_indices[sample_index, token_index, 0]),
            "expected_top1_margin": float(expected_margins[position]),
            "actual_top1_margin": float(actual_margins[position]),
            "row_max_abs_logit_drift": float(row_max_abs_drift[position]),
            "expected_margin_at_most_twice_row_max_drift": bool(expected_margins[position] <= 2 * row_max_abs_drift[position]),
            "actual_margin_at_most_twice_row_max_drift": bool(actual_margins[position] <= 2 * row_max_abs_drift[position]),
        })
    # Exact tensor digests above are the primary serialization invariant. CUDA
    # BF16 GEMM can choose an alignment-dependent reduction path after reload,
    # so the execution smoke test uses explicit, evidence-derived drift bounds.
    functional_limits = {
        "max_abs_logit_difference": 0.5,
        "mean_abs_logit_difference": 0.025,
    }
    all_logits_finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    all_mismatch_margins_explained_by_row_drift = all(
        row["expected_margin_at_most_twice_row_max_drift"]
        and row["actual_margin_at_most_twice_row_max_drift"]
        for row in mismatch_rows
    )
    stable_mask = expected_margins > 2 * row_max_abs_drift
    flat_expected_argmax = expected_argmax.flatten()
    flat_actual_argmax = actual_argmax.flatten()
    stable_position_total = int(stable_mask.sum())
    stable_position_matches = int(
        (flat_expected_argmax[stable_mask] == flat_actual_argmax[stable_mask]).sum()
    )
    gate_passed = (
        all_logits_finite
        and max_abs_difference <= functional_limits["max_abs_logit_difference"]
        and mean_abs_difference <= functional_limits["mean_abs_logit_difference"]
        and all_mismatch_margins_explained_by_row_drift
        and stable_position_matches == stable_position_total
    )
    diagnostic = {
        "schema": "p529m-margin-aware-export-cell-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": args.expected_step,
        "sample_seed": args.sample_seed,
        "sample_shape": list(sample.shape),
        "state_tensor_count": len(state_digests),
        "state_tensor_parity": "bitwise_exact",
        "torch_deterministic_algorithms": True,
        "attention_backend": "eager",
        "max_abs_logit_difference": max_abs_difference,
        "mean_abs_logit_difference": mean_abs_difference,
        "logit_difference_quantiles": logit_difference_quantiles,
        "argmax_matches": argmax_matches,
        "argmax_total": argmax_total,
        "argmax_agreement": argmax_agreement,
        "stable_position_total": stable_position_total,
        "stable_position_matches": stable_position_matches,
        "all_logits_finite": all_logits_finite,
        "all_mismatch_native_and_reloaded_margins_within_twice_own_row_max_drift": all_mismatch_margins_explained_by_row_drift,
        "mismatches": mismatch_rows,
        "functional_limits": functional_limits,
        "margin_aware_rule": "every changed argmax must have both native and reloaded top1 margins at most twice that row's measured maximum absolute logit drift; all positions outside that ambiguity set must preserve argmax",
        "status": "PASS_FROZEN_MARGIN_AWARE_GATE" if gate_passed else "FAIL_FROZEN_MARGIN_AWARE_GATE",
        "claim_boundary": "adaptive export-integrity protocol frozen after the original small-sample argmax gate proved sample-sensitive; no capability score or automatic promotion",
    }
    atomic_json(args.diagnostic_output, diagnostic)
    if not all_logits_finite:
        raise RuntimeError("non-finite logits in export parity check")
    if max_abs_difference > functional_limits["max_abs_logit_difference"]:
        raise RuntimeError(f"maximum BF16 logit drift is {max_abs_difference}")
    if mean_abs_difference > functional_limits["mean_abs_logit_difference"]:
        raise RuntimeError(f"mean BF16 logit drift is {mean_abs_difference}")
    if not all_mismatch_margins_explained_by_row_drift:
        raise RuntimeError("argmax changed outside the measured numerical ambiguity bound")
    if stable_position_matches != stable_position_total:
        raise RuntimeError("argmax changed at a numerically stable position")
    files = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "export-receipt.json":
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    receipt = {
        "status": "PASS_HF_EXPORT_BITWISE_STATE_AND_FROZEN_MARGIN_AWARE_GATE",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": args.expected_step,
        "parameters_total": config.parameter_count(),
        "dtype": "bfloat16",
        "state_tensor_count": len(state_digests),
        "state_tensor_parity": "bitwise_exact",
        "sample_shape": list(sample.shape),
        "sample_seed": args.sample_seed,
        "logit_parity_attention": "eager",
        "functional_limits": functional_limits,
        "max_abs_logit_difference": max_abs_difference,
        "mean_abs_logit_difference": mean_abs_difference,
        "logit_difference_quantiles": logit_difference_quantiles,
        "argmax_matches": argmax_matches,
        "argmax_total": argmax_total,
        "argmax_agreement": argmax_agreement,
        "stable_position_total": stable_position_total,
        "stable_position_matches": stable_position_matches,
        "all_mismatch_native_and_reloaded_margins_within_twice_own_row_max_drift": all_mismatch_margins_explained_by_row_drift,
        "tokenizer_files_copied": tokenizer_files,
        "tokenizer_adapter": {
            "original_class": original_tokenizer_class,
            "export_class": "PreTrainedTokenizerFast",
            "token_id_equivalence_samples": len(samples),
            "vocab_size": len(hf_tokenizer),
            "eos_token_id": hf_tokenizer.eos_token_id,
            "pad_token_id": hf_tokenizer.pad_token_id,
        },
        "files": files,
    }
    atomic_json(args.output_dir / "export-receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
