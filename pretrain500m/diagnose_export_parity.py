#!/usr/bin/env python3
"""Measure BF16 CUDA logit drift across the HF save/reload boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from model import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    config = Config(
        hidden_size=1280,
        intermediate_size=3584,
        num_hidden_layers=26,
        num_attention_heads=20,
        num_key_value_heads=5,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    native = LlamaForCausalLM(LlamaConfig(**config.hf_dict()))
    result = native.load_state_dict(checkpoint["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    native.config._attn_implementation = "eager"
    native = native.to(dtype=torch.bfloat16, device="cuda").eval()

    torch.manual_seed(20260912)
    sample = torch.randint(0, config.vocab_size, (2, 128), device="cuda")
    with torch.no_grad():
        expected = native(sample).logits.float().cpu()
    del native, checkpoint
    torch.cuda.empty_cache()

    reloaded = LlamaForCausalLM.from_pretrained(
        args.hf_dir, local_files_only=True, dtype=torch.bfloat16
    ).cuda().eval()
    reloaded.config._attn_implementation = "eager"
    with torch.no_grad():
        actual = reloaded(sample).logits.float().cpu()

    difference = (actual - expected).abs()
    flat = difference.flatten()
    expected_argmax = expected.argmax(dim=-1)
    actual_argmax = actual.argmax(dim=-1)
    thresholds = (0.03125, 0.0625, 0.125, 0.25, 0.5)
    report = {
        "finite": bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()),
        "elements": flat.numel(),
        "max_abs_difference": float(flat.max()),
        "mean_abs_difference": float(flat.mean()),
        "quantiles": {
            str(q): float(torch.quantile(flat, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
        },
        "counts_above": {str(t): int((flat > t).sum()) for t in thresholds},
        "argmax_matches": int((expected_argmax == actual_argmax).sum()),
        "argmax_total": expected_argmax.numel(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
