#!/usr/bin/env python3
"""Canonical Hugging Face configuration for the from-scratch 1.1B model."""

from __future__ import annotations

import json
from pathlib import Path


HF_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "eos_token_id": 0,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 5632,
    "max_position_embeddings": 2048,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 22,
    "num_key_value_heads": 4,
    "pad_token_id": 1,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 32000,
}


def write_hf_config(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "config.json"
    destination.write_text(json.dumps(HF_CONFIG, indent=2, sort_keys=True) + "\n")
    return destination

