"""Allocation-bound CUDA correctness and speed preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from cvcr_moe.config import ModelConfig
from cvcr_moe.model import MoELanguageModel


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def grouped_mm_check() -> dict:
    torch.manual_seed(1)
    hidden = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(
        3, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    offsets = torch.tensor([17, 53, 96], device="cuda", dtype=torch.int32)
    output = F.grouped_mm(hidden, weights, offs=offsets)
    loss = output.float().square().mean()
    loss.backward()
    return {
        "shape": list(output.shape),
        "loss": loss.item(),
        "hidden_grad_finite": bool(torch.isfinite(hidden.grad).all()),
        "weight_grad_finite": bool(torch.isfinite(weights.grad).all()),
    }


def model_check(method: str, measure_steps: int) -> dict:
    config = ModelConfig(
        vocab_size=50_280,
        hidden_size=256,
        expert_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=16,
        experts_per_token=2,
        max_position_embeddings=128,
        method=method,
        expert_backend="grouped_mm",
        probe_fraction=0.0625,
    )
    model = MoELanguageModel(config).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    inputs = torch.randint(0, config.vocab_size, (4, 128), device="cuda")
    targets = torch.randint(0, config.vocab_size, (4, 128), device="cuda")
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, *_ = model(inputs, targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(measure_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, *_ = model(inputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = measure_steps * inputs.numel()
    return {
        "method": method,
        "loss_first": losses[0].item(),
        "loss_last": losses[-1].item(),
        "losses_finite": bool(torch.isfinite(torch.stack(losses)).all()),
        "tokens_per_second": tokens / elapsed,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "deployed_parameters": config.deployed_parameter_count(),
        "active_parameters": config.active_parameter_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure-steps", type=int, default=10)
    arguments = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    result = {
        "status": "CUDA_PREFLIGHT_ONLY",
        "claim_boundary": (
            "This receipt proves kernel/runtime correctness on one allocated GPU. "
            "It is not a proxy-training result or 7B fit result."
        ),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_total_memory": properties.total_memory,
        "grouped_mm": grouped_mm_check(),
        "models": [
            model_check("top2", arguments.measure_steps),
            model_check("cvcr", arguments.measure_steps),
        ],
    }
    if not result["grouped_mm"]["hidden_grad_finite"]:
        raise RuntimeError("grouped_mm hidden gradient is non-finite")
    if not result["grouped_mm"]["weight_grad_finite"]:
        raise RuntimeError("grouped_mm weight gradient is non-finite")
    if not all(item["losses_finite"] for item in result["models"]):
        raise RuntimeError("model preflight produced a non-finite loss")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(result, arguments.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
