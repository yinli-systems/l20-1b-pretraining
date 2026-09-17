#!/usr/bin/env python3
"""Falsifiable forward/dX/dW benchmark for consumer-GPU MoE backends."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch

from cvcr_moe.config import ModelConfig
from cvcr_moe.model import ExpertBank


SCATTERMOE_REVISION = "47b5e1502e5a10e82c8e5945d761b877849871e7"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--tokens", type=int, nargs="+", default=[16_384, 32_768])
    result.add_argument(
        "--distributions",
        nargs="+",
        choices=("balanced", "skewed", "empty"),
        default=["balanced", "skewed", "empty"],
    )
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--steps", type=int, default=50)
    result.add_argument("--trace-dir", type=Path)
    return result


def assignments(
    tokens: int, top_k: int, experts: int, distribution: str, device: torch.device
) -> torch.Tensor:
    rows = torch.arange(tokens, device=device)
    if distribution == "balanced":
        return torch.stack(tuple((rows + slot) % experts for slot in range(top_k)), 1)
    if distribution == "skewed":
        # Keep all experts represented but send most assignments to experts 0/1.
        ids = torch.stack(tuple((rows + slot) % experts for slot in range(top_k)), 1)
        mask = (rows % 10) != 0
        ids[mask, 0] = 0
        if top_k > 1:
            ids[mask, 1] = 1
        return ids
    # Exercise zero-length segments: only the first half of the bank receives work.
    active = max(top_k, experts // 2)
    return torch.stack(tuple((rows + slot) % active for slot in range(top_k)), 1)


def new_bank(config: ModelConfig, backend: str, state: dict[str, torch.Tensor]) -> ExpertBank:
    bank = ExpertBank(replace(config, expert_backend=backend)).cuda().train()
    bank.load_state_dict(state)
    return bank


def step(
    bank: ExpertBank,
    hidden_seed: torch.Tensor,
    expert_ids: torch.Tensor,
    output_seed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bank.zero_grad(set_to_none=True)
    hidden = hidden_seed.detach().clone().requires_grad_(True)
    output = bank.forward_topk(hidden, expert_ids)
    loss = (output.float() * output_seed).sum() / output.numel()
    loss.backward()
    return (
        output.detach(),
        hidden.grad.detach(),
        bank.gate_up.grad.detach(),
        bank.down.grad.detach(),
    )


def errors(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (candidate.float() - reference.float()).abs()
    denominator = reference.float().abs().clamp_min(1e-6)
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "max_rel_at_reference_ge_1e-6": (delta / denominator).max().item(),
    }


def timed(
    bank: ExpertBank,
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    output_seed: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        step(bank, hidden, expert_ids, output_seed)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        step(bank, hidden, expert_ids, output_seed)
        ended.record()
        ended.synchronize()
        samples.append(started.elapsed_time(ended))
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "iterations": iterations,
    }


def trace(
    path: Path,
    bank: ExpertBank,
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    output_seed: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        step(bank, hidden, expert_ids, output_seed)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(path))


def main() -> None:
    arguments = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA allocation is required")
    torch.manual_seed(20260916)
    device = torch.device("cuda")
    config = ModelConfig.from_json(arguments.config)
    reference = ExpertBank(replace(config, expert_backend="grouped_mm")).cuda().train()
    state = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    header = {
        "kind": "expert_backend_benchmark_environment",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "scattermoe_revision_required": SCATTERMOE_REVISION,
        "config": str(arguments.config),
        "timestamp_unix": time.time(),
    }
    print(json.dumps(header, sort_keys=True), flush=True)

    candidate = new_bank(config, "scattermoe", state)
    for token_count in arguments.tokens:
        hidden = torch.randn(token_count, config.hidden_size, device=device)
        for distribution in arguments.distributions:
            expert_ids = assignments(
                token_count,
                config.experts_per_token,
                config.num_experts,
                distribution,
                device,
            )
            output_seed = torch.randn(
                token_count * config.experts_per_token,
                config.hidden_size,
                device=device,
            )
            reference_values = step(reference, hidden, expert_ids, output_seed)
            candidate_values = step(candidate, hidden, expert_ids, output_seed)
            reference_timing = timed(
                reference,
                hidden,
                expert_ids,
                output_seed,
                arguments.warmup,
                arguments.steps,
            )
            candidate_timing = timed(
                candidate,
                hidden,
                expert_ids,
                output_seed,
                arguments.warmup,
                arguments.steps,
            )
            record = {
                "kind": "expert_backend_benchmark",
                "tokens": token_count,
                "assignments": token_count * config.experts_per_token,
                "distribution": distribution,
                "reference": "torch_grouped_mm",
                "candidate": "scattermoe",
                "reference_timing": reference_timing,
                "candidate_timing": candidate_timing,
                "speedup": (
                    reference_timing["median_ms"] / candidate_timing["median_ms"]
                ),
                "output_error": errors(candidate_values[0], reference_values[0]),
                "input_gradient_error": errors(candidate_values[1], reference_values[1]),
                "gate_up_gradient_error": errors(
                    candidate_values[2], reference_values[2]
                ),
                "down_gradient_error": errors(candidate_values[3], reference_values[3]),
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            if arguments.trace_dir is not None and token_count == arguments.tokens[0]:
                trace(
                    arguments.trace_dir / f"native-{distribution}.json",
                    reference,
                    hidden,
                    expert_ids,
                    output_seed,
                )
                trace(
                    arguments.trace_dir / f"scattermoe-{distribution}.json",
                    candidate,
                    hidden,
                    expert_ids,
                    output_seed,
                )


if __name__ == "__main__":
    main()
