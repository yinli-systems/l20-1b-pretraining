#!/usr/bin/env python3
"""Compare one complete native/scattered optimizer update on frozen records."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import torch

from cvcr_moe.config import ModelConfig
from cvcr_moe.data import PackedReader, verify_frozen_manifest
from cvcr_moe.model import MoELanguageModel


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--candidate-config", type=Path)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--microbatch", type=int, default=2)
    result.add_argument("--accumulation", type=int, default=2)
    result.add_argument("--seed", type=int, default=20260916)
    result.add_argument(
        "--candidate-backend", choices=("grouped_mm", "scattermoe"), default="scattermoe"
    )
    result.add_argument(
        "--reference-backend", choices=("grouped_mm", "scattermoe"), default="grouped_mm"
    )
    result.add_argument("--reference-microbatch", type=int)
    result.add_argument("--reference-accumulation", type=int)
    result.add_argument("--candidate-microbatch", type=int)
    result.add_argument("--candidate-accumulation", type=int)
    result.add_argument("--compile", action="store_true")
    return result


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def tensor_errors(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    left = candidate.detach().float().cpu()
    right = reference.float()
    delta = (left - right).abs()
    reference_norm = torch.linalg.vector_norm(right).item()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "relative_l2": torch.linalg.vector_norm(delta).item()
        / max(reference_norm, 1e-30),
    }


def update(
    config: ModelConfig,
    backend: str,
    reader: PackedReader,
    microbatch: int,
    accumulation: int,
    seed: int,
    validation_microbatch: int,
    compile_model: bool,
    reference_gradients: dict[str, torch.Tensor] | None = None,
    reference_parameters: dict[str, torch.Tensor] | None = None,
) -> tuple[dict, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    seed_all(seed)
    raw_model = MoELanguageModel(replace(config, expert_backend=backend)).cuda().train()
    model = torch.compile(raw_model, mode="default") if compile_model else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for microstep in range(accumulation):
        inputs, targets = reader.batch_for_step(microstep, 0, 1, microbatch)
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, cross_entropy, router, predictor = model(inputs, targets)
        (loss / accumulation).backward()
        losses.append(
            {
                "loss": loss.item(),
                "cross_entropy": cross_entropy.item(),
                "router_auxiliary": router.item(),
                "predictor_loss": predictor.item(),
            }
        )
    torch.cuda.synchronize()

    gradient_errors = {}
    if reference_gradients is None:
        saved_gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in raw_model.named_parameters()
        }
    else:
        saved_gradients = None
        for name, parameter in raw_model.named_parameters():
            gradient_errors[name] = tensor_errors(
                parameter.grad, reference_gradients[name]
            )

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        raw_model.parameters(), 1.0, error_if_nonfinite=True
    ).item()
    optimizer.step()
    torch.cuda.synchronize()

    parameter_errors = {}
    if reference_parameters is None:
        saved_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in raw_model.named_parameters()
        }
    else:
        saved_parameters = None
        for name, parameter in raw_model.named_parameters():
            parameter_errors[name] = tensor_errors(
                parameter, reference_parameters[name]
            )

    consumed_sequences = microbatch * accumulation
    if consumed_sequences % validation_microbatch:
        raise ValueError("consumed sequences must divide the validation microbatch")
    inputs, targets = reader.batch_for_step(
        consumed_sequences // validation_microbatch,
        0,
        1,
        validation_microbatch,
    )
    raw_model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, validation_ce, _, _ = model(inputs.cuda(), targets.cuda())
    report = {
        "backend": backend,
        "compiled": compile_model,
        "microbatch_losses": losses,
        "preclip_gradient_norm": gradient_norm,
        "post_update_validation_cross_entropy": validation_ce.item(),
    }
    if gradient_errors:
        report["gradient_error_maxima"] = {
            metric: max(value[metric] for value in gradient_errors.values())
            for metric in ("max_abs", "mean_abs", "relative_l2")
        }
        report["largest_gradient_relative_l2"] = sorted(
            (
                {"name": name, **value}
                for name, value in gradient_errors.items()
            ),
            key=lambda value: value["relative_l2"],
            reverse=True,
        )[:20]
        report["parameter_error_maxima"] = {
            metric: max(value[metric] for value in parameter_errors.values())
            for metric in ("max_abs", "mean_abs", "relative_l2")
        }
        report["largest_parameter_relative_l2"] = sorted(
            (
                {"name": name, **value}
                for name, value in parameter_errors.items()
            ),
            key=lambda value: value["relative_l2"],
            reverse=True,
        )[:20]
    del model, raw_model, optimizer
    torch.cuda.empty_cache()
    return report, saved_gradients, saved_parameters


def main() -> None:
    arguments = parser().parse_args()
    verify_frozen_manifest(arguments.data_dir)
    config = ModelConfig.from_json(arguments.config)
    candidate_config = ModelConfig.from_json(
        arguments.candidate_config or arguments.config
    )
    reference_microbatch = arguments.reference_microbatch or arguments.microbatch
    reference_accumulation = (
        arguments.reference_accumulation or arguments.accumulation
    )
    candidate_microbatch = arguments.candidate_microbatch or arguments.microbatch
    candidate_accumulation = (
        arguments.candidate_accumulation or arguments.accumulation
    )
    if reference_microbatch * reference_accumulation != (
        candidate_microbatch * candidate_accumulation
    ):
        raise ValueError("reference and candidate must consume the same sequences/update")
    validation_microbatch = min(reference_microbatch, candidate_microbatch)
    reader = PackedReader(arguments.data_dir, arguments.seed, repeat=True)
    reference, gradients, parameters = update(
        config,
        arguments.reference_backend,
        reader,
        reference_microbatch,
        reference_accumulation,
        arguments.seed,
        validation_microbatch,
        arguments.compile,
    )
    assert gradients is not None and parameters is not None
    candidate, _, _ = update(
        candidate_config,
        arguments.candidate_backend,
        reader,
        candidate_microbatch,
        candidate_accumulation,
        arguments.seed,
        validation_microbatch,
        arguments.compile,
        gradients,
        parameters,
    )
    print(
        json.dumps(
            {
                "kind": "full_model_one_update_backend_parity",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "config": str(arguments.config),
                "reference_microbatch": reference_microbatch,
                "reference_accumulation": reference_accumulation,
                "candidate_microbatch": candidate_microbatch,
                "candidate_accumulation": candidate_accumulation,
                "reference_backend": arguments.reference_backend,
                "candidate_backend": arguments.candidate_backend,
                "compiled": arguments.compile,
                "reference": reference,
                "candidate": candidate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
