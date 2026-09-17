#!/usr/bin/env python3
"""Exhaustive 16-expert local-credit diagnostics on held-out packed blocks.

This measures the first-order local-credit oracle from the research protocol.
It does not claim exact downstream counterfactual loss or an unbiased discrete
Top-k language-model gradient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from cvcr_moe.config import ModelConfig
from cvcr_moe.data import PackedReader, verify_frozen_manifest
from cvcr_moe.model import MoELanguageModel


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def rank_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_rank = left.argsort(dim=-1).argsort(dim=-1).float()
    right_rank = right.argsort(dim=-1).argsort(dim=-1).float()
    left_rank -= left_rank.mean(dim=-1, keepdim=True)
    right_rank -= right_rank.mean(dim=-1, keepdim=True)
    numerator = (left_rank * right_rank).sum(dim=-1)
    denominator = torch.sqrt(
        left_rank.square().sum(dim=-1) * right_rank.square().sum(dim=-1)
    )
    return numerator / denominator.clamp_min(1e-12)


def topk_recall(predicted: torch.Tensor, exact: torch.Tensor, k: int) -> torch.Tensor:
    predicted_ids = predicted.topk(k, dim=-1).indices
    exact_ids = exact.topk(k, dim=-1).indices
    overlap = (
        predicted_ids[:, :, None] == exact_ids[:, None, :]
    ).any(dim=-1).float().sum(dim=-1)
    return overlap / k


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batches", type=int, default=2)
    result.add_argument("--oracle-tokens", type=int, default=256)
    result.add_argument("--seed", type=int, default=20260916)
    return result


def main() -> None:
    arguments = parser().parse_args()
    torch.manual_seed(arguments.seed)
    torch.cuda.set_device(0)
    data_manifest = verify_frozen_manifest(arguments.data_dir)
    config = ModelConfig.from_json(arguments.config)
    if not config.cvcr_enabled:
        raise ValueError("oracle diagnostics require a trained CVCR predictor")
    model = MoELanguageModel(config).cuda().eval()
    state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    checkpoint_step = int(state["step"])
    del state
    model.requires_grad_(False)
    reader = PackedReader(arguments.data_dir, arguments.seed)
    generator = torch.Generator(device="cuda").manual_seed(arguments.seed + 1)
    records: list[dict[str, float | int]] = []

    for batch_index in range(arguments.batches):
        inputs, targets = reader.batch_for_step(batch_index, 0, 1, 1)
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        for layer_index, block in enumerate(model.layers):
            captured: dict[str, torch.Tensor] = {}

            def capture_input(_module, values):
                captured["hidden"] = values[0].detach()

            def cut_graph(_module, _values, output):
                leaf = output[0].detach().requires_grad_(True)
                captured["mixed"] = leaf
                return leaf, output[1], output[2]

            pre_handle = block.moe.register_forward_pre_hook(capture_input)
            post_handle = block.moe.register_forward_hook(cut_graph)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(inputs, targets)[0]
            loss.backward()
            pre_handle.remove()
            post_handle.remove()

            hidden = captured["hidden"].reshape(-1, config.hidden_size)
            reference = captured["mixed"].detach().reshape(
                -1, config.hidden_size
            )
            gradient = captured["mixed"].grad.detach().reshape(
                -1, config.hidden_size
            )
            sample_count = min(arguments.oracle_tokens, hidden.shape[0])
            rows = torch.randperm(
                hidden.shape[0], generator=generator, device="cuda"
            )[:sample_count]
            hidden_sample = hidden.index_select(0, rows)
            reference_sample = reference.index_select(0, rows).float()
            gradient_sample = gradient.index_select(0, rows).float()

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = block.moe.router(hidden.float())
                selected_ids_all = logits.topk(
                    config.experts_per_token, dim=-1
                ).indices
                selected_ids = selected_ids_all.index_select(0, rows)
                expert_ids = torch.arange(
                    config.num_experts, device="cuda", dtype=torch.long
                ).repeat(sample_count)
                repeated_hidden = hidden_sample[:, None, :].expand(
                    -1, config.num_experts, -1
                ).reshape(-1, config.hidden_size)
                exact_outputs = block.moe.experts.forward_assignments(
                    repeated_hidden, expert_ids, detach_parameters=True
                ).view(sample_count, config.num_experts, config.hidden_size)
                predictor = block.moe.credit_predictor
                assert predictor is not None
                projected = predictor.projected_hidden(hidden_sample)
                latent = (
                    projected[:, None, :] * predictor.expert_scale[None, :, :]
                )
                predicted_outputs = predictor.default[None, :, :] + torch.matmul(
                    latent, predictor.output_projection
                )
                default_outputs = predictor.default[None, :, :].expand_as(
                    predicted_outputs
                )

            exact_delta = exact_outputs.float() - reference_sample[:, None, :]
            predicted_delta = (
                predicted_outputs.float() - reference_sample[:, None, :]
            )
            default_delta = default_outputs.float() - reference_sample[:, None, :]
            exact_credit = -torch.sum(
                gradient_sample[:, None, :] * exact_delta, dim=-1
            )
            predicted_credit = -torch.sum(
                gradient_sample[:, None, :] * predicted_delta, dim=-1
            )
            default_credit = -torch.sum(
                gradient_sample[:, None, :] * default_delta, dim=-1
            )
            inactive = torch.ones_like(exact_credit, dtype=torch.bool)
            inactive.scatter_(1, selected_ids, False)

            selected_counts = torch.zeros(
                config.num_experts, device="cuda", dtype=torch.float32
            )
            selected_counts.scatter_add_(
                0,
                selected_ids_all.reshape(-1),
                torch.ones_like(selected_ids_all, dtype=torch.float32).reshape(-1),
            )
            inactive_counts = hidden.shape[0] - selected_counts
            eligible = inactive_counts > 0
            eligible_count = eligible.sum()
            temporal_probability = 1.0 / config.cvcr_update_interval
            effective_probe_fraction = (
                config.probe_fraction * config.cvcr_update_interval
            )
            probe_count = round(hidden.shape[0] * effective_probe_fraction)
            inclusion = torch.ones_like(inactive_counts)
            inclusion[eligible] = (
                1.0
                - torch.pow(
                    1.0 - inactive_counts[eligible].reciprocal(), probe_count
                )
            ) / eligible_count
            variance_multiplier = inclusion.reciprocal() - 1.0

            exact_inactive = exact_credit[inactive]
            predicted_inactive = predicted_credit[inactive]
            default_inactive = default_credit[inactive]
            multiplier = variance_multiplier[None, :].expand_as(
                exact_credit
            )[inactive]
            raw_variance = (exact_inactive.square() * multiplier).mean()
            cv_variance = (
                (exact_inactive - predicted_inactive).square() * multiplier
            ).mean()
            default_variance = (
                (exact_inactive - default_inactive).square() * multiplier
            ).mean()
            exact_second_moment = exact_inactive.square().mean()
            temporal_common_variance = exact_second_moment * (
                1.0 / temporal_probability - 1.0
            )
            temporal_raw_variance = (
                raw_variance / temporal_probability + temporal_common_variance
            )
            temporal_cv_variance = (
                cv_variance / temporal_probability + temporal_common_variance
            )
            temporal_default_variance = (
                default_variance / temporal_probability + temporal_common_variance
            )
            selected_credit = exact_credit.gather(1, selected_ids)
            local_regret = exact_credit.max(dim=-1).values - selected_credit.max(
                dim=-1
            ).values

            record = {
                "batch": batch_index,
                "layer": layer_index,
                "tokens": sample_count,
                "loss": float(loss.detach()),
                "predictor_output_mse": float(
                    (predicted_outputs.float() - exact_outputs.float())
                    .square()[inactive[:, :, None].expand_as(exact_outputs)]
                    .mean()
                ),
                "predicted_credit_mse": float(
                    (predicted_inactive - exact_inactive).square().mean()
                ),
                "default_credit_mse": float(
                    (default_inactive - exact_inactive).square().mean()
                ),
                "predicted_credit_spearman": float(
                    rank_correlation(predicted_credit, exact_credit).mean()
                ),
                "default_credit_spearman": float(
                    rank_correlation(default_credit, exact_credit).mean()
                ),
                "predicted_top2_recall": float(
                    topk_recall(predicted_credit, exact_credit, 2).mean()
                ),
                "router_top2_recall": float(
                    topk_recall(logits.index_select(0, rows), exact_credit, 2).mean()
                ),
                "mean_local_router_regret": float(local_regret.mean()),
                "positive_local_regret_fraction": float(
                    (local_regret > 0).float().mean()
                ),
                "raw_ht_variance": float(raw_variance),
                "default_cv_variance": float(default_variance),
                "predicted_cv_variance": float(cv_variance),
                "predicted_vs_raw_variance_ratio": float(
                    cv_variance / raw_variance.clamp_min(1e-30)
                ),
                "predicted_vs_default_variance_ratio": float(
                    cv_variance / default_variance.clamp_min(1e-30)
                ),
                "temporal_raw_ht_variance": float(temporal_raw_variance),
                "temporal_default_cv_variance": float(temporal_default_variance),
                "temporal_predicted_cv_variance": float(temporal_cv_variance),
                "temporal_predicted_vs_raw_variance_ratio": float(
                    temporal_cv_variance
                    / temporal_raw_variance.clamp_min(1e-30)
                ),
                "temporal_predicted_vs_default_variance_ratio": float(
                    temporal_cv_variance
                    / temporal_default_variance.clamp_min(1e-30)
                ),
                "cvcr_update_interval": config.cvcr_update_interval,
                "active_microbatch_probe_fraction": effective_probe_fraction,
                "minimum_inclusion_probability": float(
                    inclusion[eligible].min()
                ),
                "maximum_importance_weight": float(
                    inclusion[eligible].min().reciprocal()
                ),
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            del loss, captured

    metric_keys = [
        key
        for key, value in records[0].items()
        if isinstance(value, float) and key != "loss"
    ]
    summary = {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in metric_keys
    }
    result = {
        "status": "ORACLE_LOCAL_CREDIT_COMPLETE",
        "claim_boundary": (
            "Exhaustive experts establish local first-order credit diagnostics only; "
            "this is not exact downstream counterfactual loss or a true discrete-router gradient."
        ),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": digest(arguments.checkpoint),
        "checkpoint_step": checkpoint_step,
        "config": config.as_dict(),
        "data_status": data_manifest["status"],
        "data_revision": data_manifest.get("revision"),
        "batches": arguments.batches,
        "records": records,
        "summary": summary,
        "unix": time.time(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".next")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(json.dumps({"output": str(arguments.output), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
