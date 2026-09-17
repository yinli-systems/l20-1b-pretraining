#!/usr/bin/env python3
"""Exact equal-compute reroutes for a pre-specified Top-4 candidate set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from cvcr_moe.config import ModelConfig
from cvcr_moe.data import PackedReader, verify_frozen_manifest
from cvcr_moe.model import MoELanguageModel


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def ranks(values: np.ndarray) -> np.ndarray:
    return values.argsort().argsort().astype(np.float64)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--layers", default="1,3,5,7")
    result.add_argument("--tokens-per-layer", type=int, default=8)
    result.add_argument("--alternative-candidates", type=int, default=2)
    result.add_argument("--seed", type=int, default=20260916)
    return result


def mean_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


def main() -> None:
    arguments = parser().parse_args()
    torch.manual_seed(arguments.seed)
    torch.cuda.set_device(0)
    data_manifest = verify_frozen_manifest(arguments.data_dir)
    config = ModelConfig.from_json(arguments.config)
    model = MoELanguageModel(config).cuda().eval().requires_grad_(False)
    state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    checkpoint_step = int(state["step"])
    del state
    reader = PackedReader(arguments.data_dir, arguments.seed + 41)
    inputs, targets = reader.batch_for_step(0, 0, 1, 1)
    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    layer_ids = [int(value) for value in arguments.layers.split(",")]
    if any(value < 0 or value >= len(model.layers) for value in layer_ids):
        raise ValueError("requested counterfactual layer is out of range")
    if 2 + arguments.alternative_candidates > config.num_experts:
        raise ValueError("candidate set is larger than the expert bank")
    generator = torch.Generator(device="cuda").manual_seed(arguments.seed + 42)
    records: list[dict[str, float | int]] = []

    for layer_index in layer_ids:
        moe = model.layers[layer_index].moe
        captured: dict[str, torch.Tensor] = {}

        def capture_input(_module, values):
            captured["hidden"] = values[0].detach()

        def cut_graph(_module, _values, output):
            leaf = output[0].detach().requires_grad_(True)
            captured["mixed"] = leaf
            return leaf, output[1], output[2]

        pre_handle = moe.register_forward_pre_hook(capture_input)
        post_handle = moe.register_forward_hook(cut_graph)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            baseline_logits = model(inputs)
        baseline_loss = mean_ce(baseline_logits, targets)
        baseline_loss.backward()
        pre_handle.remove()
        post_handle.remove()

        hidden = captured["hidden"].reshape(-1, config.hidden_size)
        baseline_mixed = captured["mixed"].detach().reshape(
            -1, config.hidden_size
        )
        downstream_gradient = captured["mixed"].grad.detach().reshape(
            -1, config.hidden_size
        )
        positions = torch.randperm(
            hidden.shape[0], generator=generator, device="cuda"
        )[: arguments.tokens_per_layer]

        with torch.no_grad():
            router_logits = moe.router(hidden.float())
            candidate_ids = router_logits.topk(
                2 + arguments.alternative_candidates, dim=-1
            ).indices

        for position_tensor in positions:
            position = int(position_tensor.item())
            strongest = candidate_ids[position, 0]
            for candidate_offset in range(arguments.alternative_candidates):
                candidate = candidate_ids[position, 2 + candidate_offset]
                route = torch.stack((strongest, candidate))
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    expert_outputs = moe.experts.forward_assignments(
                        hidden[position : position + 1].expand(2, -1),
                        route,
                        detach_parameters=True,
                    )
                    route_weights = router_logits[position, route].softmax(
                        dim=-1
                    ).to(expert_outputs.dtype)
                    alternative_output = (
                        expert_outputs * route_weights[:, None]
                    ).sum(dim=0)

                def override(_module, _values, output):
                    changed = output[0].clone()
                    changed.reshape(-1, config.hidden_size)[position] = (
                        alternative_output
                    )
                    return changed, output[1], output[2]

                handle = moe.register_forward_hook(override)
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    alternative_logits = model(inputs)
                    alternative_loss = mean_ce(alternative_logits, targets)
                handle.remove()
                local_improvement = -torch.dot(
                    downstream_gradient[position].float(),
                    (
                        alternative_output.float()
                        - baseline_mixed[position].float()
                    ),
                )
                actual_improvement = baseline_loss.detach() - alternative_loss
                record = {
                    "layer": layer_index,
                    "token_position": position,
                    "candidate_rank": 3 + candidate_offset,
                    "candidate_expert": int(candidate.item()),
                    "local_predicted_improvement": float(local_improvement),
                    "actual_mean_ce_improvement": float(actual_improvement),
                }
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
        del captured, baseline_logits, baseline_loss

    local = np.asarray(
        [record["local_predicted_improvement"] for record in records],
        dtype=np.float64,
    )
    actual = np.asarray(
        [record["actual_mean_ce_improvement"] for record in records],
        dtype=np.float64,
    )
    grouped: dict[tuple[int, int], list[float]] = {}
    for record in records:
        key = (int(record["layer"]), int(record["token_position"]))
        grouped.setdefault(key, []).append(float(record["actual_mean_ce_improvement"]))
    regret = np.asarray(
        [max(0.0, max(improvements)) for improvements in grouped.values()],
        dtype=np.float64,
    )
    result = {
        "status": "EXACT_EQUAL_COMPUTE_REROUTES_COMPLETE",
        "claim_boundary": (
            "Each rerun replaces the weaker Top-2 route member with router rank 3 or 4 "
            "at one token/layer and measures exact full-block mean CE. The fixed candidate "
            "set is not an exhaustive search over routes."
        ),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": digest(arguments.checkpoint),
        "checkpoint_step": checkpoint_step,
        "data_status": data_manifest["status"],
        "data_revision": data_manifest.get("revision"),
        "layers": layer_ids,
        "tokens_per_layer": arguments.tokens_per_layer,
        "alternative_candidates": arguments.alternative_candidates,
        "reroutes": len(records),
        "positive_alternative_fraction": float((actual > 0).mean()),
        "mean_positive_equal_compute_regret": float(regret.mean()),
        "tokens_with_positive_regret_fraction": float((regret > 0).mean()),
        "local_actual_pearson": correlation(local, actual),
        "local_actual_spearman": correlation(ranks(local), ranks(actual)),
        "records": records,
        "unix": time.time(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".next")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True))


if __name__ == "__main__":
    main()
