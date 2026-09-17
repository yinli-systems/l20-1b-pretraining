#!/usr/bin/env python3
"""Paired frozen-block CE comparison with a non-token-IID bootstrap."""

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


def load_model(config_path: Path, checkpoint_path: Path) -> MoELanguageModel:
    config = ModelConfig.from_json(config_path)
    model = MoELanguageModel(config).cuda().eval().requires_grad_(False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    del state
    return model


def block_cross_entropy(
    model: MoELanguageModel, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs)
    losses = F.cross_entropy(
        logits.float().transpose(1, 2), targets, reduction="none"
    ).mean(dim=-1)
    return losses.cpu()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--baseline-config", type=Path, required=True)
    result.add_argument("--baseline-checkpoint", type=Path, required=True)
    result.add_argument("--candidate-config", type=Path, required=True)
    result.add_argument("--candidate-checkpoint", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--blocks", type=int, default=256)
    result.add_argument("--microbatch", type=int, default=2)
    result.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=20260916)
    return result


def main() -> None:
    arguments = parser().parse_args()
    torch.manual_seed(arguments.seed)
    torch.cuda.set_device(0)
    manifest = verify_frozen_manifest(arguments.data_dir)
    baseline = load_model(arguments.baseline_config, arguments.baseline_checkpoint)
    candidate = load_model(arguments.candidate_config, arguments.candidate_checkpoint)
    if baseline.config.vocab_size != candidate.config.vocab_size:
        raise ValueError("paired models use different vocabularies")
    if baseline.config.max_position_embeddings != candidate.config.max_position_embeddings:
        raise ValueError("paired models use different sequence lengths")
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    shared_deployed_keys = sorted(
        key
        for key in baseline_state
        if ".credit_predictor." not in key and ".probe_sampler." not in key
    )
    missing_deployed_keys = [
        key for key in shared_deployed_keys if key not in candidate_state
    ]
    if missing_deployed_keys:
        raise ValueError(f"candidate is missing deployed keys: {missing_deployed_keys[:4]}")
    deployed_mismatch_tensors = 0
    deployed_max_absolute_difference = 0.0
    for key in shared_deployed_keys:
        left = baseline_state[key]
        right = candidate_state[key]
        if not torch.equal(left, right):
            deployed_mismatch_tensors += 1
            deployed_max_absolute_difference = max(
                deployed_max_absolute_difference,
                float((left.float() - right.float()).abs().max().item()),
            )
    reader = PackedReader(arguments.data_dir, arguments.seed)
    baseline_losses: list[torch.Tensor] = []
    candidate_losses: list[torch.Tensor] = []
    batches = (arguments.blocks + arguments.microbatch - 1) // arguments.microbatch
    for batch_index in range(batches):
        inputs, targets = reader.batch_for_step(
            batch_index, 0, 1, arguments.microbatch
        )
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        baseline_losses.append(block_cross_entropy(baseline, inputs, targets))
        candidate_losses.append(block_cross_entropy(candidate, inputs, targets))
        if (batch_index + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "evaluated_blocks": min(
                            (batch_index + 1) * arguments.microbatch,
                            arguments.blocks,
                        )
                    }
                ),
                flush=True,
            )
    baseline_array = torch.cat(baseline_losses)[: arguments.blocks].numpy()
    candidate_array = torch.cat(candidate_losses)[: arguments.blocks].numpy()
    difference = candidate_array - baseline_array
    rng = np.random.default_rng(arguments.seed + 1)
    bootstrap = np.empty(arguments.bootstrap_repetitions, dtype=np.float64)
    for start in range(0, arguments.bootstrap_repetitions, 1000):
        stop = min(start + 1000, arguments.bootstrap_repetitions)
        indices = rng.integers(
            0, arguments.blocks, size=(stop - start, arguments.blocks)
        )
        bootstrap[start:stop] = difference[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    result = {
        "status": "PAIRED_BLOCK_BOOTSTRAP_COMPLETE",
        "claim_boundary": (
            "The sampling unit is a frozen packed 2049-token block, not an IID token "
            "and not necessarily an original source document."
        ),
        "baseline_checkpoint": str(arguments.baseline_checkpoint),
        "baseline_checkpoint_sha256": digest(arguments.baseline_checkpoint),
        "candidate_checkpoint": str(arguments.candidate_checkpoint),
        "candidate_checkpoint_sha256": digest(arguments.candidate_checkpoint),
        "shared_deployed_tensors": len(shared_deployed_keys),
        "deployed_mismatch_tensors": deployed_mismatch_tensors,
        "deployed_max_absolute_difference": deployed_max_absolute_difference,
        "data_status": manifest["status"],
        "data_revision": manifest.get("revision"),
        "blocks": arguments.blocks,
        "bootstrap_repetitions": arguments.bootstrap_repetitions,
        "baseline_cross_entropy": float(baseline_array.mean()),
        "candidate_cross_entropy": float(candidate_array.mean()),
        "candidate_minus_baseline_cross_entropy": float(difference.mean()),
        "paired_95_percent_interval": [float(lower), float(upper)],
        "candidate_lower_loss_block_fraction": float((difference < 0).mean()),
        "unix": time.time(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".next")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
