from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cvcr_moe.config import ModelConfig
from cvcr_moe.model import ExpertBank


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        hidden_size=4,
        expert_intermediate_size=3,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        experts_per_token=2,
        max_position_embeddings=8,
        tie_word_embeddings=True,
        eos_token_id=15,
        pad_token_id=0,
        expert_backend="loop",
        loss_backend="eager",
    )


def _worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}"
    )
    try:
        torch.manual_seed(17)
        baseline = ExpertBank(_config())
        with torch.no_grad():
            baseline.gate_up.copy_(
                torch.linspace(-0.2, 0.2, baseline.gate_up.numel()).view_as(
                    baseline.gate_up
                )
            )
            baseline.down.copy_(
                torch.linspace(0.15, -0.15, baseline.down.numel()).view_as(
                    baseline.down
                )
            )
        candidate = copy.deepcopy(baseline)
        candidate.enable_expert_parallel(rank, 2)

        hidden = (
            torch.arange(12, dtype=torch.float32).view(3, 4) / 11 + rank / 7
        ).requires_grad_()
        candidate_hidden = hidden.detach().clone().requires_grad_()
        ids = (
            torch.tensor([[0, 1], [2, 3], [0, 3]])
            if rank == 0
            else torch.tensor([[3, 2], [1, 0], [1, 2]])
        )

        expected = baseline.forward_topk(hidden, ids)
        expected.square().sum().backward()
        for parameter in baseline.parameters():
            dist.all_reduce(parameter.grad)

        observed = candidate.forward_topk(candidate_hidden, ids)
        observed.square().sum().backward()
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            candidate_hidden.grad, hidden.grad, rtol=1e-6, atol=1e-6
        )

        start = rank * 2
        stop = start + 2
        torch.testing.assert_close(
            candidate.gate_up.grad,
            baseline.gate_up.grad[start:stop],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            candidate.down.grad,
            baseline.down.grad[start:stop],
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_expert_parallel_matches_global_outputs_and_gradients() -> None:
    handle, rendezvous = tempfile.mkstemp(prefix="cvcr-ep-test-")
    os.close(handle)
    Path(rendezvous).unlink()
    try:
        mp.spawn(_worker, args=(rendezvous,), nprocs=2, join=True)
    finally:
        Path(rendezvous).unlink(missing_ok=True)


def test_partition_keeps_disjoint_globally_initialized_experts() -> None:
    torch.manual_seed(29)
    global_bank = ExpertBank(_config())
    with torch.no_grad():
        global_bank.gate_up.copy_(
            torch.linspace(-0.3, 0.3, global_bank.gate_up.numel()).view_as(
                global_bank.gate_up
            )
        )
        global_bank.down.copy_(
            torch.linspace(0.25, -0.25, global_bank.down.numel()).view_as(
                global_bank.down
            )
        )
    expected_gate_up = global_bank.gate_up.detach().clone()
    expected_down = global_bank.down.detach().clone()
    shards = []
    for rank in range(2):
        shard = copy.deepcopy(global_bank)
        shard.enable_expert_parallel(rank, 2)
        shards.append(shard)
    torch.testing.assert_close(
        torch.cat([shard.gate_up for shard in shards]), expected_gate_up
    )
    torch.testing.assert_close(
        torch.cat([shard.down for shard in shards]), expected_down
    )
