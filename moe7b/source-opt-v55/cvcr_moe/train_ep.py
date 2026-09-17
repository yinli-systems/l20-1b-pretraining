"""Bounded 7B Top-2 expert-parallel throughput and memory qualification."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .config import ModelConfig
from .data import PackedReader, verify_frozen_manifest
from .model import DecoderBlock, MoELanguageModel
from .train import atomic_json, digest, learning_rate_for_step


CLAIM = (
    "A bounded pure-BF16 expert-parallel screen proves construction, exact token "
    "dispatch, forward/backward, optimizer execution, memory, and measured throughput. "
    "It is not numerically promoted and is not a completed pretraining run."
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--val-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--target-tokens", type=int, required=True)
    result.add_argument("--microbatch", type=int, default=1)
    result.add_argument("--accumulation", type=int, default=8)
    result.add_argument("--max-steps", type=int, default=12)
    result.add_argument("--peak-lr", type=float, default=4e-4)
    result.add_argument("--warmup-tokens", type=int, default=200_000_000)
    result.add_argument("--log-every", type=int, default=1)
    result.add_argument("--seed", type=int, default=20260917)
    result.add_argument("--ddp-bucket-cap-mb", type=int, default=100)
    result.add_argument("--rated-dense-bf16-tflops-per-gpu", type=float, default=209.5)
    result.add_argument("--required-device", default="NVIDIA GeForce RTX 5090")
    return result


def checkpoint_block(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)


def expert_parameter_ids(model: MoELanguageModel) -> set[int]:
    return {
        id(parameter)
        for layer in model.layers
        for parameter in layer.moe.experts.parameters()
    }


def global_clip_grad_norm(
    model: MoELanguageModel, expert_ids: set[int], maximum: float
) -> torch.Tensor:
    device = next(model.parameters()).device
    shared_square = torch.zeros((), device=device, dtype=torch.float32)
    expert_square = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        square = parameter.grad.detach().float().square().sum()
        if id(parameter) in expert_ids:
            expert_square += square
        else:
            shared_square += square
    dist.all_reduce(expert_square)
    norm = (shared_square + expert_square).sqrt().float()
    scale = torch.clamp(maximum / (norm + 1e-6), max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    return norm


def main() -> None:
    arguments = parser().parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (4, 8):
        raise ValueError("the frozen screen requires four or eight expert-parallel ranks")
    if arguments.max_steps < 8 or arguments.log_every < 1:
        raise ValueError("the bounded screen needs at least eight steps and positive logging")
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError("refusing to write into a non-empty output directory")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(arguments.seed)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config = ModelConfig.from_json(arguments.config)
    if config.method != "top2" or config.expert_backend != "scattermoe":
        raise ValueError("the expert-parallel screen requires Top-2 ScatterMoE")
    if config.loss_backend != "liger" or config.num_experts != 16:
        raise ValueError("the expert-parallel screen requires Liger CE and 16 experts")
    if config.num_experts % world_size:
        raise ValueError("the expert count must divide the expert-parallel world size")
    data_manifest = verify_frozen_manifest(arguments.data_dir)
    train_reader = PackedReader(arguments.data_dir, arguments.seed, repeat=True)
    tokens_per_step = (
        arguments.microbatch
        * arguments.accumulation
        * world_size
        * config.max_position_embeddings
    )
    if tokens_per_step != 131_072:
        raise ValueError("the screen must preserve 131,072 prediction tokens per step")
    total_steps = min(arguments.max_steps, arguments.target_tokens // tokens_per_step)
    warmup_steps = max(1, arguments.warmup_tokens // tokens_per_step)

    if rank == 0:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    raw_model = MoELanguageModel(config)
    global_parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    if global_parameters != config.deployed_parameter_count():
        raise RuntimeError("materialized global 7B parameter count changed")
    raw_model.enable_expert_parallel(rank, world_size)
    local_expert_ids = expert_parameter_ids(raw_model)
    for index, layer in enumerate(list(raw_model.layers)):
        if not isinstance(layer, DecoderBlock):
            raise TypeError("unexpected decoder block type")
        raw_model.layers[index] = checkpoint_wrapper(
            layer,
            checkpoint_fn=checkpoint_block,
        )
    raw_model.to(device=device, dtype=torch.bfloat16)
    raw_model.cosine = raw_model.cosine.float()
    raw_model.sine = raw_model.sine.float()
    ignored = {
        name
        for name, parameter in raw_model.named_parameters()
        if id(parameter) in local_expert_ids
    }
    DDP._set_params_and_buffers_to_ignore_for_model(raw_model, ignored)
    model = DDP(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        bucket_cap_mb=arguments.ddp_bucket_cap_mb,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )

    devices: list[dict | None] = [None] * world_size
    descriptor = {
        "rank": rank,
        "hostname": os.uname().nodename,
        "device": torch.cuda.get_device_name(local_rank),
        "memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
    }
    dist.all_gather_object(devices, descriptor)
    if any(row is None or row["device"] != arguments.required_device for row in devices):
        raise RuntimeError("the screen did not receive the exact requested GPU type")

    local_parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    local_expert_parameters = sum(
        parameter.numel()
        for parameter in raw_model.parameters()
        if id(parameter) in local_expert_ids
    )
    active_flops = raw_model.active_flops_per_token(config.max_position_embeddings)
    causal_flops = raw_model.causal_matmul_flops_per_token(config.max_position_embeddings)
    peak = world_size * arguments.rated_dense_bf16_tflops_per_gpu * 1e12
    identity = {
        "model_config": config.as_dict(),
        "global_deployed_parameters": global_parameters,
        "active_parameters": config.active_parameter_count(),
        "local_materialized_parameters": local_parameters,
        "local_expert_parameters": local_expert_parameters,
        "world_size": world_size,
        "expert_parallel_degree": world_size,
        "experts_per_rank": config.num_experts // world_size,
        "data_parallel_degree_for_shared_parameters": world_size,
        "microbatch": arguments.microbatch,
        "accumulation": arguments.accumulation,
        "tokens_per_step": tokens_per_step,
        "target_tokens": total_steps * tokens_per_step,
        "max_steps": total_steps,
        "precision": "pure_bfloat16_parameters_gradients_and_adam_moments",
        "ddp_ignored_parameter_names": sorted(ignored),
        "data_manifest_sha256": digest(arguments.data_dir.parent / "manifest.json"),
        "config_sha256": digest(arguments.config),
        "code_sha256": {
            name: digest(Path(__file__).resolve().parent / name)
            for name in ("train_ep.py", "model.py", "config.py", "data.py")
        },
    }
    fingerprint = __import__("hashlib").sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if rank == 0:
        atomic_json(
            {
                "schema": "cvcr-moe-7b-expert-parallel-screen-v1",
                "status": "bounded_feasibility_screen",
                "claim_boundary": CLAIM,
                **identity,
                "run_fingerprint_sha256": fingerprint,
                "active_flops_per_token_full_square": active_flops,
                "causal_useful_matmul_flops_per_token": causal_flops,
                "rated_dense_bf16_tflops_per_gpu": arguments.rated_dense_bf16_tflops_per_gpu,
                "runtime": {"torch": torch.__version__, "devices": devices},
            },
            arguments.output_dir / "run-manifest.json",
        )

    model.train()
    metrics_path = arguments.output_dir / "metrics.jsonl"
    started = time.monotonic()
    logging_started = started
    logging_step = 0
    torch.cuda.reset_peak_memory_stats(device)
    mfu_samples: list[float] = []
    for step in range(1, total_steps + 1):
        rate = learning_rate_for_step(
            step - 1,
            max(total_steps, 1),
            arguments.peak_lr,
            warmup_steps,
            max(1, total_steps // 10),
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        ce_sum = torch.zeros((), device=device, dtype=torch.float64)
        router_sum = torch.zeros((), device=device, dtype=torch.float64)
        for accumulation_step in range(arguments.accumulation):
            inputs, targets = train_reader.batch_for_step(
                (step - 1) * arguments.accumulation + accumulation_step,
                rank,
                world_size,
                arguments.microbatch,
            )
            inputs = inputs.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
            sync_context = (
                contextlib.nullcontext()
                if accumulation_step + 1 == arguments.accumulation
                else model.no_sync()
            )
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, cross_entropy, router_auxiliary, _ = model(inputs, targets)
                scaled_loss = loss / arguments.accumulation
            scaled_loss.backward()
            loss_sum += scaled_loss.detach().double()
            ce_sum += cross_entropy.double() / arguments.accumulation
            router_sum += router_auxiliary.double() / arguments.accumulation

        # DDP has already averaged replicated gradients. Each expert owner has
        # accumulated contributions from every source rank, so apply the same
        # global-mean scale exactly once to its disjoint expert gradients.
        for parameter in raw_model.parameters():
            if id(parameter) in local_expert_ids and parameter.grad is not None:
                parameter.grad.div_(world_size)
        gradient_norm = global_clip_grad_norm(raw_model, local_expert_ids, 1.0)
        optimizer.step()

        should_log = step == 1 or step % arguments.log_every == 0 or step == total_steps
        if should_log:
            torch.cuda.synchronize(device)
            reductions = torch.stack((loss_sum, ce_sum, router_sum))
            dist.all_reduce(reductions)
            reductions /= world_size
            duration = torch.tensor(time.monotonic() - logging_started, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            interval_steps = step - logging_step
            seconds = float(duration)
            throughput = interval_steps * tokens_per_step / seconds
            peak_memory = torch.tensor(
                [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                device=device,
                dtype=torch.int64,
            )
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            record = {
                "step": step,
                "prediction_tokens": step * tokens_per_step,
                "loss": float(reductions[0]),
                "cross_entropy": float(reductions[1]),
                "router_auxiliary": float(reductions[2]),
                "learning_rate": rate,
                "gradient_norm": float(gradient_norm),
                "measurement_steps": interval_steps,
                "step_seconds_rank_max": seconds / interval_steps,
                "tokens_per_second": throughput,
                "rated_full_square_mfu": throughput * active_flops / peak,
                "causal_useful_matmul_mfu": throughput * causal_flops / peak,
                "peak_memory_allocated_bytes_rank_max": int(peak_memory[0]),
                "peak_memory_reserved_bytes_rank_max": int(peak_memory[1]),
                "elapsed_seconds": time.monotonic() - started,
                "unix": time.time(),
            }
            if rank == 0:
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if step > 2:
                    mfu_samples.append(record["causal_useful_matmul_mfu"])
            logging_started = time.monotonic()
            logging_step = step
            torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        atomic_json(
            {
                "status": "COMPLETED",
                "step": total_steps,
                "prediction_tokens": total_steps * tokens_per_step,
                "median_causal_useful_matmul_mfu_after_warmup": statistics.median(mfu_samples),
                "unix": time.time(),
            },
            arguments.output_dir / "training-status.json",
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
