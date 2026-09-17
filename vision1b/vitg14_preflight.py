#!/usr/bin/env python3
"""Bounded ViT-g/14 student/teacher memory and throughput qualification."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--batch-per-rank", type=int, default=1)
    parser.add_argument("--global-views", type=int, default=2)
    parser.add_argument("--local-views", type=int, default=8)
    parser.add_argument("--global-size", type=int, default=224)
    parser.add_argument("--local-size", type=int, default=98)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--ffn-width", type=int, default=4096)
    parser.add_argument("--projection-dim", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--peak-bf16-tflops-per-gpu", type=float, default=165.2)
    parser.add_argument("--shard-group-size", type=int, default=0)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-every-n-blocks", type=int, default=1)
    parser.add_argument(
        "--fused-multicrop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--teacher-reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--student-reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compile-mode", choices=("none", "default", "max-autotune-no-cudagraphs"), default="none")
    parser.add_argument(
        "--float8-recipe",
        choices=("none", "tensorwise", "rowwise", "rowwise_with_gw_hp"),
        default="none",
    )
    parser.add_argument(
        "--float8-fsdp-all-gather",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


class Attention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.proj = nn.Linear(width, width, bias=True)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, tokens, width = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        state = F.scaled_dot_product_attention(query, key, value)
        return self.proj(state.transpose(1, 2).reshape(batch, tokens, width))


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate_value = nn.Linear(width, 2 * hidden, bias=True)
        self.output = nn.Linear(hidden, width, bias=True)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.gate_value(inputs).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        ffn_width: int,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, eps=1e-6)
        self.attention = Attention(width, heads)
        self.ffn_norm = nn.LayerNorm(width, eps=1e-6)
        self.ffn = SwiGLU(width, ffn_width)
        self.attention_scale = nn.Parameter(torch.full((width,), layer_scale_init))
        self.ffn_scale = nn.Parameter(torch.full((width,), layer_scale_init))

    def forward_one(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention_scale * self.attention(self.attention_norm(inputs))
        return inputs + self.ffn_scale * self.ffn(self.ffn_norm(inputs))

    def forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        outputs = tuple(self.forward_one(state) for state in inputs)
        return outputs[0] if len(outputs) == 1 else outputs


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        depth: int,
        heads: int,
        ffn_width: int,
        projection_dim: int,
        activation_checkpointing: bool,
        checkpoint_every_n_blocks: int = 1,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.activation_checkpointing = activation_checkpointing
        if checkpoint_every_n_blocks < 1:
            raise ValueError("checkpoint_every_n_blocks must be at least 1")
        self.checkpoint_every_n_blocks = checkpoint_every_n_blocks
        grid = image_size // patch_size
        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.empty(1, 1, width))
        self.position = nn.Parameter(torch.empty(1, 1 + grid * grid, width))
        self.blocks = nn.ModuleList(
            [Block(width, heads, ffn_width, layer_scale_init) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.projection = nn.Linear(width, projection_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def position_for(self, grid: int) -> Tensor:
        source_grid = round(math.sqrt(self.position.shape[1] - 1))
        if grid == source_grid:
            return self.position
        cls_position = self.position[:, :1]
        patch_position = self.position[:, 1:].reshape(1, source_grid, source_grid, self.width)
        patch_position = patch_position.permute(0, 3, 1, 2)
        patch_position = F.interpolate(
            patch_position.float(), size=(grid, grid), mode="bicubic", align_corners=False
        ).to(self.position.dtype)
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(1, grid * grid, self.width)
        return torch.cat((cls_position, patch_position), dim=1)

    def embed(self, pixels: Tensor) -> Tensor:
        patches = self.patch_embed(pixels).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(patches.shape[0], -1, -1)
        state = torch.cat((cls, patches), dim=1)
        grid = pixels.shape[-1] // self.patch_size
        return state + self.position_for(grid)

    def forward(
        self, pixels: Tensor | tuple[Tensor, ...]
    ) -> Tensor | tuple[Tensor, ...]:
        multicrop = isinstance(pixels, tuple)
        pixel_views = pixels if multicrop else (pixels,)
        states = tuple(self.embed(view) for view in pixel_views)
        for block_index, block in enumerate(self.blocks):
            if (
                self.training
                and self.activation_checkpointing
                and block_index % self.checkpoint_every_n_blocks == 0
            ):
                block_output = checkpoint(block, *states, use_reentrant=False)
            else:
                block_output = block(*states)
            states = block_output if isinstance(block_output, tuple) else (block_output,)
        outputs = tuple(self.projection(self.norm(state[:, 0])) for state in states)
        return outputs if multicrop else outputs[0]


def model_parameters(arguments: argparse.Namespace) -> int:
    with torch.device("meta"):
        model = VisionTransformer(
            arguments.global_size,
            arguments.patch_size,
            arguments.width,
            arguments.depth,
            arguments.heads,
            arguments.ffn_width,
            arguments.projection_dim,
            activation_checkpointing=True,
            checkpoint_every_n_blocks=arguments.checkpoint_every_n_blocks,
        )
    return sum(parameter.numel() for parameter in model.parameters())


def forward_flops(arguments: argparse.Namespace, image_size: int) -> int:
    """Count dense matmul FLOPs for one image, excluding elementwise kernels."""
    grid = image_size // arguments.patch_size
    tokens = 1 + grid * grid
    patch_projection = (
        2 * grid * grid * 3 * arguments.patch_size**2 * arguments.width
    )
    attention_projections = 8 * tokens * arguments.width**2
    attention_matmuls = 4 * tokens**2 * arguments.width
    feed_forward = 6 * tokens * arguments.width * arguments.ffn_width
    blocks = arguments.depth * (
        attention_projections + attention_matmuls + feed_forward
    )
    projection = 2 * arguments.width * arguments.projection_dim
    return patch_projection + blocks + projection


def useful_training_flops_per_source(arguments: argparse.Namespace) -> int:
    """Useful model FLOPs: student forward/backward plus teacher forward."""
    global_forward = forward_flops(arguments, arguments.global_size)
    local_forward = forward_flops(arguments, arguments.local_size)
    student = 3 * (
        arguments.global_views * global_forward
        + arguments.local_views * local_forward
    )
    teacher = arguments.global_views * global_forward
    return student + teacher


def hardware_training_flops_per_source(arguments: argparse.Namespace) -> int:
    """Executed model FLOPs, including activation-checkpoint recomputation."""
    useful = useful_training_flops_per_source(arguments)
    if not arguments.activation_checkpointing:
        return useful
    global_forward = forward_flops(arguments, arguments.global_size)
    local_forward = forward_flops(arguments, arguments.local_size)
    checkpointed_blocks = (arguments.depth - 1) // arguments.checkpoint_every_n_blocks + 1
    recompute = (
        arguments.global_views * global_forward
        + arguments.local_views * local_forward
    ) * checkpointed_blocks // arguments.depth
    return useful + recompute


def shard_model(
    model: VisionTransformer,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool = True,
) -> None:
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=policy,
        reshard_after_forward=reshard_after_forward,
    )


def convert_float8(model: VisionTransformer, arguments: argparse.Namespace) -> int:
    """Convert eligible Linear layers before FSDP2 wrapping and compilation."""
    if arguments.float8_recipe == "none":
        if arguments.float8_fsdp_all_gather:
            raise ValueError("float8 FSDP all-gather requires a float8 recipe")
        return 0
    if arguments.float8_fsdp_all_gather and arguments.float8_recipe != "tensorwise":
        raise ValueError("float8 FSDP all-gather is supported only for tensorwise scaling")

    from torchao.float8 import Float8LinearConfig
    from torchao.float8.float8_linear import Float8Linear
    from torchao.float8.float8_linear_utils import convert_to_float8_training

    config = Float8LinearConfig.from_recipe_name(arguments.float8_recipe)
    if arguments.float8_fsdp_all_gather:
        config = replace(config, enable_fsdp_float8_all_gather=True)

    def eligible(module: nn.Module, _fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        output_features, input_features = module.weight.shape
        return input_features % 16 == 0 and output_features % 16 == 0

    convert_to_float8_training(model, module_filter_fn=eligible, config=config)
    return sum(isinstance(module, Float8Linear) for module in model.modules())


def normalized(outputs: Tensor) -> Tensor:
    return F.normalize(outputs.float(), dim=-1)


def main() -> None:
    arguments = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(arguments.seed + rank)
    # FSDP2 does not broadcast independently initialized parameters. All ranks
    # must construct exactly the same unsharded weights before fully_shard().
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    shard_group_size = arguments.shard_group_size or world_size
    if world_size % shard_group_size:
        raise RuntimeError(
            f"world size {world_size} must be divisible by shard group size {shard_group_size}"
        )
    shard_mesh = None
    if shard_group_size < world_size:
        shard_mesh = init_device_mesh(
            "cuda",
            (world_size // shard_group_size, shard_group_size),
            mesh_dim_names=("replicate", "shard"),
        )
    parameter_count = model_parameters(arguments)
    useful_source_flops = useful_training_flops_per_source(arguments)
    hardware_source_flops = hardware_training_flops_per_source(arguments)
    if not 1_000_000_000 <= parameter_count <= 1_250_000_000:
        raise RuntimeError(f"unexpected ViT-g parameter count: {parameter_count}")

    student = VisionTransformer(
        arguments.global_size,
        arguments.patch_size,
        arguments.width,
        arguments.depth,
        arguments.heads,
        arguments.ffn_width,
        arguments.projection_dim,
        activation_checkpointing=arguments.activation_checkpointing,
        checkpoint_every_n_blocks=arguments.checkpoint_every_n_blocks,
    ).to(device)
    teacher = VisionTransformer(
        arguments.global_size,
        arguments.patch_size,
        arguments.width,
        arguments.depth,
        arguments.heads,
        arguments.ffn_width,
        arguments.projection_dim,
        activation_checkpointing=False,
        checkpoint_every_n_blocks=arguments.checkpoint_every_n_blocks,
    ).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    student_float8_linears = convert_float8(student, arguments)
    teacher_float8_linears = convert_float8(teacher, arguments)
    shard_model(
        student,
        mesh=shard_mesh,
        reshard_after_forward=arguments.student_reshard_after_forward,
    )
    shard_model(
        teacher,
        mesh=shard_mesh,
        reshard_after_forward=arguments.teacher_reshard_after_forward,
    )
    student.train()
    teacher.eval()

    if arguments.compile_mode != "none":
        student = torch.compile(student, mode=arguments.compile_mode)
        teacher = torch.compile(teacher, mode=arguments.compile_mode)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=arguments.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.04,
        fused=True,
    )
    generator = torch.Generator(device=device).manual_seed(arguments.seed + rank)
    global_pixels = torch.randn(
        arguments.batch_per_rank * arguments.global_views,
        3,
        arguments.global_size,
        arguments.global_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    local_pixels = torch.randn(
        arguments.batch_per_rank * arguments.local_views,
        3,
        arguments.local_size,
        arguments.local_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    manifest = {
        "status": "systems_qualification_only",
        "architecture": "DINO-style ViT-g/14",
        "student_parameters": parameter_count,
        "teacher_parameters": parameter_count,
        "world_size": world_size,
        "shard_group_size": shard_group_size,
        "replica_group_size": world_size // shard_group_size,
        "parallelism": "HSDP" if shard_mesh is not None else "FSDP2",
        "batch_per_rank": arguments.batch_per_rank,
        "global_views": arguments.global_views,
        "local_views": arguments.local_views,
        "global_size": arguments.global_size,
        "local_size": arguments.local_size,
        "compile_mode": arguments.compile_mode,
        "float8_recipe": arguments.float8_recipe,
        "float8_fsdp_all_gather": arguments.float8_fsdp_all_gather,
        "student_float8_linears": student_float8_linears,
        "teacher_float8_linears": teacher_float8_linears,
        "activation_checkpointing": arguments.activation_checkpointing,
        "checkpoint_every_n_blocks": arguments.checkpoint_every_n_blocks,
        "fused_multicrop": arguments.fused_multicrop,
        "teacher_reshard_after_forward": arguments.teacher_reshard_after_forward,
        "student_reshard_after_forward": arguments.student_reshard_after_forward,
        "dtype": (
            "FP8 eligible Linear GEMMs with BF16 attention/non-Linear compute and FP32 parameters/reductions/optimizer state"
            if arguments.float8_recipe != "none"
            else "BF16 compute with FP32 parameters/reductions/optimizer state"
        ),
        "synthetic_inputs": True,
        "useful_training_flops_per_source_image": useful_source_flops,
        "hardware_training_flops_per_source_image": hardware_source_flops,
        "peak_bf16_tflops_per_gpu": arguments.peak_bf16_tflops_per_gpu,
        "mfu_definition": "useful dense matmul FLOPs for student forward/backward plus global teacher forward divided by configured aggregate non-sparse dense BF16 peak; checkpoint recomputation and elementwise kernels are excluded",
        "hfu_definition": "executed dense matmul FLOPs including checkpoint recomputation divided by configured aggregate non-sparse dense BF16 peak; elementwise kernels are excluded",
        "float8_metric_boundary": (
            "For FP8 runs, BF16-peak ratios are throughput-equivalent ratios only and are not FP8 MFU/HFU."
            if arguments.float8_recipe != "none"
            else None
        ),
        "claim_boundary": "This run tests construction, forward, backward, optimizer, memory and synthetic throughput only. It is not image pretraining and provides no quality or convergence evidence.",
    }
    if rank == 0:
        (arguments.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    torch.cuda.reset_peak_memory_stats(device)
    metrics_path = arguments.output_dir / "metrics.jsonl"
    for step in range(1, arguments.steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_global = normalized(teacher(global_pixels))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if arguments.fused_multicrop:
                student_global_raw, student_local_raw = student(
                    (global_pixels, local_pixels)
                )
            else:
                student_global_raw = student(global_pixels)
                student_local_raw = student(local_pixels)
            student_global = normalized(student_global_raw)
            student_local = normalized(student_local_raw)
            target = teacher_global.reshape(
                arguments.batch_per_rank, arguments.global_views, -1
            ).mean(dim=1)
            global_target = target.repeat_interleave(arguments.global_views, dim=0)
            local_target = target.repeat_interleave(arguments.local_views, dim=0)
            loss = F.mse_loss(student_global, global_target) + F.mse_loss(
                student_local, local_target
            )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        reduced = loss.detach().float()
        if distributed:
            dist.all_reduce(reduced)
            reduced /= world_size
        useful_peak_ratio = (
            useful_source_flops
            * arguments.batch_per_rank
            * world_size
            / elapsed
            / (world_size * arguments.peak_bf16_tflops_per_gpu * 1e12)
        )
        hardware_peak_ratio = (
            hardware_source_flops
            * arguments.batch_per_rank
            * world_size
            / elapsed
            / (world_size * arguments.peak_bf16_tflops_per_gpu * 1e12)
        )
        record = {
            "step": step,
            "loss": reduced.item(),
            "finite": bool(torch.isfinite(reduced).item()),
            "step_seconds": elapsed,
            "source_images_per_second": arguments.batch_per_rank * world_size / elapsed,
            "augmented_views_per_second": arguments.batch_per_rank * world_size * (arguments.global_views + arguments.local_views) / elapsed,
            "estimated_dense_bf16_mfu": useful_peak_ratio if arguments.float8_recipe == "none" else None,
            "estimated_dense_bf16_hfu": hardware_peak_ratio if arguments.float8_recipe == "none" else None,
            "bf16_peak_equivalent_useful_ratio": useful_peak_ratio if arguments.float8_recipe != "none" else None,
            "bf16_peak_equivalent_hardware_ratio": hardware_peak_ratio if arguments.float8_recipe != "none" else None,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        if not record["finite"]:
            raise RuntimeError(f"non-finite loss at step {step}")
        if rank == 0:
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

    if rank == 0:
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        measured = records[arguments.warmup_steps :]
        if not measured:
            raise RuntimeError("no measured steps after warmup")
        measured.sort(key=lambda item: item["source_images_per_second"])
        middle = measured[len(measured) // 2]
        summary = {
            "status": "PASS" if all(item["finite"] for item in measured) else "FAIL_NONFINITE",
            "measured_steps": len(measured),
            "median_source_images_per_second": middle["source_images_per_second"],
            "median_augmented_views_per_second": middle["augmented_views_per_second"],
            "median_estimated_dense_bf16_mfu": middle["estimated_dense_bf16_mfu"],
            "median_estimated_dense_bf16_hfu": middle["estimated_dense_bf16_hfu"],
            "median_bf16_peak_equivalent_useful_ratio": middle["bf16_peak_equivalent_useful_ratio"],
            "median_bf16_peak_equivalent_hardware_ratio": middle["bf16_peak_equivalent_hardware_ratio"],
            "peak_memory_allocated_bytes": max(item["peak_memory_allocated_bytes"] for item in records),
            "peak_memory_reserved_bytes": max(item["peak_memory_reserved_bytes"] for item in records),
            "claim_boundary": manifest["claim_boundary"],
        }
        (arguments.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
