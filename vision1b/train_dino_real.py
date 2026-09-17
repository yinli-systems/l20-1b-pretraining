#!/usr/bin/env python3
"""Resumable real-image DINO class-token pilot for the qualified ViT-g/14."""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader, Dataset

import vitg14_preflight as backbone_source
from vitg14_preflight import (
    VisionTransformer,
    hardware_training_flops_per_source,
    useful_training_flops_per_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--continuation-protocol", type=Path)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--batch-per-rank", type=int, default=64)
    parser.add_argument("--workers-per-rank", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=10)
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
    parser.add_argument("--bottleneck-dim", type=int, default=384)
    parser.add_argument("--head-hidden-dim", type=int, default=2048)
    parser.add_argument("--layer-scale-init", type=float, default=1e-2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--student-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-temperature", type=float, default=0.04)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--ema-start", type=float, default=0.9995)
    parser.add_argument("--ema-end", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=3.0)
    parser.add_argument("--checkpoint-every-n-blocks", type=int, default=1)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune", "max-autotune-no-cudagraphs"),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--loss-scale", type=float, default=1024.0)
    parser.add_argument("--rehash-images", action="store_true")
    parser.add_argument(
        "--peak-dense-tflops-per-gpu",
        "--peak-bf16-tflops-per-gpu",
        dest="peak_dense_tflops_per_gpu",
        type=float,
        default=209.5,
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.set_defaults(activation_checkpointing=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def random_resized_crop(image: Image.Image, size: int, scale: tuple[float, float], rng: random.Random) -> Image.Image:
    width, height = image.size
    area = width * height
    ratio_min, ratio_max = 3 / 4, 4 / 3
    for _ in range(10):
        target = rng.uniform(*scale) * area
        ratio = math.exp(rng.uniform(math.log(ratio_min), math.log(ratio_max)))
        crop_width = round(math.sqrt(target * ratio))
        crop_height = round(math.sqrt(target / ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = rng.randint(0, width - crop_width)
            top = rng.randint(0, height - crop_height)
            return image.crop((left, top, left + crop_width, top + crop_height)).resize(
                (size, size), Image.Resampling.BICUBIC
            )
    source_ratio = width / height
    if source_ratio < ratio_min:
        crop_width, crop_height = width, round(width / ratio_min)
    elif source_ratio > ratio_max:
        crop_width, crop_height = round(height * ratio_max), height
    else:
        crop_width, crop_height = width, height
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (size, size), Image.Resampling.BICUBIC
    )


def color_jitter(image: Image.Image, rng: random.Random) -> Image.Image:
    operations = [
        lambda value: ImageEnhance.Brightness(value).enhance(rng.uniform(0.6, 1.4)),
        lambda value: ImageEnhance.Contrast(value).enhance(rng.uniform(0.6, 1.4)),
        lambda value: ImageEnhance.Color(value).enhance(rng.uniform(0.8, 1.2)),
    ]
    rng.shuffle(operations)
    for operation in operations:
        image = operation(image)
    return image


def to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(height, width, 3).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return tensor.sub_(mean).div_(std)


def augment(image: Image.Image, size: int, scale: tuple[float, float], rng: random.Random, blur: float, solarize: float) -> torch.Tensor:
    image = random_resized_crop(image, size, scale, rng)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.8:
        image = color_jitter(image, rng)
    if rng.random() < 0.2:
        image = ImageOps.grayscale(image).convert("RGB")
    if rng.random() < blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 2.0)))
    if rng.random() < solarize:
        image = ImageOps.solarize(image, threshold=128)
    return to_normalized_tensor(image)


class MultiCropDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        *,
        start_step: int,
        steps: int,
        batch_per_rank: int,
        rank: int,
        world_size: int,
        global_size: int,
        local_size: int,
        local_views: int,
        seed: int,
    ) -> None:
        self.rows = rows
        self.start_step = start_step
        self.steps = steps
        self.batch = batch_per_rank
        self.rank = rank
        self.world_size = world_size
        self.global_size = global_size
        self.local_size = local_size
        self.local_views = local_views
        self.seed = seed

    def __len__(self) -> int:
        return max(0, self.steps - self.start_step) * self.batch

    def __getitem__(self, local_index: int):
        step = self.start_step + local_index // self.batch
        within_rank = local_index % self.batch
        sample_id = step * self.world_size * self.batch + self.rank * self.batch + within_rank
        selection = hashlib.sha256(f"{self.seed}|image|{sample_id}".encode()).digest()
        row = self.rows[int.from_bytes(selection[:8], "big") % len(self.rows)]
        with Image.open(row["image_path"]) as handle:
            image = handle.convert("RGB").copy()
        base = int.from_bytes(hashlib.sha256(f"{self.seed}|augment|{sample_id}".encode()).digest()[:8], "big")
        global_a = augment(image, self.global_size, (0.32, 1.0), random.Random(base), 1.0, 0.0)
        global_b = augment(image, self.global_size, (0.32, 1.0), random.Random(base + 1), 0.1, 0.2)
        local = torch.stack(
            [
                augment(image, self.local_size, (0.05, 0.32), random.Random(base + 2 + index), 0.5, 0.0)
                for index in range(self.local_views)
            ]
        )
        return global_a, global_b, local, row["image_id"]


def load_manifest(path: Path, expected_images: int, *, rehash_images: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != expected_images or len({row["image_id"] for row in rows}) != expected_images:
        raise RuntimeError(f"the real-data pilot requires exactly {expected_images} unique images")
    for row in rows:
        image = Path(row["image_path"])
        valid = (
            row["split"] == "train"
            and image.is_file()
            and image.stat().st_size == row["image_bytes"]
            and (not rehash_images or sha256(image) == row["image_sha256"])
        )
        if not valid:
            raise RuntimeError(f"manifest integrity failure: {row['image_id']}")
    return rows


class DINOHead(nn.Module):
    def __init__(self, arguments: argparse.Namespace) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(arguments.width, arguments.head_hidden_dim),
            nn.GELU(),
            nn.Linear(arguments.head_hidden_dim, arguments.head_hidden_dim),
            nn.GELU(),
            nn.Linear(arguments.head_hidden_dim, arguments.bottleneck_dim),
        )
        self.mlp.apply(self._init_weights)
        self.prototype_weight = nn.Parameter(
            torch.empty(arguments.projection_dim, arguments.bottleneck_dim)
        )
        nn.init.trunc_normal_(self.prototype_weight, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.mlp(value)
        value = F.normalize(value.float(), dim=-1).to(value.dtype)
        prototype_weight = F.normalize(self.prototype_weight.float(), dim=-1).to(value.dtype)
        return F.linear(value, prototype_weight)


class DINOModel(nn.Module):
    def __init__(self, arguments: argparse.Namespace, activation_checkpointing: bool) -> None:
        super().__init__()
        self.backbone = VisionTransformer(
            arguments.global_size,
            arguments.patch_size,
            arguments.width,
            arguments.depth,
            arguments.heads,
            arguments.ffn_width,
            arguments.width,
            activation_checkpointing,
            arguments.checkpoint_every_n_blocks,
            arguments.layer_scale_init,
        )
        # The imported systems proxy has a terminal projection. Real DINO uses
        # the normalized CLS token as the backbone output and a separate MLP head.
        self.backbone.projection = nn.Identity()
        self.head = DINOHead(arguments)

    def forward(
        self,
        pixels: torch.Tensor | tuple[torch.Tensor, ...],
        return_features: bool = False,
    ):
        features = self.backbone(pixels)

        def project(value: torch.Tensor) -> torch.Tensor:
            return self.head(value)

        if isinstance(features, tuple):
            logits = tuple(project(value) for value in features)
        else:
            logits = project(features)
        return (logits, features) if return_features else logits


def shard_dino_model(
    model: DINOModel,
    *,
    reshard_after_forward: bool,
    compute_dtype: torch.dtype,
) -> None:
    policy = MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=torch.float32)
    for block in model.backbone.blocks:
        fully_shard(block, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    fully_shard(model.backbone, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mp_policy=policy, reshard_after_forward=reshard_after_forward)


def training_flops(arguments: argparse.Namespace) -> tuple[int, int]:
    backbone_arguments = copy(arguments)
    backbone_arguments.projection_dim = 0
    useful = useful_training_flops_per_source(backbone_arguments)
    hardware = hardware_training_flops_per_source(backbone_arguments)
    head_forward = 2 * (
        arguments.width * arguments.head_hidden_dim
        + arguments.head_hidden_dim * arguments.head_hidden_dim
        + arguments.head_hidden_dim * arguments.bottleneck_dim
        + arguments.bottleneck_dim * arguments.projection_dim
    )
    student_views = arguments.global_views + arguments.local_views
    teacher_views = arguments.global_views
    head_useful = 3 * student_views * head_forward + teacher_views * head_forward
    return useful + head_useful, hardware + head_useful


@torch.no_grad()
def sinkhorn_knopp_teacher(
    teacher_logits: torch.Tensor,
    temperature: float,
    iterations: int,
) -> torch.Tensor:
    world_size = dist.get_world_size()
    logits = teacher_logits.float()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    assignments = torch.exp(logits / temperature).transpose(0, 1)
    global_batch = assignments.shape[1] * world_size
    prototypes = assignments.shape[0]
    total = assignments.sum()
    dist.all_reduce(total)
    assignments /= total
    for _ in range(iterations):
        row_sum = assignments.sum(dim=1, keepdim=True)
        dist.all_reduce(row_sum)
        assignments /= row_sum.clamp_min(1e-12)
        assignments /= prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
        assignments /= global_batch
    return (assignments * global_batch).transpose(0, 1)


def dino_loss(
    teacher_logits: torch.Tensor,
    student_global: torch.Tensor,
    student_local: torch.Tensor,
    center: torch.Tensor,
    batch: int,
    global_views: int,
    local_views: int,
    teacher_temperature: float,
    student_temperature: float,
    teacher_assignment: str,
    sinkhorn_iterations: int,
) -> tuple[torch.Tensor, float, float]:
    if teacher_assignment == "sinkhorn_knopp":
        teacher = sinkhorn_knopp_teacher(
            teacher_logits,
            teacher_temperature,
            sinkhorn_iterations,
        )
    elif teacher_assignment == "centering":
        teacher = F.softmax((teacher_logits.float() - center) / teacher_temperature, dim=-1)
    else:
        raise RuntimeError(f"unknown teacher assignment: {teacher_assignment}")
    teacher = teacher.reshape(batch, global_views, -1).detach()
    student = torch.cat(
        (
            student_global.reshape(batch, global_views, -1),
            student_local.reshape(batch, local_views, -1),
        ),
        dim=1,
    )
    student = F.log_softmax(student.float() / student_temperature, dim=-1)
    total = student.new_zeros(())
    pairs = 0
    for teacher_view in range(global_views):
        for student_view in range(global_views + local_views):
            if student_view == teacher_view:
                continue
            total = total + torch.sum(-teacher[:, teacher_view] * student[:, student_view], dim=-1).mean()
            pairs += 1
    entropy = float((-(teacher * teacher.clamp_min(1e-12).log()).sum(dim=-1).mean()).item())
    maximum_probability = float(teacher.max(dim=-1).values.mean().item())
    return total / pairs, entropy, maximum_probability


def koleo_loss(
    global_features: torch.Tensor,
    batch: int,
    global_views: int,
) -> tuple[torch.Tensor, float]:
    features = F.normalize(global_features.float(), dim=-1).reshape(batch, global_views, -1)
    total = features.new_zeros(())
    mean_distance = 0.0
    for view in range(global_views):
        values = features[:, view]
        similarities = values @ values.transpose(0, 1)
        similarities.fill_diagonal_(-1.0)
        nearest_indices = similarities.max(dim=1).indices
        distances = F.pairwise_distance(values, values[nearest_indices], p=2, eps=1e-8)
        total = total - torch.log(distances + 1e-8).mean()
        mean_distance += float(distances.detach().mean().item())
    return total / global_views, mean_distance / global_views


@torch.no_grad()
def update_center(center: torch.Tensor, teacher_logits: torch.Tensor, momentum: float, world_size: int) -> None:
    batch_center = teacher_logits.float().mean(dim=0)
    if world_size > 1:
        dist.all_reduce(batch_center)
        batch_center /= world_size
    center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)


@torch.no_grad()
def update_teacher(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float) -> None:
    student_parameters = list(student.parameters())
    teacher_parameters = list(teacher.parameters())
    if len(student_parameters) != len(teacher_parameters):
        raise RuntimeError("student/teacher parameter mismatch")
    for student_parameter, teacher_parameter in zip(student_parameters, teacher_parameters, strict=True):
        teacher_parameter.mul_(momentum).add_(student_parameter.detach(), alpha=1.0 - momentum)


@torch.no_grad()
def clip_grad_norm_fsdp2(module: torch.nn.Module, maximum: float) -> torch.Tensor:
    """Clip the global norm of FSDP2 DTensor gradient shards."""
    squared = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float64)
    gradients = []
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
        gradients.append(gradient)
        squared.add_(local.detach().double().square().sum())
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    norm = squared.sqrt().float()
    coefficient = (maximum / (norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient)
    return norm


@torch.no_grad()
def unscale_gradients(module: torch.nn.Module, scale: float) -> None:
    if scale == 1.0:
        return
    inverse = 1.0 / scale
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(inverse)


def schedule(step: int, steps: int, warmup: int, maximum: float, minimum_ratio: float = 0.1) -> float:
    if step <= warmup:
        return maximum * step / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    return maximum * (minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def ema_schedule(step: int, steps: int, start: float, end: float) -> float:
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * step / max(1, steps)))


def cosine_transition(step: int, steps: int, start: float, end: float) -> float:
    """Move continuously from start to end over a bounded continuation stage."""
    progress = min(1.0, max(0.0, step / max(1, steps)))
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * progress))


def save_checkpoint(
    root: Path,
    step: int,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    center: torch.Tensor,
    metadata: dict,
    rank: int,
) -> Path:
    final = root / "checkpoints" / f"step-{step:06d}"
    temporary = final.with_name(final.name + ".tmp")
    if rank == 0:
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.parent.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    state = {
        "student": get_model_state_dict(student),
        "teacher": get_model_state_dict(teacher),
        "optimizer": get_optimizer_state_dict(student, optimizer),
        "center": center,
    }
    dcp.save(state, checkpoint_id=str(temporary))
    dist.barrier()
    if rank == 0:
        atomic_json(temporary / "metadata.json", {**metadata, "completed_step": step})
        if final.exists():
            shutil.rmtree(final)
        os.replace(temporary, final)
        latest = final.parent / "latest"
        latest_tmp = final.parent / ".latest.tmp"
        latest_tmp.unlink(missing_ok=True)
        latest_tmp.symlink_to(final.name)
        os.replace(latest_tmp, latest)
    dist.barrier()
    return final


def load_checkpoint(
    path: Path,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    center: torch.Tensor,
    metadata: dict,
    expected_parent_identity: dict | None = None,
) -> int:
    saved = json.loads((path / "metadata.json").read_text())
    expected = expected_parent_identity or metadata
    for key in (
        "manifest_sha256",
        "protocol_sha256",
        "source_sha256",
        "backbone_source_sha256",
        "acquisition_receipt_sha256",
    ):
        if saved.get(key) != expected.get(key):
            raise RuntimeError(f"resume identity mismatch for {key}")
    state = {
        "student": get_model_state_dict(student),
        "teacher": get_model_state_dict(teacher),
        "optimizer": get_optimizer_state_dict(student, optimizer),
        "center": center,
    }
    dcp.load(state, checkpoint_id=str(path))
    set_model_state_dict(student, state["student"])
    set_model_state_dict(teacher, state["teacher"])
    set_optimizer_state_dict(student, optimizer, state["optimizer"])
    center.copy_(state["center"])
    return int(saved["completed_step"])


def main() -> None:
    arguments = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("this pilot requires a distributed FSDP2 launch")
    if arguments.peak_dense_tflops_per_gpu <= 0:
        raise RuntimeError("peak dense throughput must be positive")
    if arguments.loss_scale <= 0:
        raise RuntimeError("loss scale must be positive")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    compute_dtype = torch.float16 if arguments.precision == "fp16" else torch.bfloat16
    effective_loss_scale = arguments.loss_scale if arguments.precision == "fp16" else 1.0
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "authorized_bounded_real_image_self_distillation_pilot":
        raise RuntimeError("authorized real-data pilot protocol required")
    objective = protocol["objective"]
    expected_objective_arguments = {
        "global_views": arguments.global_views,
        "local_views": arguments.local_views,
        "student_temperature": arguments.student_temperature,
        "teacher_temperature": arguments.teacher_temperature,
        "teacher_center_momentum": arguments.center_momentum,
        "teacher_ema_start": arguments.ema_start,
        "teacher_ema_end": arguments.ema_end,
        "bottleneck_dim": arguments.bottleneck_dim,
        "head_hidden_dim": arguments.head_hidden_dim,
        "weight_normalized_prototypes": arguments.projection_dim,
        "layer_scale_init": arguments.layer_scale_init,
        "learning_rate": arguments.learning_rate,
        "loss_scale": arguments.loss_scale,
    }
    for key, actual in expected_objective_arguments.items():
        if objective.get(key) != actual:
            raise RuntimeError(
                f"protocol/runtime objective mismatch for {key}: "
                f"{objective.get(key)} != {actual}"
            )
    if objective.get("precision") != "fp16_compute_fp32_state_static_loss_scale" or arguments.precision != "fp16":
        raise RuntimeError("this protocol requires the admitted FP16 precision recipe")
    if objective.get("adamw_betas") != [0.9, 0.999]:
        raise RuntimeError("this protocol requires the admitted AdamW betas")
    expected_images = int(protocol["source"]["expected_unique_images"])
    receipt = json.loads(arguments.acquisition_receipt.read_text())
    visual_audit_required = bool(
        protocol.get("promotion_gates", {}).get("fixed_visual_audit_required_before_training")
    )
    accepted_receipt_statuses = (
        {"PASS_FIXED_VISUAL_AUDIT"}
        if visual_audit_required
        else {"PASS", "PASS_FIXED_VISUAL_AUDIT"}
    )
    if receipt.get("status") not in accepted_receipt_statuses:
        raise RuntimeError(f"acquisition receipt is not admitted to training: {receipt.get('status')}")
    if receipt.get("protocol_sha256") != sha256(arguments.protocol):
        raise RuntimeError("acquisition receipt/protocol identity mismatch")
    if (
        receipt.get("manifest", {}).get("rows") != expected_images
        or receipt.get("manifest", {}).get("sha256") != sha256(arguments.manifest)
    ):
        raise RuntimeError("acquisition receipt/manifest identity mismatch")
    if visual_audit_required:
        audit_identity = receipt.get("visual_audit", {})
        audit_path = Path(audit_identity.get("path", ""))
        if not audit_path.is_absolute():
            audit_path = arguments.acquisition_receipt.parent / audit_path
        if not audit_path.is_file() or sha256(audit_path) != audit_identity.get("sha256"):
            raise RuntimeError("fixed visual audit file identity mismatch")
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("status") != "COMPLETE"
            or audit.get("decision") != "PASS_FOR_BOUNDED_SELF_SUPERVISED_RESEARCH_PILOT"
            or audit.get("protocol_sha256") != sha256(arguments.protocol)
            or audit.get("manifest", {}).get("rows") != expected_images
            or audit.get("manifest", {}).get("sha256") != sha256(arguments.manifest)
        ):
            raise RuntimeError("fixed visual audit decision/identity mismatch")
    rows = load_manifest(
        arguments.manifest,
        expected_images,
        rehash_images=arguments.rehash_images,
    )
    source_path = Path(__file__).resolve()
    identity = {
        "manifest_sha256": sha256(arguments.manifest),
        "protocol_sha256": sha256(arguments.protocol),
        "source_sha256": sha256(source_path),
        "backbone_source_sha256": sha256(Path(backbone_source.__file__).resolve()),
        "acquisition_receipt_sha256": sha256(arguments.acquisition_receipt),
    }
    continuation = None
    parent_identity = None
    continuation_start_step = 0
    if arguments.continuation_protocol is not None:
        if arguments.resume is None:
            raise RuntimeError("continuation protocol requires an exact parent checkpoint")
        continuation = json.loads(arguments.continuation_protocol.read_text())
        if continuation.get("status") != "AUTHORIZED_BOUNDED_STAGE2_CONTINUATION":
            raise RuntimeError("continuation protocol is not authorized")
        parent = continuation["parent_checkpoint"]
        stage = continuation["stage"]
        if arguments.resume.resolve() != Path(parent["path"]).resolve():
            raise RuntimeError("resume path does not match the frozen continuation parent")
        if sha256(arguments.resume / "metadata.json") != parent["metadata_sha256"]:
            raise RuntimeError("parent checkpoint metadata hash mismatch")
        receipt_path = Path(parent["sha256s_path"])
        if sha256(receipt_path) != parent["sha256s_sha256"]:
            raise RuntimeError("parent checkpoint receipt hash mismatch")
        parent_identity = parent["identity"]
        for key in (
            "manifest_sha256",
            "protocol_sha256",
            "backbone_source_sha256",
            "acquisition_receipt_sha256",
        ):
            if identity[key] != parent_identity[key]:
                raise RuntimeError(f"continuation data/objective identity mismatch for {key}")
        continuation_start_step = int(parent["completed_step"])
        expected_runtime = {
            "target_steps": arguments.steps,
            "batch_per_rank": arguments.batch_per_rank,
            "world_size": world_size,
            "save_every": arguments.save_every,
        }
        for key, actual in expected_runtime.items():
            if int(stage[key]) != actual:
                raise RuntimeError(f"continuation runtime mismatch for {key}")
        if not 0 < float(stage["learning_rate_end"]) <= float(stage["learning_rate_start"]):
            raise RuntimeError("invalid continuation learning-rate interval")
        if not 0 < float(stage["ema_momentum_start"]) <= float(stage["ema_momentum_end"]) <= 1:
            raise RuntimeError("invalid continuation EMA interval")
        identity.update(
            {
                "continuation_protocol_sha256": sha256(arguments.continuation_protocol),
                "parent_checkpoint_sha256s_sha256": parent["sha256s_sha256"],
            }
        )
    student = DINOModel(arguments, activation_checkpointing=True).to(device)
    teacher = DINOModel(arguments, activation_checkpointing=False).to(device)
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    if not 1_000_000_000 <= parameter_count <= 1_250_000_000:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    shard_dino_model(student, reshard_after_forward=True, compute_dtype=compute_dtype)
    shard_dino_model(teacher, reshard_after_forward=True, compute_dtype=compute_dtype)
    student.train()
    teacher.eval()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=arguments.learning_rate, betas=(0.9, 0.999),
        weight_decay=arguments.weight_decay, fused=True,
    )
    center = torch.zeros(arguments.projection_dim, device=device)
    completed_step = 0
    if arguments.resume is not None:
        completed_step = load_checkpoint(
            arguments.resume,
            student,
            teacher,
            optimizer,
            center,
            identity,
            expected_parent_identity=parent_identity,
        )
    if continuation is not None and completed_step != continuation_start_step:
        raise RuntimeError("parent checkpoint step does not match continuation start")
    run_end = min(arguments.steps, arguments.stop_after or arguments.steps)
    if completed_step >= run_end:
        raise RuntimeError(f"checkpoint step {completed_step} already reaches this run end {run_end}")

    student_run = student
    teacher_run = teacher
    if arguments.compile_mode != "none":
        student_run = torch.compile(student, mode=arguments.compile_mode)
        teacher_run = torch.compile(teacher, mode=arguments.compile_mode)

    dataset = MultiCropDataset(
        rows,
        start_step=completed_step,
        steps=run_end,
        batch_per_rank=arguments.batch_per_rank,
        rank=rank,
        world_size=world_size,
        global_size=arguments.global_size,
        local_size=arguments.local_size,
        local_views=arguments.local_views,
        seed=arguments.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_per_rank,
        shuffle=False,
        num_workers=arguments.workers_per_rank,
        pin_memory=True,
        persistent_workers=arguments.workers_per_rank > 0,
        prefetch_factor=2 if arguments.workers_per_rank > 0 else None,
        drop_last=True,
    )
    iterator = iter(loader)
    useful_flops, hardware_flops = training_flops(arguments)
    manifest = {
        "status": "running_real_image_dino_class_token_pilot",
        "architecture": "ViT-g/14",
        "student_parameters": parameter_count,
        "teacher_parameters": parameter_count,
        "world_size": world_size,
        "batch_per_rank": arguments.batch_per_rank,
        "global_batch": world_size * arguments.batch_per_rank,
        "steps": arguments.steps,
        "run_end_step": run_end,
        "start_step": completed_step,
        "global_views": arguments.global_views,
        "local_views": arguments.local_views,
        "unique_source_images": len(rows),
        "objective": "cross-view DINO class-token cross-entropy with EMA teacher",
        "teacher_assignment": protocol["objective"]["teacher_assignment"],
        "koleo_weight": float(protocol["objective"].get("koleo_weight", 0.0)),
        "image_byte_rehash_at_training_start": arguments.rehash_images,
        "precision": arguments.precision,
        "loss_scale": effective_loss_scale,
        "layer_scale_init": arguments.layer_scale_init,
        "head_hidden_dim": arguments.head_hidden_dim,
        "peak_dense_non_sparse_tflops_per_gpu": arguments.peak_dense_tflops_per_gpu,
        "device_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "mfu_definition": "useful dense matmul FLOPs divided by configured aggregate non-sparse FP16/BF16 Tensor Core peak",
        "bottleneck_dim": arguments.bottleneck_dim,
        "weight_normalized_prototypes": arguments.projection_dim,
        "full_ibot_objective": False,
        "compile_mode": arguments.compile_mode,
        "teacher_reshard_after_forward": True,
        "student_reshard_after_forward": True,
        "activation_checkpointing": True,
        "checkpoint_every_n_blocks": arguments.checkpoint_every_n_blocks,
        "checkpoint_format": "PyTorch distributed checkpoint plus identity metadata",
        "continuation_stage": continuation["stage"] if continuation is not None else None,
        "dtype": f"{arguments.precision.upper()} compute with FP32 parameters, reductions, optimizer state and loss",
        **identity,
        "claim_boundary": (
            continuation["claim_boundary"]
            if continuation is not None
            else protocol["claim_boundary"]
        ),
    }
    if rank == 0:
        atomic_json(arguments.output_dir / "manifest.json", manifest)
    metrics_path = arguments.output_dir / "metrics.jsonl"
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    run_started = time.perf_counter()
    for step in range(completed_step + 1, run_end + 1):
        iteration_started = time.perf_counter()
        global_a, global_b, local_crops, _image_ids = next(iterator)
        data_wait = time.perf_counter() - iteration_started
        global_pixels = torch.stack((global_a, global_b), dim=1).flatten(0, 1)
        local_pixels = local_crops.flatten(0, 1)
        global_pixels = global_pixels.to(device, dtype=compute_dtype, non_blocking=True)
        local_pixels = local_pixels.to(device, dtype=compute_dtype, non_blocking=True)
        compute_started = time.perf_counter()
        if continuation is None:
            learning_rate = schedule(
                step, arguments.steps, arguments.warmup_steps, arguments.learning_rate
            )
        else:
            stage = continuation["stage"]
            learning_rate = cosine_transition(
                step - continuation_start_step,
                arguments.steps - continuation_start_step,
                float(stage["learning_rate_start"]),
                float(stage["learning_rate_end"]),
            )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=compute_dtype):
            teacher_logits = teacher_run(global_pixels).float()
        with torch.autocast("cuda", dtype=compute_dtype):
            (student_global_raw, student_local_raw), (student_global_features, _student_local_features) = student_run(
                (global_pixels, local_pixels), return_features=True
            )
            student_global = student_global_raw.float()
            student_local = student_local_raw.float()
            cross_view_loss, teacher_entropy, teacher_max_probability = dino_loss(
                teacher_logits, student_global, student_local, center,
                arguments.batch_per_rank, arguments.global_views, arguments.local_views,
                arguments.teacher_temperature, arguments.student_temperature,
                protocol["objective"]["teacher_assignment"],
                int(protocol["objective"].get("sinkhorn_iterations", 3)),
            )
            representation_loss, feature_nearest_distance = koleo_loss(
                student_global_features,
                arguments.batch_per_rank,
                arguments.global_views,
            )
            koleo_weight = float(protocol["objective"].get("koleo_weight", 0.0))
            loss = cross_view_loss + koleo_weight * representation_loss
        (loss * effective_loss_scale).backward()
        unscale_gradients(student, effective_loss_scale)
        grad_norm = clip_grad_norm_fsdp2(student, arguments.gradient_clip)
        grad_is_finite = torch.tensor(
            int(torch.isfinite(grad_norm).item()), device=device, dtype=torch.int32
        )
        dist.all_reduce(grad_is_finite, op=dist.ReduceOp.MIN)
        if not bool(grad_is_finite.item()):
            raise RuntimeError(f"non-finite gradient norm before optimizer step {step}")
        optimizer.step()
        if continuation is None:
            momentum = ema_schedule(
                step, arguments.steps, arguments.ema_start, arguments.ema_end
            )
        else:
            stage = continuation["stage"]
            momentum = cosine_transition(
                step - continuation_start_step,
                arguments.steps - continuation_start_step,
                float(stage["ema_momentum_start"]),
                float(stage["ema_momentum_end"]),
            )
        update_teacher(student, teacher, momentum)
        if protocol["objective"]["teacher_assignment"] == "centering":
            update_center(center, teacher_logits, arguments.center_momentum, world_size)
        torch.cuda.synchronize(device)
        compute_seconds = time.perf_counter() - compute_started
        iteration_seconds = time.perf_counter() - iteration_started
        reduced = loss.detach().float()
        reduced_grad = grad_norm.detach().float() if isinstance(grad_norm, torch.Tensor) else torch.tensor(float(grad_norm), device=device)
        dist.all_reduce(reduced)
        dist.all_reduce(reduced_grad)
        reduced /= world_size
        reduced_grad /= world_size
        record = {
            "step": step,
            "loss": reduced.item(),
            "cross_view_loss": float(cross_view_loss.detach().item()),
            "koleo_loss": float(representation_loss.detach().item()),
            "feature_nearest_neighbor_distance": feature_nearest_distance,
            "finite": bool(torch.isfinite(reduced).item() and torch.isfinite(reduced_grad).item()),
            "teacher_entropy": teacher_entropy,
            "teacher_max_probability": teacher_max_probability,
            "teacher_logit_std": float(teacher_logits.std().item()),
            "teacher_cross_sample_logit_std": float(
                teacher_logits.reshape(arguments.batch_per_rank, arguments.global_views, -1)
                .std(dim=0)
                .mean()
                .item()
            ),
            "teacher_ema_momentum": momentum,
            "learning_rate": learning_rate,
            "gradient_norm_mean_ranks": reduced_grad.item(),
            "data_wait_seconds": data_wait,
            "compute_seconds": compute_seconds,
            "iteration_seconds": iteration_seconds,
            "end_to_end_source_images_per_second": arguments.batch_per_rank * world_size / iteration_seconds,
            "model_source_images_per_second": arguments.batch_per_rank * world_size / compute_seconds,
            "end_to_end_estimated_dense_mfu": useful_flops * arguments.batch_per_rank / iteration_seconds / (arguments.peak_dense_tflops_per_gpu * 1e12),
            "model_estimated_dense_mfu": useful_flops * arguments.batch_per_rank / compute_seconds / (arguments.peak_dense_tflops_per_gpu * 1e12),
            "model_estimated_dense_hfu": hardware_flops * arguments.batch_per_rank / compute_seconds / (arguments.peak_dense_tflops_per_gpu * 1e12),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "elapsed_seconds": time.perf_counter() - run_started,
        }
        if rank == 0:
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if not record["finite"]:
            raise RuntimeError(f"non-finite state at step {step}: {record}")
        minimum_feature_distance = float(
            protocol["promotion_gates"]["feature_nearest_neighbor_distance_minimum"]
        )
        if feature_nearest_distance <= minimum_feature_distance:
            raise RuntimeError(
                f"feature diversity gate failed at step {step}: "
                f"{feature_nearest_distance} <= {minimum_feature_distance}"
            )
        minimum_teacher_sample_std = float(
            protocol["promotion_gates"]["teacher_cross_sample_logit_std_minimum"]
        )
        if record["teacher_cross_sample_logit_std"] <= minimum_teacher_sample_std:
            raise RuntimeError(
                f"teacher cross-sample logit diversity gate failed at step {step}: "
                f"{record['teacher_cross_sample_logit_std']} <= {minimum_teacher_sample_std}"
            )
        entropy_ceiling = math.log(arguments.projection_dim) - float(
            protocol["promotion_gates"]["teacher_entropy_margin_below_log_prototypes"]
        )
        entropy_gate_active = (
            protocol["objective"]["teacher_assignment"] == "sinkhorn_knopp"
            or step >= arguments.warmup_steps
        )
        if entropy_gate_active and teacher_entropy >= entropy_ceiling:
            raise RuntimeError(
                f"teacher entropy collapse gate failed at step {step}: "
                f"{teacher_entropy} >= {entropy_ceiling}"
            )
        if step % arguments.save_every == 0 or step == run_end:
            save_checkpoint(arguments.output_dir, step, student, teacher, optimizer, center, identity, rank)

    if rank == 0:
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        current_records = [row for row in records if row["step"] > completed_step]
        measured = current_records[min(arguments.warmup_steps, len(current_records) - 1):]
        ordered = sorted(measured, key=lambda row: row["end_to_end_source_images_per_second"])
        median = ordered[len(ordered) // 2]
        promotion_gates = (
            continuation["promotion_gates"]
            if continuation is not None
            else protocol["promotion_gates"]
        )
        minimum_mfu = float(promotion_gates["median_end_to_end_mfu_minimum"])
        if median["end_to_end_estimated_dense_mfu"] < minimum_mfu:
            raise RuntimeError(
                "median end-to-end MFU gate failed: "
                f"{median['end_to_end_estimated_dense_mfu']} < {minimum_mfu}"
            )
        minimum_gradient = float(protocol["promotion_gates"]["gradient_norm_minimum"])
        if min(row["gradient_norm_mean_ranks"] for row in measured) <= minimum_gradient:
            raise RuntimeError("gradient norm promotion gate failed")
        peak_reserved = max(row["peak_memory_reserved_bytes"] for row in current_records)
        if continuation is not None:
            minimum_free_bytes = int(
                float(promotion_gates["minimum_free_gpu_memory_gib"]) * 1024**3
            )
            device_memory = torch.cuda.get_device_properties(device).total_memory
            if device_memory - peak_reserved < minimum_free_bytes:
                raise RuntimeError("continuation GPU memory-headroom gate failed")
        summary = {
            "status": "PASS",
            "target_steps": arguments.steps,
            "completed_steps": run_end,
            "measured_steps": len(measured),
            "initial_loss": current_records[0]["loss"],
            "final_loss": current_records[-1]["loss"],
            "median_end_to_end_source_images_per_second": median["end_to_end_source_images_per_second"],
            "median_model_source_images_per_second": median["model_source_images_per_second"],
            "median_end_to_end_estimated_dense_mfu": median["end_to_end_estimated_dense_mfu"],
            "median_model_estimated_dense_mfu": median["model_estimated_dense_mfu"],
            "peak_memory_reserved_bytes": peak_reserved,
            "latest_checkpoint": str(arguments.output_dir / "checkpoints" / "latest"),
            "claim_boundary": (
                continuation["claim_boundary"]
                if continuation is not None
                else protocol["claim_boundary"]
            ),
        }
        atomic_json(arguments.output_dir / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
