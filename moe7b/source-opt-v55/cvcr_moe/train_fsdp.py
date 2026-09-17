"""FSDP2 BF16 pretraining for the frozen 7B dropless Top-2 baseline.

This path is intentionally separate from the proxy DDP trainer.  It shards
parameters, gradients, and Adam state across one data-parallel device mesh,
checkpoints every rank, and reports useful (non-recompute) MoE MFU.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .config import ModelConfig
from .data import (
    PackedReader,
    StageManifestPackedReader,
    verify_frozen_manifest,
    verify_stage_admission,
)
from .model import DecoderBlock, MoELanguageModel
from .train import atomic_json, digest, learning_rate_for_step


CHECKPOINT_NAME = re.compile(r"checkpoint-step-(\d{8})$")
RUN_ROLE_CLAIMS = {
    "bounded_feasibility_screen": (
        "A bounded screen proves only real 7B construction, forward/backward, optimizer, "
        "checkpoint/reconstruction, memory, and measured throughput. It is not a completed "
        "pretraining run."
    ),
    "systems_stability_pilot": (
        "This run uses the frozen pilot pack only for systems and stability qualification. "
        "Its tokens are outside the formal 150B budget, and it does not establish model quality."
    ),
    "formal_pretraining": (
        "This is an active 7B from-scratch Top-2 formal pretraining run on an admitted data "
        "pack; quality is not established until frozen held-out evaluations pass."
    ),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--val-dir", type=Path, required=True)
    result.add_argument(
        "--stage-manifest",
        type=Path,
        help="exact raw-uint16 stage manifest for admitted formal pretraining",
    )
    result.add_argument(
        "--data-admission",
        type=Path,
        help="fail-closed receipt admitting the exact --stage-manifest",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--target-tokens", type=int, required=True)
    result.add_argument(
        "--sequence-length",
        type=int,
        help="training sequence length; defaults to the model maximum",
    )
    result.add_argument("--microbatch", type=int, default=1)
    result.add_argument("--accumulation", type=int, default=8)
    result.add_argument("--peak-lr", type=float, default=4e-4)
    result.add_argument(
        "--bf16-optimizer-states",
        action="store_true",
        help="use the fused AdamW mixed path with FP32 parameters and BF16 moments",
    )
    result.add_argument("--warmup-tokens", type=int, default=200_000_000)
    result.add_argument("--max-steps", type=int)
    result.add_argument(
        "--run-role",
        choices=tuple(RUN_ROLE_CLAIMS),
        help=(
            "claim boundary recorded in the immutable run identity; defaults to a bounded "
            "screen when --max-steps is set and formal pretraining otherwise"
        ),
    )
    result.add_argument(
        "--steps-this-launch",
        type=int,
        help="checkpoint and exit after this many new optimizer steps without changing run identity",
    )
    result.add_argument("--log-every", type=int, default=10)
    result.add_argument("--checkpoint-every", type=int, default=1_000)
    result.add_argument("--keep-checkpoints", type=int, default=2)
    result.add_argument("--validate-every", type=int, default=1_000)
    result.add_argument("--validation-batches", type=int, default=4)
    result.add_argument("--seed", type=int, default=20260917)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--activation-checkpointing", action="store_true")
    result.add_argument(
        "--split-expert-fsdp",
        action="store_true",
        help="shard gate/up and down expert projections as independent nested FSDP units",
    )
    result.add_argument(
        "--split-embedding-fsdp",
        action="store_true",
        help="shard token embeddings independently so they stay resharded during expert backward",
    )
    result.add_argument(
        "--disable-backward-prefetch",
        action="store_true",
        help="disable overlapping the next FSDP all-gather with the current backward unit",
    )
    result.add_argument(
        "--sync-every-microbatch",
        action="store_true",
        help=(
            "reduce-scatter each microbatch and accumulate sharded gradients; "
            "uses more communication but avoids full unsharded FP32 gradient accumulation"
        ),
    )
    result.add_argument(
        "--keep-unsharded-between-microbatches",
        action="store_true",
        help=(
            "retain unsharded parameters after non-final accumulation backwards to avoid "
            "the next microbatch all-gather; requires per-microbatch gradient sync and "
            "substantially increases peak memory"
        ),
    )
    result.add_argument(
        "--keep-root-unsharded-after-forward",
        action="store_true",
        help="keep the root FSDP parameter group unsharded between forward and backward",
    )
    result.add_argument(
        "--reduce-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="FSDP reduce-scatter dtype; bfloat16 is a qualification-only numerical change",
    )
    result.add_argument(
        "--activation-checkpoint-offload-layers",
        type=int,
        default=0,
        help=(
            "number of early checkpointed block inputs to save on pinned host memory; "
            "use only when GPU memory qualification requires bounded extra headroom"
        ),
    )
    result.add_argument("--no-save-final", action="store_true")
    result.add_argument("--rated-dense-bf16-tflops-per-gpu", type=float, default=209.5)
    result.add_argument("--minimum-causal-mfu", type=float, default=0.0)
    result.add_argument("--mfu-grace-steps", type=int, default=3)
    result.add_argument("--mfu-window", type=int, default=5)
    result.add_argument(
        "--required-device",
        default="NVIDIA GeForce RTX 5090",
        help="exact CUDA device name; use the 4090 value only for a bounded preflight",
    )
    return result


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(32 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _checkpoint_payload_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.json", "METADATA.json"}
    )


def _distributed_file_manifest(root: Path, rank: int, world_size: int) -> dict[str, dict]:
    paths = _checkpoint_payload_files(root)
    local: dict[str, dict] = {}
    for index, path in enumerate(paths):
        if index % world_size != rank:
            continue
        local[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    gathered: list[dict[str, dict] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    combined: dict[str, dict] = {}
    for shard in gathered:
        if shard is None or set(combined).intersection(shard):
            raise RuntimeError("checkpoint hash assignment is incomplete or duplicated")
        combined.update(shard)
    if set(combined) != {str(path.relative_to(root)) for path in paths}:
        raise RuntimeError("checkpoint manifest does not cover every payload file")
    return dict(sorted(combined.items()))


def _verify_checkpoint_manifest(root: Path, rank: int, world_size: int) -> dict:
    manifest_path = root / "MANIFEST.json"
    metadata_path = root / "METADATA.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("checkpoint manifest or metadata is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "cvcr-moe-fsdp2-checkpoint-manifest-v1":
        raise ValueError("unexpected checkpoint manifest schema")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("checkpoint file manifest is empty")
    local_ok = 1
    for index, (relative, receipt) in enumerate(sorted(expected.items())):
        if index % world_size != rank:
            continue
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
            if (
                not path.is_file()
                or path.stat().st_size != receipt["bytes"]
                or _sha256(path) != receipt["sha256"]
            ):
                local_ok = 0
        except Exception:
            local_ok = 0
    flag = torch.tensor(local_ok, device="cuda", dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError("checkpoint payload integrity verification failed")
    return json.loads(metadata_path.read_text())


def save_checkpoint(
    *,
    model,
    optimizer,
    output_dir: Path,
    step: int,
    fingerprint: str,
    rank: int,
    world_size: int,
    keep_checkpoints: int,
) -> Path:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    checkpoints = output_dir / "checkpoints"
    name = f"checkpoint-step-{step:08d}"
    final = checkpoints / name
    temporary = checkpoints / f".incomplete-{name}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    if rank == 0:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if final.exists() or temporary.exists():
            raise FileExistsError(final if final.exists() else temporary)
        temporary.mkdir()
    dist.barrier()

    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    model_state = get_model_state_dict(model, options=options)
    dcp.save({"model": model_state}, checkpoint_id=temporary / "model")
    rank_state = {
        "schema": "cvcr-moe-fsdp2-rank-state-v1",
        "rank": rank,
        "world_size": world_size,
        "step": step,
        "run_fingerprint_sha256": fingerprint,
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(torch.cuda.current_device()),
    }
    torch.save(rank_state, temporary / f"rank-{rank:04d}.pt")
    dist.barrier()

    files = _distributed_file_manifest(temporary, rank, world_size)
    if rank == 0:
        metadata = {
            "schema": "cvcr-moe-fsdp2-checkpoint-metadata-v1",
            "step": step,
            "world_size": world_size,
            "run_fingerprint_sha256": fingerprint,
            "unix": time.time(),
        }
        atomic_json(metadata, temporary / "METADATA.json")
        atomic_json(
            {
                "schema": "cvcr-moe-fsdp2-checkpoint-manifest-v1",
                "files": files,
            },
            temporary / "MANIFEST.json",
        )
        os.replace(temporary, final)
        atomic_json(
            {
                "schema": "cvcr-moe-fsdp2-latest-v1",
                "checkpoint": name,
                "step": step,
                "run_fingerprint_sha256": fingerprint,
                "manifest_sha256": _sha256(final / "MANIFEST.json"),
                "metadata_sha256": _sha256(final / "METADATA.json"),
            },
            output_dir / "latest.json",
        )
        complete = sorted(
            path
            for path in checkpoints.iterdir()
            if path.is_dir() and CHECKPOINT_NAME.fullmatch(path.name)
        )
        for old in complete[:-keep_checkpoints]:
            if old.resolve().parent != checkpoints.resolve() or old == final:
                raise RuntimeError("checkpoint pruning target escaped its exact directory")
            shutil.rmtree(old)
    dist.barrier()
    return final


def load_checkpoint(
    *,
    model,
    optimizer,
    output_dir: Path,
    fingerprint: str,
    rank: int,
    world_size: int,
) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    latest_value = None
    if rank == 0:
        latest = json.loads((output_dir / "latest.json").read_text())
        name = latest.get("checkpoint", "")
        if not CHECKPOINT_NAME.fullmatch(name):
            raise ValueError("latest checkpoint name is unsafe")
        root = output_dir / "checkpoints" / name
        if _sha256(root / "MANIFEST.json") != latest.get("manifest_sha256"):
            raise RuntimeError("latest checkpoint manifest digest changed")
        if _sha256(root / "METADATA.json") != latest.get("metadata_sha256"):
            raise RuntimeError("latest checkpoint metadata digest changed")
        latest_value = str(root)
    objects = [latest_value]
    dist.broadcast_object_list(objects, src=0)
    root = Path(objects[0])
    metadata = _verify_checkpoint_manifest(root, rank, world_size)
    if (
        metadata.get("run_fingerprint_sha256") != fingerprint
        or metadata.get("world_size") != world_size
    ):
        raise RuntimeError("checkpoint does not match the frozen run identity")

    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    model_state = get_model_state_dict(model, options=options)
    dcp.load({"model": model_state}, checkpoint_id=root / "model")
    set_model_state_dict(model, model_state, options=options)
    rank_state = torch.load(root / f"rank-{rank:04d}.pt", weights_only=False)
    if (
        rank_state.get("schema") != "cvcr-moe-fsdp2-rank-state-v1"
        or rank_state.get("rank") != rank
        or rank_state.get("world_size") != world_size
        or rank_state.get("run_fingerprint_sha256") != fingerprint
        or rank_state.get("step") != metadata.get("step")
    ):
        raise RuntimeError("rank-local checkpoint identity mismatch")
    optimizer.load_state_dict(rank_state["optimizer"])
    random.setstate(rank_state["python_rng"])
    np.random.set_state(rank_state["numpy_rng"])
    torch.set_rng_state(rank_state["torch_rng"])
    torch.cuda.set_rng_state(rank_state["cuda_rng"], torch.cuda.current_device())
    dist.barrier()
    return int(metadata["step"])


def _precision_inventory(model, optimizer=None) -> dict:
    result = {
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "gradient_dtypes": sorted(
            {
                str(parameter.grad.dtype)
                for parameter in model.parameters()
                if parameter.grad is not None
            }
        ),
    }
    if optimizer is not None:
        result["optimizer_state_dtypes"] = sorted(
            {
                str(value.dtype)
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            }
        )
    return result


def register_bf16_adam_states(optimizer: torch.optim.Optimizer) -> None:
    """Initialize Adam moments in BF16 before AdamW's lazy state creation.

    ParaCloud's torch 2.11 fused Adam wrapper and underlying CUDA operator
    reject mixed-dtype tensor lists. The caller therefore uses the verified
    single-tensor AdamW path with FP32 master parameters, BF16 optimizer-input
    gradients and moments, and an FP32 scalar step counter.
    """

    def initialize_states(optim, _args, _kwargs) -> None:
        for group in optim.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = optim.state[parameter]
                if state:
                    continue
                state["step"] = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=(
                        parameter.device
                        if group.get("capturable") or group.get("fused")
                        else "cpu"
                    ),
                )
                state["exp_avg"] = torch.zeros_like(
                    parameter, dtype=torch.bfloat16, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter, dtype=torch.bfloat16, memory_format=torch.preserve_format
                )

    optimizer.register_step_pre_hook(initialize_states)


def prepare_bf16_adam_gradients(model: nn.Module) -> None:
    """Quantize already-reduced FP32 gradients for the BF16-moment update."""

    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        parameter.grad = None
        parameter.grad_dtype = torch.bfloat16
        parameter.grad = gradient.to(torch.bfloat16)


def clear_bf16_adam_gradients(model: nn.Module) -> None:
    """Restore the FP32 gradient contract for the next FSDP backward."""

    for parameter in model.parameters():
        if parameter.grad is None and parameter.grad_dtype == torch.float32:
            continue
        parameter.grad = None
        parameter.grad_dtype = torch.float32


def normalize_bf16_adam_states(optimizer: torch.optim.Optimizer) -> None:
    """Undo Optimizer.load_state_dict's automatic moment cast to param dtype."""

    for state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and value.dtype != torch.bfloat16:
                state[name] = value.to(torch.bfloat16)


def checkpoint_with_cpu_saved_tensors(function, *args, **kwargs):
    """Non-reentrant checkpoint whose saved forward inputs live on pinned CPU RAM."""

    def context_fn():
        return torch.autograd.graph.save_on_cpu(pin_memory=True), contextlib.nullcontext()

    return torch_checkpoint(
        function,
        *args,
        use_reentrant=False,
        context_fn=context_fn,
        **kwargs,
    )


def main() -> None:
    arguments = parser().parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2 or arguments.microbatch < 1 or arguments.accumulation < 1:
        raise ValueError("FSDP2 requires at least two ranks and positive batches")
    if arguments.target_tokens <= 0 or arguments.warmup_tokens <= 0:
        raise ValueError("token budgets must be positive")
    if arguments.log_every < 1 or (
        arguments.steps_this_launch is not None and arguments.steps_this_launch < 1
    ):
        raise ValueError("logging and per-launch step limits must be positive")
    if arguments.keep_checkpoints < 1:
        raise ValueError("at least one complete checkpoint must be retained")
    if arguments.activation_checkpoint_offload_layers < 0:
        raise ValueError("activation checkpoint offload layer count cannot be negative")
    if arguments.activation_checkpoint_offload_layers and not arguments.activation_checkpointing:
        raise ValueError("checkpoint input offload requires --activation-checkpointing")
    if arguments.keep_unsharded_between_microbatches and not arguments.sync_every_microbatch:
        raise ValueError(
            "keeping full parameters between microbatches requires per-microbatch gradient sync"
        )
    run_role = arguments.run_role or (
        "bounded_feasibility_screen" if arguments.max_steps else "formal_pretraining"
    )
    if arguments.max_steps is not None and run_role != "bounded_feasibility_screen":
        raise ValueError("--max-steps runs must retain the bounded feasibility claim boundary")
    if (arguments.stage_manifest is None) != (arguments.data_admission is None):
        raise ValueError("--stage-manifest and --data-admission must be supplied together")
    formal_stage_data = arguments.stage_manifest is not None
    if formal_stage_data and run_role == "systems_stability_pilot":
        raise ValueError("admitted stage-manifest data cannot be claimed as a systems pilot")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    torch.manual_seed(arguments.seed)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config = ModelConfig.from_json(arguments.config)
    if config.method != "top2" or config.expert_backend != "scattermoe":
        raise ValueError("the 7B FSDP2 baseline requires Top-2 ScatterMoE")
    if config.loss_backend != "liger" or config.router_dtype != "float32":
        raise ValueError("the 7B FSDP2 baseline requires Liger CE and FP32 routing")
    sequence_length = arguments.sequence_length or config.max_position_embeddings
    if not 1 <= sequence_length <= config.max_position_embeddings:
        raise ValueError("sequence length must be within the model context window")
    if arguments.activation_checkpoint_offload_layers > config.num_hidden_layers:
        raise ValueError("checkpoint input offload exceeds the model layer count")
    if config.vocab_size <= config.eos_token_id:
        raise ValueError("tokenizer/model vocabulary mismatch")
    if formal_stage_data:
        assert arguments.stage_manifest is not None
        assert arguments.data_admission is not None
        data_manifest = verify_stage_admission(
            arguments.stage_manifest, arguments.data_admission
        )
        train_reader = StageManifestPackedReader(
            arguments.stage_manifest,
            arguments.seed,
            sequence_length=sequence_length,
            pad_id=config.pad_token_id,
        )
        if arguments.target_tokens != train_reader.total_prediction_tokens:
            raise ValueError(
                "formal target tokens must exactly equal the admitted stage manifest"
            )
        manifest_path = arguments.stage_manifest
    else:
        data_manifest = verify_frozen_manifest(arguments.data_dir)
        train_reader = PackedReader(arguments.data_dir, arguments.seed, repeat=True)
        manifest_path = arguments.data_dir.parent / "manifest.json"
    validation_reader = PackedReader(arguments.val_dir, arguments.seed)
    tokens_per_step = (
        arguments.microbatch
        * arguments.accumulation
        * world_size
        * sequence_length
    )
    total_steps = max(
        1,
        math.ceil(arguments.target_tokens / tokens_per_step)
        if formal_stage_data
        else arguments.target_tokens // tokens_per_step,
    )
    if arguments.max_steps is not None:
        total_steps = min(total_steps, arguments.max_steps)
    target_tokens = (
        train_reader.prediction_tokens_before_sequences(
            total_steps
            * arguments.accumulation
            * world_size
            * arguments.microbatch
        )
        if formal_stage_data
        else total_steps * tokens_per_step
    )
    warmup_steps = max(1, arguments.warmup_tokens // tokens_per_step)
    decay_steps = max(1, total_steps // 10)

    code_root = Path(__file__).resolve().parent
    code_hashes = {
        name: digest(code_root / name)
        for name in ("train_fsdp.py", "model.py", "config.py", "data.py")
    }
    identity = {
        "model_config": config.as_dict(),
        "deployed_parameters": config.deployed_parameter_count(),
        "active_parameters": config.active_parameter_count(),
        "world_size": world_size,
        "microbatch": arguments.microbatch,
        "accumulation": arguments.accumulation,
        "tokens_per_step": tokens_per_step,
        "sequence_length": sequence_length,
        "target_tokens": target_tokens,
        "peak_lr": arguments.peak_lr,
        "warmup_tokens": arguments.warmup_tokens,
        "seed": arguments.seed,
        "activation_checkpointing": arguments.activation_checkpointing,
        "split_expert_fsdp": arguments.split_expert_fsdp,
        "split_embedding_fsdp": arguments.split_embedding_fsdp,
        "backward_prefetch": (
            "self_noop" if arguments.disable_backward_prefetch else "default_next_module"
        ),
        "gradient_sync": (
            "every_microbatch"
            if arguments.sync_every_microbatch
            else "last_microbatch"
        ),
        "keep_unsharded_between_microbatches": (
            arguments.keep_unsharded_between_microbatches
        ),
        "root_reshard_after_forward": (
            not arguments.keep_root_unsharded_after_forward
        ),
        "reduce_dtype": arguments.reduce_dtype,
        "activation_checkpoint_offload_layers": (
            arguments.activation_checkpoint_offload_layers
        ),
        "bf16_optimizer_states": arguments.bf16_optimizer_states,
        "optimizer_implementation": (
            "single_tensor_bf16_moments"
            if arguments.bf16_optimizer_states
            else "fused_fp32_moments"
        ),
        "optimizer_input_gradient_dtype": (
            "bfloat16" if arguments.bf16_optimizer_states else "float32"
        ),
        "run_role": run_role,
        "data_manifest_sha256": digest(manifest_path),
        "data_admission_sha256": (
            digest(arguments.data_admission) if arguments.data_admission else None
        ),
        "formal_stage_data": formal_stage_data,
        "config_sha256": digest(arguments.config),
        "code_sha256": code_hashes,
        "scattermoe_commit": os.environ.get("SCATTERMOE_COMMIT"),
        "liger_wheel_sha256": os.environ.get("LIGER_WHEEL_SHA256"),
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    if rank == 0:
        if arguments.resume:
            if not (arguments.output_dir / "latest.json").is_file():
                raise FileNotFoundError("--resume requested without latest.json")
        else:
            if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
                raise FileExistsError("refusing to start in a non-empty output directory")
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "event": "constructing_7b_model",
                    "deployed_parameters": config.deployed_parameter_count(),
                    "active_parameters": config.active_parameter_count(),
                    "world_size": world_size,
                }
            ),
            flush=True,
        )
    dist.barrier()

    # Build on host RAM, then let bottom-up FSDP2 move and shard one block at a
    # time.  Moving the unsharded 28.3 GB FP32 model to a 32 GB GPU first would
    # create an avoidable and unsafe initialization peak.
    raw_model = MoELanguageModel(config)
    observed_parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    if observed_parameters != config.deployed_parameter_count():
        raise RuntimeError("materialized 7B parameter count changed")
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda", (world_size,))
    reduce_dtype = (
        torch.float32 if arguments.reduce_dtype == "float32" else torch.bfloat16
    )
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=reduce_dtype,
        output_dtype=torch.bfloat16,
    )
    if arguments.split_embedding_fsdp:
        fully_shard(
            raw_model.embedding,
            mesh=mesh,
            mp_policy=mixed_precision,
            reshard_after_forward=True,
        )
    for index, layer in enumerate(list(raw_model.layers)):
        if not isinstance(layer, DecoderBlock):
            raise TypeError("unexpected decoder block type")
        if arguments.split_expert_fsdp:
            for projection in layer.moe.experts.fsdp_projection_units():
                fully_shard(
                    projection,
                    mesh=mesh,
                    mp_policy=mixed_precision,
                    reshard_after_forward=True,
                )
        if arguments.activation_checkpointing:
            if index < arguments.activation_checkpoint_offload_layers:
                layer = checkpoint_wrapper(
                    layer,
                    checkpoint_fn=checkpoint_with_cpu_saved_tensors,
                )
            else:
                layer = checkpoint_wrapper(
                    layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                )
            raw_model.layers[index] = layer
        fully_shard(layer, mesh=mesh, mp_policy=mixed_precision, reshard_after_forward=True)
    fully_shard(
        raw_model,
        mesh=mesh,
        mp_policy=mixed_precision,
        reshard_after_forward=not arguments.keep_root_unsharded_after_forward,
    )
    if arguments.disable_backward_prefetch:
        for module in raw_model.modules():
            if isinstance(module, FSDPModule):
                # In torch 2.11 an empty explicit list means "fall back to the
                # default next-module prefetch".  Pointing each module at
                # itself suppresses that default and unshard() is a no-op
                # because the current module is already materialized.
                module.set_modules_to_backward_prefetch([module])
    gc.collect()
    torch.cuda.empty_cache()
    model = raw_model
    optimizer_kwargs = {
        "lr": arguments.peak_lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    if arguments.bf16_optimizer_states:
        optimizer_kwargs.update({"fused": False, "foreach": False})
    else:
        optimizer_kwargs.update({"fused": True})
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    if arguments.bf16_optimizer_states:
        register_bf16_adam_states(optimizer)
    step = (
        load_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=arguments.output_dir,
            fingerprint=fingerprint,
            rank=rank,
            world_size=world_size,
        )
        if arguments.resume
        else 0
    )
    if arguments.resume and arguments.bf16_optimizer_states:
        normalize_bf16_adam_states(optimizer)

    devices: list[dict | None] = [None] * world_size
    descriptor = {
        "rank": rank,
        "hostname": os.uname().nodename,
        "device": torch.cuda.get_device_name(local_rank),
        "memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
    }
    dist.all_gather_object(devices, descriptor)
    if any(row is None or row["device"] != arguments.required_device for row in devices):
        raise RuntimeError(
            f"the 7B launch is not running exclusively on {arguments.required_device} GPUs"
        )

    active_flops = model.active_flops_per_token(sequence_length)
    causal_flops = model.causal_matmul_flops_per_token(sequence_length)
    peak = world_size * arguments.rated_dense_bf16_tflops_per_gpu * 1e12
    run_manifest = {
        "schema": "cvcr-moe-7b-fsdp2-run-v1",
        "status": run_role,
        "claim_boundary": RUN_ROLE_CLAIMS[run_role],
        **identity,
        "run_fingerprint_sha256": fingerprint,
        "total_steps": total_steps,
        "active_flops_per_token_full_square": active_flops,
        "causal_useful_matmul_flops_per_token": causal_flops,
        "rated_dense_bf16_tflops_per_gpu": arguments.rated_dense_bf16_tflops_per_gpu,
        "mfu_structured_sparsity_multiplier": False,
        "parallelism": {
            "type": "FSDP2_FULL_SHARD",
            "data_parallel_degree": world_size,
            "expert_parallel_degree": 1,
            "tensor_parallel_degree": 1,
            "pipeline_parallel_degree": 1,
            "reshard_after_forward": True,
            "root_reshard_after_forward": (
                not arguments.keep_root_unsharded_after_forward
            ),
            "keep_unsharded_between_microbatches": (
                arguments.keep_unsharded_between_microbatches
            ),
            "activation_checkpointing": arguments.activation_checkpointing,
            "split_expert_fsdp": arguments.split_expert_fsdp,
            "split_embedding_fsdp": arguments.split_embedding_fsdp,
            "backward_prefetch": (
                "self_noop"
                if arguments.disable_backward_prefetch
                else "default_next_module"
            ),
            "activation_checkpoint_offload_layers": (
                arguments.activation_checkpoint_offload_layers
            ),
            "required_device": arguments.required_device,
            "param_compute_dtype": "bfloat16",
            "master_parameter_dtype": "float32",
            "optimizer_moment_dtype": (
                "bfloat16" if arguments.bf16_optimizer_states else "float32"
            ),
            "reduce_dtype": arguments.reduce_dtype,
        },
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "devices": devices,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "data_status": data_manifest["status"],
        "data_revision": data_manifest.get("revision"),
    }
    if rank == 0:
        atomic_json(run_manifest, arguments.output_dir / "run-manifest.json")
        with (arguments.output_dir / "launches.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "unix": time.time(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "resume": arguments.resume,
                        "step_at_launch": step,
                        "run_fingerprint_sha256": fingerprint,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    stop_requested = False
    stop_reason: str | None = None

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested, stop_reason
        stop_requested = True
        stop_reason = "signal"

    for signal_name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), request_stop)

    model.train()
    metrics_path = arguments.output_dir / "metrics.jsonl"
    started = time.monotonic()
    logging_started = started
    logging_step = step
    def prediction_tokens_after_step(completed_step: int) -> int:
        if formal_stage_data:
            return train_reader.prediction_tokens_before_sequences(
                completed_step
                * arguments.accumulation
                * world_size
                * arguments.microbatch
            )
        return completed_step * tokens_per_step

    logging_prediction_tokens = prediction_tokens_after_step(step)
    launch_start_step = step
    causal_mfu_samples: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    while step < total_steps:
        rate = learning_rate_for_step(
            step, total_steps, arguments.peak_lr, warmup_steps, decay_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        ce_sum = torch.zeros((), device=device, dtype=torch.float64)
        router_sum = torch.zeros((), device=device, dtype=torch.float64)
        microbatches = []
        local_step_predictions = 0
        for accumulation_step in range(arguments.accumulation):
            microstep = step * arguments.accumulation + accumulation_step
            if formal_stage_data:
                inputs, targets, local_predictions = train_reader.batch_for_step(
                    microstep, rank, world_size, arguments.microbatch
                )
            else:
                inputs, targets = train_reader.batch_for_step(
                    microstep, rank, world_size, arguments.microbatch
                )
                local_predictions = arguments.microbatch * sequence_length
            microbatches.append((inputs, targets, local_predictions))
            local_step_predictions += local_predictions
        global_step_predictions = torch.tensor(
            local_step_predictions, device=device, dtype=torch.int64
        )
        dist.all_reduce(global_step_predictions)
        if not 0 < int(global_step_predictions) <= tokens_per_step:
            raise RuntimeError("optimizer step has an invalid prediction-token count")
        for accumulation_step, (inputs, targets, local_predictions) in enumerate(
            microbatches
        ):
            inputs = inputs[:, :sequence_length]
            targets = targets[:, :sequence_length]
            inputs = inputs.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
            model.set_reshard_after_backward(
                not (
                    arguments.keep_unsharded_between_microbatches
                    and accumulation_step + 1 < arguments.accumulation
                )
            )
            model.set_requires_gradient_sync(
                arguments.sync_every_microbatch
                or accumulation_step + 1 == arguments.accumulation
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, ce_loss, router_loss, _ = model(
                    inputs,
                    targets,
                    all_targets_ignored=local_predictions == 0,
                )
                loss_weight = (
                    world_size * local_predictions / int(global_step_predictions)
                )
                scaled_loss = loss * loss_weight
            scaled_loss.backward()
            loss_sum.add_(scaled_loss.detach().double())
            ce_sum.add_(ce_loss.double() * loss_weight)
            router_sum.add_(router_loss.double() * loss_weight)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        if arguments.bf16_optimizer_states:
            prepare_bf16_adam_gradients(model)
        optimizer.step()
        if arguments.bf16_optimizer_states:
            clear_bf16_adam_gradients(model)
        step += 1

        should_log = (
            step == 1
            or step % arguments.log_every == 0
            or step == total_steps
            or (
                arguments.steps_this_launch is not None
                and step - launch_start_step >= arguments.steps_this_launch
            )
        )
        if should_log:
            torch.cuda.synchronize(device)
            reductions = torch.stack((loss_sum, ce_sum, router_sum))
            dist.all_reduce(reductions)
            reductions /= world_size
            duration = torch.tensor(time.monotonic() - logging_started, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            interval_steps = step - logging_step
            seconds = float(duration)
            current_prediction_tokens = prediction_tokens_after_step(step)
            interval_prediction_tokens = (
                current_prediction_tokens - logging_prediction_tokens
            )
            tokens_per_second = interval_prediction_tokens / seconds
            causal_mfu = tokens_per_second * causal_flops / peak
            full_square_mfu = tokens_per_second * active_flops / peak
            peak_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(device),
                    torch.cuda.max_memory_reserved(device),
                ],
                device=device,
                dtype=torch.int64,
            )
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            record = {
                "step": step,
                "prediction_tokens": current_prediction_tokens,
                "loss": float(reductions[0]),
                "cross_entropy": float(reductions[1]),
                "router_auxiliary": float(reductions[2]),
                "learning_rate": rate,
                "gradient_norm": float(gradient_norm),
                "measurement_steps": interval_steps,
                "step_seconds_rank_max": seconds / interval_steps,
                "tokens_per_second": tokens_per_second,
                "rated_full_square_mfu": full_square_mfu,
                "causal_useful_matmul_mfu": causal_mfu,
                "peak_memory_allocated_bytes_rank_max": int(peak_memory[0]),
                "peak_memory_reserved_bytes_rank_max": int(peak_memory[1]),
                "elapsed_seconds": time.monotonic() - started,
                "unix": time.time(),
            }
            if step == 1:
                record["precision_inventory"] = _precision_inventory(model, optimizer)
            if rank == 0:
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if step > arguments.mfu_grace_steps:
                    causal_mfu_samples.append(causal_mfu)
                    if (
                        arguments.minimum_causal_mfu > 0
                        and len(causal_mfu_samples) >= arguments.mfu_window
                        and statistics.median(causal_mfu_samples[-arguments.mfu_window :])
                        < arguments.minimum_causal_mfu
                    ):
                        stop_requested = True
                        stop_reason = "causal_mfu_below_minimum"
            logging_started = time.monotonic()
            logging_step = step
            logging_prediction_tokens = current_prediction_tokens
            torch.cuda.reset_peak_memory_stats(device)

        stop_flag = torch.tensor(int(stop_requested), device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        stop_requested = bool(stop_flag.item())
        if rank == 0 and stop_requested and stop_reason is None:
            stop_reason = "peer_signal"

        launch_segment_complete = (
            arguments.steps_this_launch is not None
            and step - launch_start_step >= arguments.steps_this_launch
            and step < total_steps
        )

        if arguments.validate_every > 0 and (
            step % arguments.validate_every == 0 or step == total_steps
        ):
            model.eval()
            values = []
            with torch.no_grad():
                for validation_step in range(arguments.validation_batches):
                    inputs, targets = validation_reader.batch_for_step(
                        validation_step, rank, world_size, arguments.microbatch
                    )
                    inputs = inputs[:, :sequence_length]
                    targets = targets[:, :sequence_length]
                    inputs = inputs.pin_memory().to(device, non_blocking=True)
                    targets = targets.pin_memory().to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, value, _, _ = model(inputs, targets)
                    values.append(value.double())
            validation = torch.stack(values).mean()
            dist.all_reduce(validation)
            validation /= world_size
            model.train()
            if rank == 0:
                with metrics_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "validation_cross_entropy": float(validation),
                                "validation_perplexity": math.exp(float(validation)),
                                "unix": time.time(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

        should_save = (
            (arguments.checkpoint_every > 0 and step % arguments.checkpoint_every == 0)
            or (step == total_steps and not arguments.no_save_final)
            or launch_segment_complete
            or stop_requested
        )
        if should_save:
            checkpoint = save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=arguments.output_dir,
                step=step,
                fingerprint=fingerprint,
                rank=rank,
                world_size=world_size,
                keep_checkpoints=arguments.keep_checkpoints,
            )
            if rank == 0:
                print(
                    json.dumps(
                        {"event": "checkpoint_committed", "step": step, "path": str(checkpoint)}
                    ),
                    flush=True,
                )
        if stop_requested:
            break
        if launch_segment_complete:
            break

    if rank == 0:
        status = {
            "status": (
                "CHECKPOINTED_STOP"
                if stop_requested
                else "CHECKPOINTED_SEGMENT"
                if step < total_steps
                else "COMPLETED"
            ),
            "reason": stop_reason or ("steps_this_launch" if step < total_steps else None),
            "step": step,
            "prediction_tokens": prediction_tokens_after_step(step),
            "unix": time.time(),
        }
        atomic_json(status, arguments.output_dir / "training-status.json")
        print(json.dumps(status, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if stop_reason == "causal_mfu_below_minimum":
        raise SystemExit(65)


if __name__ == "__main__":
    main()
