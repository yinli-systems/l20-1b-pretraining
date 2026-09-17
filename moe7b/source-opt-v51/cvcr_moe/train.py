"""DDP BF16 proxy pretraining with evidence receipts and fail-closed MFU gates."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import ModelConfig
from .data import PackedReader, verify_frozen_manifest
from .model import MoELanguageModel


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verify_digest_receipt(path: Path, receipt: Path) -> str:
    """Fail closed unless a one-line SHA-256 receipt matches ``path``."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if not receipt.is_file():
        raise FileNotFoundError(receipt)
    fields = receipt.read_text().strip().split()
    if len(fields) != 2 or fields[1] != path.name or len(fields[0]) != 64:
        raise ValueError(f"invalid checksum receipt: {receipt}")
    expected = fields[0].lower()
    if any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"invalid SHA-256 in receipt: {receipt}")
    observed = digest(path)
    if observed != expected:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    torch.save(value, temporary)
    os.replace(temporary, path)


def learning_rate_for_step(
    step: int, total: int, peak: float, warmup: int, decay: int
) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < total - decay:
        return peak
    progress = (step - (total - decay)) / decay
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--val-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--target-tokens", type=int, default=2_000_000_000)
    result.add_argument("--microbatch", type=int, default=1)
    result.add_argument("--accumulation", type=int, default=8)
    result.add_argument("--peak-lr", type=float, default=6e-4)
    result.add_argument("--warmup-tokens", type=int, default=20_000_000)
    result.add_argument("--max-steps", type=int)
    result.add_argument("--save-every", type=int, default=1_000)
    result.add_argument("--validate-every", type=int, default=250)
    result.add_argument("--validation-batches", type=int, default=4)
    result.add_argument(
        "--validation-microbatch",
        type=int,
        help="fixed per-rank validation batch; defaults to the training microbatch",
    )
    result.add_argument("--seed", type=int, default=20260916)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--compile", action="store_true")
    result.add_argument("--compile-mode", default="default")
    result.add_argument(
        "--ddp-before-compile",
        action="store_true",
        help="wrap the raw module in DDP before torch.compile for an opt-in overlap screen",
    )
    result.add_argument("--ddp-bucket-cap-mb", type=float, default=25.0)
    result.add_argument("--deterministic", action="store_true")
    result.add_argument("--no-save-final", action="store_true")
    # Retain the historical denominator so archived runs can be replayed
    # exactly, while emitting a separately versioned rated-hardware metric.
    result.add_argument("--dense-bf16-tflops-per-gpu", type=float, required=True)
    result.add_argument(
        "--rated-dense-bf16-tflops-per-gpu", type=float, default=165.2
    )
    result.add_argument("--minimum-active-mfu", type=float, default=0.0)
    result.add_argument("--mfu-grace-steps", type=int, default=5)
    result.add_argument("--mfu-window", type=int, default=10)
    return result


def main() -> None:
    arguments = parser().parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.set_num_threads(4)
    torch.manual_seed(arguments.seed)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    if arguments.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

    config = ModelConfig.from_json(arguments.config)
    if (
        config.cvcr_enabled
        and arguments.accumulation % config.cvcr_update_interval != 0
    ):
        raise ValueError(
            "gradient accumulation must be divisible by cvcr_update_interval"
        )
    data_manifest = verify_frozen_manifest(arguments.data_dir)
    validation_microbatch = arguments.validation_microbatch or arguments.microbatch
    raw_model = MoELanguageModel(config).cuda()
    deployed_parameters = sum(
        parameter.numel()
        for name, parameter in raw_model.named_parameters()
        if ".credit_predictor." not in name
    )
    training_parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    if deployed_parameters != config.deployed_parameter_count():
        raise RuntimeError(
            f"deployed parameter mismatch: {deployed_parameters} != {config.deployed_parameter_count()}"
        )
    if training_parameters != config.training_parameter_count():
        raise RuntimeError(
            f"training parameter mismatch: {training_parameters} != {config.training_parameter_count()}"
        )

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=arguments.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = arguments.output_dir / "resume.pt"
    train_reader = PackedReader(arguments.data_dir, arguments.seed, repeat=True)
    validation_reader = PackedReader(arguments.val_dir, arguments.seed)
    tokens_per_step = (
        arguments.microbatch
        * arguments.accumulation
        * world_size
        * config.max_position_embeddings
    )
    total_steps = max(1, arguments.target_tokens // tokens_per_step)
    if arguments.max_steps is not None:
        total_steps = min(total_steps, arguments.max_steps)
    target_tokens = total_steps * tokens_per_step
    warmup_steps = max(1, arguments.warmup_tokens // tokens_per_step)
    decay_steps = max(1, total_steps // 10)

    manifest_path = arguments.data_dir.parent / "manifest.json"
    code_directory = Path(__file__).parent
    code_hashes = {
        name: digest(code_directory / name)
        for name in ("train.py", "model.py", "cvcr.py", "config.py", "data.py")
    }
    manifest = {
        "status": "performance_screen" if arguments.max_steps else "proxy_training",
        "claim_boundary": (
            "A completed screen proves only runtime, loss finiteness, and measured throughput. "
            "It does not prove CVCR quality, 7B feasibility, or a released model."
        ),
        "model_config": config.as_dict(),
        "deployed_parameters": deployed_parameters,
        "active_parameters": config.active_parameter_count(),
        "training_parameters": training_parameters,
        "training_only_predictor_parameters": config.predictor_parameter_count(),
        "world_size": world_size,
        "microbatch": arguments.microbatch,
        "accumulation": arguments.accumulation,
        "validation_microbatch": validation_microbatch,
        "tokens_per_step": tokens_per_step,
        "target_tokens": target_tokens,
        "optimizer": "AdamW",
        "peak_lr": arguments.peak_lr,
        "seed": arguments.seed,
        "compile": arguments.compile,
        "compile_mode": arguments.compile_mode if arguments.compile else None,
        "ddp_before_compile": arguments.ddp_before_compile,
        "ddp_bucket_cap_mb": arguments.ddp_bucket_cap_mb,
        "data_directory": str(arguments.data_dir),
        "validation_directory": str(arguments.val_dir),
        "data_status": data_manifest["status"],
        "data_manifest_sha256": digest(manifest_path),
        "data_revision": data_manifest.get("revision"),
        "legacy_full_square_flops_per_token": raw_model.active_flops_per_token(
            config.max_position_embeddings
        ),
        "causal_matmul_flops_per_token": raw_model.causal_matmul_flops_per_token(
            config.max_position_embeddings
        ),
        "expected_probe_flops_per_token": raw_model.expected_probe_flops_per_token(),
        "legacy_dense_bf16_tflops_per_gpu": arguments.dense_bf16_tflops_per_gpu,
        "rated_dense_bf16_tflops_per_gpu": (
            arguments.rated_dense_bf16_tflops_per_gpu
        ),
        "mfu_accounting": {
            "legacy_reported_mfu": (
                "legacy full-square numerator divided by the archived launch peak"
            ),
            "rated_full_square_mfu": (
                "legacy full-square numerator divided by rated dense BF16 peak"
            ),
            "causal_useful_matmul_mfu": (
                "causal-pair useful matmuls divided by rated dense BF16 peak"
            ),
            "structured_sparsity_multiplier": False,
        },
        "code_sha256": code_hashes,
        "config_sha256": digest(arguments.config),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    fingerprint_fields = {
        key: manifest[key]
        for key in (
            "model_config",
            "deployed_parameters",
            "active_parameters",
            "world_size",
            "microbatch",
            "accumulation",
            "validation_microbatch",
            "tokens_per_step",
            "target_tokens",
            "optimizer",
            "peak_lr",
            "seed",
            "ddp_before_compile",
            "ddp_bucket_cap_mb",
            "data_manifest_sha256",
            "config_sha256",
        )
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["run_fingerprint_sha256"] = fingerprint

    step = 0
    if arguments.resume:
        resume_ok = torch.ones((), dtype=torch.int32, device="cuda")
        if rank == 0:
            try:
                verify_digest_receipt(
                    resume_path, arguments.output_dir / "resume.sha256"
                )
            except Exception as error:
                resume_ok.zero_()
                print(f"resume integrity verification failed: {error}", flush=True)
        dist.broadcast(resume_ok, src=0)
        if not resume_ok.item():
            raise RuntimeError("resume integrity verification failed on rank 0")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("run_fingerprint") != fingerprint:
            raise RuntimeError("resume fingerprint does not match the frozen run")
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        random.setstate(state["python_rng"][rank])
        np.random.set_state(state["numpy_rng"][rank])
        torch.set_rng_state(state["torch_rng"][rank])
        torch.cuda.set_rng_state(state["cuda_rng"][rank], local_rank)

    if arguments.ddp_before_compile:
        ddp_model = DDP(
            raw_model,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,
            bucket_cap_mb=arguments.ddp_bucket_cap_mb,
        )
        model = (
            torch.compile(ddp_model, mode=arguments.compile_mode)
            if arguments.compile
            else ddp_model
        )
    else:
        compiled_model = (
            torch.compile(raw_model, mode=arguments.compile_mode)
            if arguments.compile
            else raw_model
        )
        ddp_model = DDP(
            compiled_model,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,
            bucket_cap_mb=arguments.ddp_bucket_cap_mb,
        )
        model = ddp_model

    if rank == 0:
        atomic_json(manifest, arguments.output_dir / "run-manifest.json")
        launch = {
            "unix": time.time(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "resume": arguments.resume,
            "step_at_launch": step,
            "run_fingerprint_sha256": fingerprint,
            "code_sha256": code_hashes,
        }
        with (arguments.output_dir / "launches.jsonl").open("a") as handle:
            handle.write(json.dumps(launch, sort_keys=True) + "\n")

    metrics_path = arguments.output_dir / "metrics.jsonl"
    active_flops = raw_model.active_flops_per_token(config.max_position_embeddings)
    causal_flops = raw_model.causal_matmul_flops_per_token(
        config.max_position_embeddings
    )
    probe_flops = raw_model.expected_probe_flops_per_token()
    legacy_peak = world_size * arguments.dense_bf16_tflops_per_gpu * 1e12
    rated_peak = (
        world_size * arguments.rated_dense_bf16_tflops_per_gpu * 1e12
    )
    started = time.monotonic()
    active_mfu_samples: list[float] = []
    stop_requested = False
    stop_reason: str | None = None

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested, stop_reason
        stop_requested = True
        stop_reason = "signal"

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    while step < total_steps:
        iteration_started = time.monotonic()
        rate = learning_rate_for_step(
            step, total_steps, arguments.peak_lr, warmup_steps, decay_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device="cuda")
        ce_sum = torch.zeros((), device="cuda")
        router_sum = torch.zeros((), device="cuda")
        predictor_sum = torch.zeros((), device="cuda")
        for accumulation_step in range(arguments.accumulation):
            microstep = step * arguments.accumulation + accumulation_step
            # Put the synchronized (last) microbatch on the CVCR graph so DDP
            # reduces the predictor gradients accumulated under no_sync().
            # Earlier skipped microbatches can then be a true Top-2 hot path
            # with no dummy predictor reductions.
            cvcr_active = (
                (accumulation_step + 1) % config.cvcr_update_interval == 0
                or accumulation_step == arguments.accumulation - 1
            )
            raw_model.set_cvcr_active(cvcr_active)
            inputs, targets = train_reader.batch_for_step(
                microstep,
                rank,
                world_size,
                arguments.microbatch,
            )
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            synchronization = (
                ddp_model.no_sync()
                if accumulation_step < arguments.accumulation - 1
                else contextlib.nullcontext()
            )
            with synchronization, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, ce_loss, router_loss, predictor_loss = model(inputs, targets)
                scaled_loss = loss / arguments.accumulation
            scaled_loss.backward()
            loss_sum += scaled_loss.detach().float()
            ce_sum += ce_loss.float() / arguments.accumulation
            router_sum += router_loss.float() / arguments.accumulation
            predictor_sum += predictor_loss.float() / arguments.accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        step += 1
        torch.cuda.synchronize()

        reductions = torch.stack((loss_sum, ce_sum, router_sum, predictor_sum))
        dist.all_reduce(reductions)
        reductions /= world_size
        now = time.monotonic()
        duration = now - iteration_started
        tokens_per_second = tokens_per_step / duration
        legacy_reported_mfu = tokens_per_second * active_flops / legacy_peak
        legacy_hardware_work_mfu = (
            tokens_per_second * (active_flops + probe_flops) / legacy_peak
        )
        rated_full_square_mfu = tokens_per_second * active_flops / rated_peak
        causal_useful_matmul_mfu = tokens_per_second * causal_flops / rated_peak
        modeled_main_plus_probe_matmul_utilization = (
            tokens_per_second * (active_flops + probe_flops) / rated_peak
        )
        if rank == 0:
            record = {
                "step": step,
                "prediction_tokens": step * tokens_per_step,
                "loss": reductions[0].item(),
                "cross_entropy": reductions[1].item(),
                "router_auxiliary": reductions[2].item(),
                "predictor_loss": reductions[3].item(),
                "learning_rate": rate,
                "gradient_norm": gradient_norm.item(),
                "step_seconds": duration,
                "tokens_per_second": tokens_per_second,
                # Compatibility aliases remain replayable but are explicitly
                # labeled legacy in the same record.
                "active_mfu": legacy_reported_mfu,
                "legacy_reported_mfu": legacy_reported_mfu,
                "hardware_work_mfu_including_probes": legacy_hardware_work_mfu,
                "rated_full_square_mfu": rated_full_square_mfu,
                "causal_useful_matmul_mfu": causal_useful_matmul_mfu,
                "modeled_main_plus_probe_matmul_utilization": (
                    modeled_main_plus_probe_matmul_utilization
                ),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "elapsed_seconds": now - started,
                "unix": time.time(),
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if step > arguments.mfu_grace_steps:
                active_mfu_samples.append(legacy_reported_mfu)
                if (
                    arguments.minimum_active_mfu > 0
                    and len(active_mfu_samples) >= arguments.mfu_window
                    and statistics.median(active_mfu_samples[-arguments.mfu_window :])
                    < arguments.minimum_active_mfu
                ):
                    stop_requested = True
                    stop_reason = "active_mfu_below_minimum"

        stop_flag = torch.tensor(int(stop_requested), device="cuda")
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        stop_requested = bool(stop_flag.item())
        if rank == 0 and stop_requested and stop_reason is None:
            stop_reason = "peer_signal"

        if step % arguments.validate_every == 0 or step == total_steps:
            raw_model.eval()
            validation_values = []
            with torch.no_grad():
                for validation_step in range(arguments.validation_batches):
                    inputs, targets = validation_reader.batch_for_step(
                        validation_step,
                        rank,
                        world_size,
                        validation_microbatch,
                    )
                    inputs = inputs.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, value, _, _ = raw_model(inputs, targets)
                    validation_values.append(value.float())
            validation = torch.stack(validation_values).mean()
            dist.all_reduce(validation)
            validation /= world_size
            raw_model.train()
            if rank == 0:
                with metrics_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "validation_cross_entropy": validation.item(),
                                "validation_perplexity": math.exp(validation.item()),
                                "unix": time.time(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

        should_save = (
            (arguments.save_every > 0 and step % arguments.save_every == 0)
            or (step == total_steps and not arguments.no_save_final)
            or stop_requested
        )
        if should_save:
            rng = {
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(local_rank),
            }
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(rng, gathered, dst=0)
            if rank == 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "run_fingerprint": fingerprint,
                    **{key: [item[key] for item in gathered] for key in rng},
                }
                atomic_save(checkpoint, resume_path)
                (arguments.output_dir / "resume.sha256").write_text(
                    digest(resume_path) + "  resume.pt\n"
                )
            dist.barrier()
        if stop_requested:
            break

    if rank == 0:
        status = {
            "status": "CHECKPOINTED_STOP" if stop_requested else "COMPLETED",
            "reason": stop_reason,
            "step": step,
            "prediction_tokens": step * tokens_per_step,
            "unix": time.time(),
        }
        if resume_path.is_file():
            status["checkpoint_sha256"] = digest(resume_path)
        atomic_json(status, arguments.output_dir / "training-status.json")
        print(json.dumps(status, sort_keys=True), flush=True)
    exit_code = 65 if stop_reason == "active_mfu_below_minimum" else 0
    dist.destroy_process_group()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
