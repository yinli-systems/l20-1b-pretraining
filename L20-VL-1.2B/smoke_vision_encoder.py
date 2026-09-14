#!/usr/bin/env python3
"""Load the frozen pinned SigLIP2 vision encoder and record a Stage-0 smoke test."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import SiglipVisionModel


MODEL_DIR = Path(
    "/home/hhai/l20-vl-1.2b/models/siglip2-base-patch16-224-75de2d55"
)
WEIGHTS = MODEL_DIR / "model.safetensors"
EXPECTED_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
BASE_WEIGHTS = Path("/home/hhai/pretrain/evaluations/final/hf/pytorch_model.bin")
EXPECTED_BASE_SHA256 = "bc7c43425cf538d5dcea333fdd1513faef0a994679acc08681505a879b3cd1f3"
EVIDENCE = Path("/home/hhai/l20-vl-1.2b/evidence/stage0-vision-smoke.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA unavailable")
    weights_sha = sha256(WEIGHTS)
    base_sha_before = sha256(BASE_WEIGHTS)
    if weights_sha != EXPECTED_SHA256:
        raise SystemExit(f"FAIL: SigLIP2 hash mismatch: {weights_sha}")
    if base_sha_before != EXPECTED_BASE_SHA256:
        raise SystemExit(f"FAIL: Base hash mismatch before smoke: {base_sha_before}")

    torch.manual_seed(20260913)
    torch.cuda.manual_seed_all(20260913)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = SiglipVisionModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).eval().cuda()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    pixels = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    durations: list[float] = []
    output = None
    with torch.inference_mode():
        for _ in range(3):
            output = model(pixel_values=pixels)
        torch.cuda.synchronize()
        for _ in range(20):
            start = time.perf_counter()
            output = model(pixel_values=pixels)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)

    assert output is not None
    hidden = output.last_hidden_state
    finite = bool(torch.isfinite(hidden).all().item())
    if hidden.shape != (1, 196, 768) or not finite:
        raise SystemExit(f"FAIL: output shape/finite check: {tuple(hidden.shape)}, {finite}")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SystemExit("FAIL: encoder is not fully frozen")

    base_sha_after = sha256(BASE_WEIGHTS)
    if base_sha_after != base_sha_before:
        raise SystemExit("FAIL: Base weights changed during vision smoke")

    sorted_durations = sorted(durations)
    payload = {
        "schema_version": "2026-09-13-v1",
        "status": "pass",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen_vision_encoder_forward_only_no_real_data",
        "training": False,
        "training_prediction_tokens": 0,
        "model": {
            "repo": "google/siglip2-base-patch16-224",
            "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            "weights_bytes": WEIGHTS.stat().st_size,
            "weights_sha256": weights_sha,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "dtype": str(next(model.parameters()).dtype),
        },
        "base_integrity": {
            "weights_sha256_before": base_sha_before,
            "weights_sha256_after": base_sha_after,
            "unchanged": base_sha_before == base_sha_after,
        },
        "input": {
            "shape": list(pixels.shape),
            "synthetic": True,
        },
        "output": {
            "last_hidden_state_shape": list(hidden.shape),
            "finite": finite,
        },
        "timing": {
            "load_seconds": load_seconds,
            "warmup_forwards": 3,
            "measured_forwards": len(durations),
            "median_forward_ms_batch_1": 1000 * sorted_durations[len(durations) // 2],
            "mean_forward_ms_batch_1": 1000 * sum(durations) / len(durations),
            "images_per_second_from_mean": 1 / (sum(durations) / len(durations)),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "claim_boundary": "Synthetic forward-only encoder timing excludes image decoding, preprocessing, connector, language decoder, data loading, and quality.",
    }
    if not all(math.isfinite(value) for value in durations):
        raise SystemExit("FAIL: non-finite timing")
    atomic_json(EVIDENCE, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
