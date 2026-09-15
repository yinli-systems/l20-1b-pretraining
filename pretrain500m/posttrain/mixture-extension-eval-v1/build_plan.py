#!/usr/bin/env python3
"""Freeze R4 extension two-seed evaluation only after verified completion."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
JOBS = ((20260916, 1591570), (20260917, 1591571))
BASE_AGGREGATE_SHA256 = "c87aa54ec060154d594c6a7ca44b79b28cec27497cc867b0845ad492397535c1"
BASE_CHECKPOINT_SHA256 = "13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf"
PROTOCOL_SHA256 = "a71ce3c8cc85cac6fe6751cb46f7e50dfb78d266602d6377cb2e65249f11c761"
CACHE_MANIFEST_SHA256 = "699558cb80318870306eccf77fb1d200413bfdc03b6548fe41c9d39ed0044620"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def parse_slurm_states(text: str) -> dict[int, tuple[str, str]]:
    states = {}
    for line in text.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 3 and fields[0].isdigit():
            states[int(fields[0])] = (fields[1], fields[2])
    return states


def build(root: Path, output: Path, slurm_states: dict[int, tuple[str, str]]) -> dict:
    root = root.resolve(strict=True)
    if output.exists():
        raise ValueError("evaluation plan output already exists")
    for _, job in JOBS:
        if slurm_states.get(job) != ("COMPLETED", "0:0"):
            raise ValueError(f"extension job {job} has not completed successfully")
    expected = (
        (root / "posttrain/seven-task-confirmation-2gpu-v1/1590907/base/seven-task-aggregate.json", BASE_AGGREGATE_SHA256),
        (root / "formal/run-v5/resume.pt", BASE_CHECKPOINT_SHA256),
        (root / "source/protocol-v5.json", PROTOCOL_SHA256),
        (root / "artifacts/eval-hf-home/eval-cache-manifest.json", CACHE_MANIFEST_SHA256),
    )
    for path, value in expected:
        if digest(path) != value:
            raise ValueError(f"frozen evaluation input mismatch: {path}")

    candidates = []
    for seed, job in JOBS:
        run = root / "posttrain/mixture-extension-v1" / f"r4-lr6e5-t1b-seed{seed}-{job}"
        train = run / "train"
        status = json.loads((train / "training-status.json").read_text())
        manifest = json.loads((train / "run-manifest.json").read_text())
        if status.get("status") != "QUALIFICATION_COMPLETED" or status.get("mfu_window_passed") is not True:
            raise ValueError(f"extension job {job} did not pass qualification")
        if status.get("step") != 512 or status.get("tokens") != 1_073_741_824:
            raise ValueError(f"extension job {job} did not process the frozen target")
        if manifest.get("seed") != seed or manifest.get("world_size") != 4:
            raise ValueError(f"extension job {job} has unexpected seed or GPU identity")
        if manifest.get("initial_checkpoint_sha256") != BASE_CHECKPOINT_SHA256:
            raise ValueError(f"extension job {job} does not start from immutable Base")
        checkpoint = train / "model-final.pt"
        checkpoint_sha = digest(checkpoint)
        recorded = (train / "model-final.sha256").read_text().split()[0]
        if checkpoint_sha != recorded or checkpoint_sha != status.get("model_checkpoint_sha256"):
            raise ValueError(f"extension job {job} checkpoint SHA-256 mismatch")
        candidates.append({
            "id": f"r4-lr6e5-t1b-seed{seed}",
            "recipe": "R4_current_code_guard",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "expected_step": 512,
            "training_status_sha256": digest(train / "training-status.json"),
            "run_manifest_sha256": digest(train / "run-manifest.json"),
        })

    plan = {
        "schema": "p529m-r4-extension-capability-screen-v1",
        "status": "FROZEN_BEFORE_EXTENSION_CAPABILITY_SCREEN",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base": {
            "aggregate": str(expected[0][0]),
            "aggregate_sha256": BASE_AGGREGATE_SHA256,
            "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        },
        "candidates": candidates,
        "protocol": str(expected[2][0]),
        "protocol_id": "p500m-english-base-v5",
        "protocol_sha256": PROTOCOL_SHA256,
        "eval_cache_manifest": str(expected[3][0]),
        "eval_cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "execution": {
            "world_size": 2,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "dtype": "bfloat16",
            "batch_size": "auto",
        },
        "claim_boundary": "adaptive two-seed seven-task capability and retention screen; no automatic promotion or market-superiority claim",
    }
    output.mkdir()
    plan_path = output / "candidates.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "p529m-r4-extension-capability-plan-build-v1",
        "status": "PASS_TWO_VERIFIED_EXTENSION_CHECKPOINTS_FROZEN_FOR_EVALUATION",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": digest(plan_path),
        "checkpoint_sha256": {item["id"]: item["checkpoint_sha256"] for item in candidates},
        "slurm_states": {str(job): slurm_states[job] for _, job in JOBS},
        "evaluation_launched": False,
        "formal_promotion": False,
        "claim_boundary": plan["claim_boundary"],
    }
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run([
        "sacct", "-n", "-X", "-j", ",".join(str(job) for _, job in JOBS),
        "--format=JobIDRaw,State,ExitCode", "-P",
    ], check=True, capture_output=True, text=True)
    receipt = build(args.root, args.output, parse_slurm_states(result.stdout))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
