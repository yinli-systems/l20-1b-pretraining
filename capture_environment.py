#!/usr/bin/env python3
"""Capture a secret-free software/hardware provenance receipt."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/home/hhai/pretrain")


def output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    project = ROOT / "src/hq-pretrain"
    tokenizer = ROOT / "tokenizer"
    packages = {}
    for name in (
        "torch",
        "lightning",
        "litgpt",
        "litdata",
        "tokenizers",
        "transformers",
        "datasets",
        "pyarrow",
        "numpy",
        "boto3",
        "accelerate",
        "lm_eval",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_fields = output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).split(", ")
    receipt = {
        "captured_unix": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": {
            "name": gpu_fields[0],
            "uuid": gpu_fields[1],
            "driver": gpu_fields[2],
            "memory_mib": int(gpu_fields[3]),
        },
        "litgpt": {
            "commit": output(["git", "rev-parse", "HEAD"], ROOT / "src/litgpt"),
            "status": output(["git", "status", "--short"], ROOT / "src/litgpt"),
        },
        "project_sha256": {
            path.name: sha256(path)
            for path in sorted(project.glob("*.py"))
            if path.name != Path(__file__).name or path.is_file()
        },
        "tokenizer_sha256": {path.name: sha256(path) for path in sorted(tokenizer.iterdir()) if path.is_file()},
    }
    destination = ROOT / "manifests/environment-receipt.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
