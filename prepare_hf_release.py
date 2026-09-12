#!/usr/bin/env python3
"""Create and verify a safe, sharded Hugging Face release from the final export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {output}")
    if not (source / "pytorch_model.bin").is_file():
        raise FileNotFoundError(source / "pytorch_model.bin")

    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source, local_files_only=True, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        source_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
        if parameter_count != 1_100_048_384:
            raise ValueError(f"Unexpected parameter count: {parameter_count}")
        if source_dtypes != ["torch.float32"]:
            raise ValueError(f"Unexpected source dtypes: {source_dtypes}")

        model.config.torch_dtype = torch.float32
        model.save_pretrained(tmp, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(tmp)
        for name in ("model_config.yaml",):
            path = source / name
            if path.is_file():
                shutil.copy2(path, tmp / name)

        original = model.state_dict()
        seen: set[str] = set()
        shard_files = sorted(tmp.glob("*.safetensors"))
        if not shard_files:
            raise ValueError("No safetensors shards were written")
        for shard in shard_files:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in seen:
                        raise ValueError(f"Duplicate tensor in release: {name}")
                    candidate = handle.get_tensor(name)
                    reference = original[name]
                    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
                        raise ValueError(f"Tensor metadata mismatch: {name}")
                    if not torch.equal(candidate, reference):
                        raise ValueError(f"Tensor value mismatch: {name}")
                    seen.add(name)
        if seen != set(original):
            missing = sorted(set(original) - seen)
            extra = sorted(seen - set(original))
            raise ValueError(f"Tensor key mismatch; missing={missing[:5]} extra={extra[:5]}")

        files = {}
        for path in sorted(tmp.iterdir()):
            if path.is_file():
                files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "created_unix": time.time(),
            "source": str(source),
            "source_pytorch_model_bin": {
                "bytes": (source / "pytorch_model.bin").stat().st_size,
                "sha256": sha256(source / "pytorch_model.bin"),
            },
            "format": "Hugging Face Transformers sharded safetensors",
            "safe_serialization": True,
            "parameter_count": parameter_count,
            "stored_dtype": source_dtypes,
            "tensor_count": len(seen),
            "tensor_exact_equality_verified": True,
            "files": files,
        }
        atomic_json(tmp / "release-manifest.json", manifest)
        os.replace(tmp, output)
        print(json.dumps({
            "status": "complete",
            "output": str(output),
            "parameter_count": parameter_count,
            "tensor_count": len(seen),
            "shards": len(shard_files),
        }))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
