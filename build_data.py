#!/usr/bin/env python3
"""Build the gate or full corpus sequentially with durable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/hhai/pretrain")
SOURCE_PRIORITY = {
    "gate": ("math", "code", "synthetic", "web"),
    # For a short, from-scratch run, follow the evidence-backed foundation mix:
    # 85% English web (FineWeb-Edu/DCLM 50/50), 12% code, and 3% math.
    "full": ("math", "code", "web", "dclm"),
}
TARGETS = {
    "gate": {"math": 32_000_000, "code": 11_886_249, "synthetic": 10_175_334, "web": 32_000_000},
    # A 1% unique-data reserve prevents random weighted sampling from exhausting
    # one source before the requested 20B prediction tokens have been consumed.
    "full": {"math": 606_000_000, "code": 2_424_000_000, "web": 8_585_000_000, "dclm": 8_585_000_000},
}
GATE_VALIDATION_TARGETS = {"code": 22_539, "synthetic": 6_147}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        banner = json.dumps({"time": time.time(), "command": command})
        print(banner, flush=True)
        log.write(banner + "\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def npy_tokens(path: Path) -> int:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.uint16 or array.ndim != 1 or array.size % 2049:
        raise RuntimeError(f"invalid token shard {path}: dtype={array.dtype}, shape={array.shape}")
    return int(array.size)


def packed_tokens(path: Path) -> int:
    index = json.loads((path / "index.json").read_text())
    return sum(int(chunk["dim"]) for chunk in index["chunks"])


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(TARGETS), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pack-workers", type=int, default=6)
    args = parser.parse_args()

    python = str(ROOT / ".venv/bin/python")
    project = ROOT / "src/hq-pretrain"
    npy_root = ROOT / "data/full-npy"
    packed_root = ROOT / "data/packed"
    manifests = ROOT / "manifests"
    stage_receipt: dict = {
        "stage": args.stage,
        "source_priority": SOURCE_PRIORITY[args.stage],
        "started_unix": time.time(),
        "sources": {},
    }

    for source in SOURCE_PRIORITY[args.stage]:
        target = TARGETS[args.stage][source]
        pack_command = [
            python,
            str(project / "pack_data.py"),
            "--source",
            source,
            "--tokenizer-dir",
            str(ROOT / "tokenizer"),
            "--output-dir",
            str(npy_root),
            "--dedup-db",
            str(manifests / "full-dedup.sqlite"),
            "--decontam-db",
            str(manifests / "decontam-13gram.sqlite"),
            "--source-state-db",
            str(manifests / "source-files.sqlite"),
            "--delete-completed-raw",
            "--target-tokens",
            str(target),
            "--holdout-modulus",
            "1000",
            "--workers",
            str(args.pack_workers),
            "--batch-docs",
            "1024",
            "--allow-existing-excess",
        ]
        if args.stage == "gate" and source in GATE_VALIDATION_TARGETS:
            pack_command.extend(["--validation-tokens", str(GATE_VALIDATION_TARGETS[source])])
        run_logged(pack_command, ROOT / f"logs/data-{args.stage}-{source}.log")

        source_manifest_path = npy_root / source / "manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        source_receipt = {
            "requested_train_tokens": target,
            "actual_train_tokens": int(source_manifest["train_tokens"]),
            "actual_validation_tokens": int(source_manifest["validation_tokens"]),
            "manifest_sha256": hash_file(source_manifest_path),
            "splits": {},
        }
        if source_receipt["actual_train_tokens"] < target // 2049 * 2049:
            raise RuntimeError(f"unexpected train token count for {source}: {source_receipt}")

        for npy_split, packed_split in (("train-npy", "train"), ("val-npy", "val")):
            input_dir = npy_root / source / npy_split
            output_dir = packed_root / source / packed_split
            source_npy_tokens = sum(npy_tokens(path) for path in sorted(input_dir.glob("*.npy")))
            existing_packed_tokens = None
            if (output_dir / "index.json").is_file():
                existing_packed_tokens = packed_tokens(output_dir)
            if existing_packed_tokens != source_npy_tokens:
                run_logged(
                    [
                        python,
                        str(project / "convert_litdata.py"),
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--workers",
                        str(args.workers),
                    ],
                    ROOT / f"logs/convert-{args.stage}-{source}-{packed_split}.log",
                )
            source_packed_tokens = packed_tokens(output_dir)
            if source_npy_tokens != source_packed_tokens:
                raise RuntimeError(
                    f"conversion mismatch for {source}/{packed_split}: "
                    f"npy={source_npy_tokens}, packed={source_packed_tokens}"
                )
            source_receipt["splits"][packed_split] = {
                "npy_tokens": source_npy_tokens,
                "packed_tokens": source_packed_tokens,
                "index_sha256": hash_file(output_dir / "index.json"),
            }

        stage_receipt["sources"][source] = source_receipt
        atomic_json(manifests / f"data-{args.stage}-receipt.json", stage_receipt)
        print(json.dumps({"completed_source": source, **source_receipt}, sort_keys=True), flush=True)

    stage_receipt["completed_unix"] = time.time()
    stage_receipt["total_train_tokens"] = sum(
        receipt["actual_train_tokens"] for receipt in stage_receipt["sources"].values()
    )
    stage_receipt["total_validation_tokens"] = sum(
        receipt["actual_validation_tokens"] for receipt in stage_receipt["sources"].values()
    )
    atomic_json(manifests / f"data-{args.stage}-receipt.json", stage_receipt)
    print(json.dumps(stage_receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}), file=sys.stderr, flush=True)
        raise
