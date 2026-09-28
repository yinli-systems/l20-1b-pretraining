#!/usr/bin/env python3
"""Convert uint16 NumPy token shards into resumable LitData streams."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from litdata import optimize
from litdata.streaming import TokensLoader


def load_array(path: str) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.uint16 or array.ndim != 1:
        raise ValueError(f"bad token shard {path}: dtype={array.dtype}, shape={array.shape}")
    return np.asarray(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    inputs = [str(path) for path in sorted(args.input_dir.glob("*.npy"))]
    if not inputs:
        raise SystemExit(f"no .npy shards found in {args.input_dir}")
    optimize(
        load_array,
        inputs=inputs,
        output_dir=str(args.output_dir),
        chunk_bytes="64MB",
        num_workers=args.workers,
        item_loader=TokensLoader(),
        mode="overwrite",
        keep_data_ordered=True,
    )
    index = json.loads((args.output_dir / "index.json").read_text())
    total_tokens = sum(int(chunk["dim"]) for chunk in index["chunks"])
    print(json.dumps({"output_dir": str(args.output_dir), "chunks": len(index["chunks"]), "tokens": total_tokens}))
    # LitData's spawned worker bookkeeping can leave a Python 3.12 finalizer
    # thread alive on this host after all workers and files are complete.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
