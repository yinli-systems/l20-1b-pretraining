#!/usr/bin/env python3
"""Persist a balanced, pinned corpus sample for tokenizer training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import zstandard as zstd

from source_docs import REVISIONS, iter_source


DEFAULT_CHAR_BUDGETS = {"web": 130_000_000, "synthetic": 10_000_000, "math": 40_000_000, "code": 20_000_000}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"seed": args.seed, "revisions": REVISIONS, "sources": {}}
    for source, base_budget in DEFAULT_CHAR_BUDGETS.items():
        budget = int(base_budget * args.scale)
        path = args.output_dir / f"{source}.jsonl.zst"
        chars = docs = 0
        digest = hashlib.sha256()
        compressor = zstd.ZstdCompressor(level=9, threads=2)
        with path.open("wb") as raw, compressor.stream_writer(raw) as compressed:
            for row in iter_source(source, seed=args.seed):
                encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                compressed.write(encoded)
                digest.update(encoded)
                chars += len(row["text"])
                docs += 1
                if docs % 1000 == 0:
                    print(json.dumps({"source": source, "docs": docs, "chars": chars}), flush=True)
                if chars >= budget:
                    break
        manifest["sources"][source] = {
            "path": str(path), "documents": docs, "characters": chars, "sha256_uncompressed": digest.hexdigest()
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # pyarrow/fsspec leaves a C callback thread behind under Python 3.12 on this host.
    # All files are closed above; bypass interpreter finalization to avoid a false fatal exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
