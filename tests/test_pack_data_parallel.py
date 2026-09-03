#!/usr/bin/env python3
"""Check that batched/multiprocess packing is token-for-token serial equivalent."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExitCalled(Exception):
    pass


class OsProxy:
    def __getattr__(self, name):
        if name == "_exit":
            return lambda code: (_ for _ in ()).throw(ExitCalled(code))
        return getattr(os, name)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("pack_data_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def documents():
    marker = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    rows = []
    for index in range(3000):
        body = (
            f"Document {index} explains deterministic systems, language modeling, mathematics, and software. "
            f"The unique checksum label is item-{index:06d}. "
        ) * 3
        if index % 211 == 0:
            body += " " + marker
        rows.append({"text": body, "id": str(index), "source": "web"})
        if index % 137 == 0:
            rows.append(dict(rows[-1]))
    return rows, marker


def run(module, root: Path, rows: list[dict], marker: str, workers: int, batch_docs: int):
    output = root / f"out-{workers}-{batch_docs}"
    dedup = root / f"dedup-{workers}-{batch_docs}.sqlite"
    decontam = root / "decontam.sqlite"
    if not decontam.exists():
        connection = sqlite3.connect(decontam)
        connection.execute("CREATE TABLE ngrams (hash BLOB PRIMARY KEY)")
        digest = hashlib.blake2b(marker.encode(), digest_size=8).digest()
        connection.execute("INSERT INTO ngrams(hash) VALUES (?)", (digest,))
        connection.commit()
        connection.close()

    module.iter_source = lambda *args, **kwargs: iter(rows)
    module.os = OsProxy()
    argv = [
        str(Path(module.__file__)),
        "--source", "web",
        "--tokenizer-dir", str(PROJECT_ROOT / "artifacts/tokenizer"),
        "--output-dir", str(output),
        "--dedup-db", str(dedup),
        "--decontam-db", str(decontam),
        "--target-tokens", str(40 * 2049),
        "--validation-tokens", str(5 * 2049),
        "--shard-tokens", str(7 * 2049),
        "--holdout-modulus", "10",
        "--progress-every", "100000",
        "--workers", str(workers),
        "--batch-docs", str(batch_docs),
    ]
    previous = sys.argv
    sys.argv = argv
    try:
        module.main()
    except ExitCalled as error:
        if error.args != (0,):
            raise
    finally:
        sys.argv = previous

    source = output / "web"
    arrays = {}
    for split in ("train-npy", "val-npy"):
        arrays[split] = np.concatenate(
            [np.load(path, allow_pickle=False) for path in sorted((source / split).glob("*.npy"))]
        )
    return arrays, json.loads((source / "manifest.json").read_text())


def main() -> None:
    module = load_module(PROJECT_ROOT / "pack_data.py")
    rows, marker = documents()
    with tempfile.TemporaryDirectory(prefix="pack-equivalence-") as directory:
        root = Path(directory)
        serial_arrays, serial_manifest = run(module, root, rows, marker, 1, 1)
        parallel_arrays, parallel_manifest = run(module, root, rows, marker, 6, 1024)
        for split in serial_arrays:
            if not np.array_equal(serial_arrays[split], parallel_arrays[split]):
                raise RuntimeError(f"token mismatch in {split}")
        if serial_manifest["counters"] != parallel_manifest["counters"]:
            raise RuntimeError(
                f"counter mismatch: {serial_manifest['counters']} != {parallel_manifest['counters']}"
            )
        print(
            json.dumps(
                {
                    "passed": True,
                    "train_tokens": int(serial_arrays["train-npy"].size),
                    "validation_tokens": int(serial_arrays["val-npy"].size),
                    "counters": serial_manifest["counters"],
                },
                sort_keys=True,
            )
        )


def test_parallel_packing_matches_serial_output() -> None:
    main()


if __name__ == "__main__":
    main()
