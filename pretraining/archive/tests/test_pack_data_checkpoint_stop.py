#!/usr/bin/env python3
"""Verify SIGTERM checkpoints all block-aligned token buffers before exit."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("pack_data_stop_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def slow_documents():
    index = 0
    while True:
        time.sleep(0.01)
        yield {
            "text": (
                f"Slow checkpoint document {index} contains enough unique educational text "
                "about systems, mathematics, language, and software engineering. "
            ) * 5,
            "id": str(index),
            "source": "web",
        }
        index += 1


def main() -> None:
    module = load_module(PROJECT_ROOT / "pack_data.py")
    module.iter_source = lambda *args, **kwargs: slow_documents()
    with tempfile.TemporaryDirectory(prefix="pack-stop-") as directory:
        root = Path(directory)
        decontam = root / "decontam.sqlite"
        connection = sqlite3.connect(decontam)
        connection.execute("CREATE TABLE ngrams (hash BLOB PRIMARY KEY)")
        connection.commit()
        connection.close()
        output = root / "out"
        sys.argv = [
            str(Path(module.__file__)),
            "--source", "web",
            "--tokenizer-dir", str(PROJECT_ROOT / "artifacts/tokenizer"),
            "--output-dir", str(output),
            "--dedup-db", str(root / "dedup.sqlite"),
            "--decontam-db", str(decontam),
            "--target-tokens", str(1000 * 2049),
            "--validation-tokens", str(100 * 2049),
            "--shard-tokens", str(100 * 2049),
            "--holdout-modulus", "10",
            "--workers", "6",
            "--batch-docs", "64",
        ]
        timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            module.main()
        except SystemExit as error:
            if error.code != 130:
                raise
        finally:
            timer.cancel()

        progress = json.loads((output / "web/progress.json").read_text())
        if not progress.get("checkpointed_stop"):
            raise RuntimeError(f"missing checkpointed stop receipt: {progress}")
        split_tokens = {}
        for split, key in (("train-npy", "train_tokens"), ("val-npy", "val_tokens")):
            count = sum(
                int(np.load(path, mmap_mode="r", allow_pickle=False).size)
                for path in (output / "web" / split).glob("*.npy")
            )
            split_tokens[split] = count
            if count != progress[key] or count % 2049:
                raise RuntimeError(f"checkpoint mismatch for {split}: files={count}, progress={progress[key]}")
        print(json.dumps({"passed": True, "progress": progress, "split_tokens": split_tokens}, sort_keys=True))


def test_sigterm_checkpoints_block_aligned_buffers() -> None:
    main()


if __name__ == "__main__":
    main()
