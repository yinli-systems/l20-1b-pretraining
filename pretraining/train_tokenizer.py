#!/usr/bin/env python3
"""Train a new 32k byte-level BPE tokenizer; no pretrained tokenizer is reused."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import zstandard as zstd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from hf_config import write_hf_config


def text_iterator(paths: list[Path]):
    readers = []
    try:
        for path in paths:
            raw = path.open("rb")
            stream = zstd.ZstdDecompressor().stream_reader(raw)
            import io

            reader = io.TextIOWrapper(stream, encoding="utf-8")
            readers.append((reader, raw))
        active = list(readers)
        while active:
            next_active = []
            for reader, raw in active:
                for _ in range(32):
                    line = reader.readline()
                    if not line:
                        break
                    yield json.loads(line)["text"]
                if line:
                    next_active.append((reader, raw))
            active = next_active
    finally:
        for reader, raw in readers:
            reader.close()
            raw.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    args = parser.parse_args()
    paths = [args.sample_dir / f"{source}.jsonl.zst" for source in ("web", "synthetic", "math", "code")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing tokenizer samples: {missing}")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator(paths), trainer=trainer)
    if tokenizer.get_vocab_size(with_added_tokens=False) != args.vocab_size:
        raise SystemExit(f"unexpected vocabulary size: {tokenizer.get_vocab_size(with_added_tokens=False)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output_dir / "tokenizer.json"))
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|padding|>",
        "model_max_length": 2048,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    (args.output_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": None,
                "eos_token_id": tokenizer.token_to_id("<|endoftext|>"),
                "pad_token_id": tokenizer.token_to_id("<|padding|>"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_hf_config(args.output_dir)
    probes = ["The quick brown fox.", "def fibonacci(n: int) -> int:\n", "Let \\(x^2 + y^2 = 1\\)."]
    report = {
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=False),
        "eos_id": tokenizer.token_to_id("<|endoftext|>"),
        "round_trip": [tokenizer.decode(tokenizer.encode(probe).ids) == probe for probe in probes],
    }
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
