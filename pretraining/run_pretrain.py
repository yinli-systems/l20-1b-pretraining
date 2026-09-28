#!/usr/bin/env python3
"""Launch random-initialized pretraining in smoke, gate, or full mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from litgpt.args import EvalArgs, TrainArgs
import litgpt.pretrain as litgpt_pretrain

from data_module import HighQualityEnglish


def resolve_max_tokens(selected: dict[str, int], micro_batch_size: int) -> int:
    """Resolve token budget without eagerly reading a mode-only key."""
    if "max_tokens" in selected:
        return selected["max_tokens"]
    return selected["iterations"] * micro_batch_size * 2048


def main() -> None:
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "gate", "full"), required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("/home/hhai/pretrain/data/packed"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("/home/hhai/pretrain/tokenizer"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/hhai/pretrain/checkpoints"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-step-checkpoints", type=int, default=2)
    args = parser.parse_args()

    full_global_batch = (512 // args.micro_batch_size) * args.micro_batch_size
    requested_full_tokens = 20_000_000_000
    full_tokens_per_step = full_global_batch * 2048
    aligned_full_tokens = requested_full_tokens // full_tokens_per_step * full_tokens_per_step
    modes = {
        "smoke": {"iterations": 20, "global_batch": args.micro_batch_size, "warmup": 2, "save": 10, "eval": 10},
        "gate": {"iterations": 1000, "global_batch": args.micro_batch_size, "warmup": 50, "save": 500, "eval": 250},
        "full": {
            "max_tokens": aligned_full_tokens,
            "global_batch": full_global_batch,
            "warmup": 1000,
            "save": 500,
            "eval": 500,
        },
    }
    selected = modes[args.mode]
    max_tokens = resolve_max_tokens(selected, args.micro_batch_size)
    run_dir = args.output_dir / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "from_scratch": True,
        "initial_checkpoint": None,
        "model_name": "tiny-llama-1.1b",
        "max_tokens": max_tokens,
        "requested_max_tokens": requested_full_tokens if args.mode == "full" else max_tokens,
        "optimizer_steps": max_tokens // (selected["global_batch"] * 2048),
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": selected["global_batch"],
        "sequence_length": 2048,
        "precision": "bf16-mixed",
        "seed": 42,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_manifest, sort_keys=True), flush=True)

    def save_real_hyperparameters(_function, checkpoint_dir: Path) -> None:
        import yaml

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "hyperparameters.yaml").write_text(yaml.safe_dump(run_manifest, sort_keys=True))

    # LitGPT's default helper reparses sys.argv, which belongs to this wrapper and
    # would write incorrect defaults. Persist the already validated run manifest.
    litgpt_pretrain.save_hyperparameters = save_real_hyperparameters

    original_save_checkpoint = litgpt_pretrain.save_checkpoint

    def save_checkpoint_and_prune(*save_args, **save_kwargs) -> None:
        original_save_checkpoint(*save_args, **save_kwargs)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("prune_checkpoints.py")),
                "--run-dir",
                str(run_dir),
                "--keep",
                str(args.keep_step_checkpoints),
                "--apply",
            ],
            check=True,
        )

    litgpt_pretrain.save_checkpoint = save_checkpoint_and_prune

    litgpt_pretrain.setup(
        model_name="tiny-llama-1.1b",
        out_dir=run_dir,
        precision="bf16-mixed",
        initial_checkpoint_dir=None,
        resume="auto" if args.resume else False,
        data=HighQualityEnglish(data_path=args.data_dir, seed=42, num_workers=2),
        train=TrainArgs(
            save_interval=selected["save"],
            log_interval=1,
            global_batch_size=selected["global_batch"],
            micro_batch_size=args.micro_batch_size,
            lr_warmup_steps=selected["warmup"],
            max_tokens=max_tokens,
            max_seq_length=2048,
            max_norm=1.0,
            min_lr=4e-5,
            tie_embeddings=False,
        ),
        eval=EvalArgs(
            interval=selected["eval"],
            max_iters=25 if args.mode == "smoke" else 100,
            initial_validation=True,
            final_validation=True,
        ),
        optimizer={
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 4e-4, "betas": (0.9, 0.95), "weight_decay": 0.1},
        },
        devices=1,
        num_nodes=1,
        tokenizer_dir=args.tokenizer_dir,
        logger_name="tensorboard",
        seed=42,
    )


if __name__ == "__main__":
    main()
