#!/usr/bin/env python3
"""Run decontaminated, protocol-labelled external evaluations on the final base model."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from litgpt.eval.evaluate import convert_and_evaluate

from hf_config import write_hf_config


ROOT = Path("/home/hhai/pretrain")
SUITES = (
    {
        "name": "zero_shot_core",
        "tasks": "hellaswag,piqa,winogrande,openbookqa,arc_easy,arc_challenge,lambada_openai,truthfulqa_mc2",
        "num_fewshot": 0,
    },
    {"name": "mmlu_5shot", "tasks": "mmlu", "num_fewshot": 5},
    {"name": "gsm8k_5shot", "tasks": "gsm8k", "num_fewshot": 5},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    checkpoint = ROOT / "checkpoints/full/final"
    if not (checkpoint / "lit_model.pth").is_file():
        raise SystemExit(f"missing final checkpoint: {checkpoint}")
    # LitGPT checkpoints trained with a new tokenizer do not inherit a
    # Hugging Face config unless we provide one. Materialize the canonical
    # architecture before conversion so AutoConfig can load the result.
    write_hf_config(checkpoint)
    output = ROOT / "evaluations/final"
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"checkpoint": str(checkpoint), "started_unix": time.time(), "suites": {}}
    for index, suite in enumerate(SUITES):
        result_path = output / f"{suite['name']}.json"
        convert_and_evaluate(
            checkpoint_dir=checkpoint,
            tasks=suite["tasks"],
            out_dir=output / "hf",
            force_conversion=index == 0,
            num_fewshot=suite["num_fewshot"],
            batch_size="auto:4",
            device="cuda",
            dtype="bfloat16",
            seed=42,
            save_filepath=result_path,
        )
        receipt["suites"][suite["name"]] = {
            "tasks": suite["tasks"],
            "num_fewshot": suite["num_fewshot"],
            "result": str(result_path),
            "sha256": sha256(result_path),
        }
    receipt["completed_unix"] = time.time()
    destination = ROOT / "manifests/final-evaluation-receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
