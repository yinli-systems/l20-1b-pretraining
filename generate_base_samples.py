#!/usr/bin/env python3
"""CPU-only, identity-checked qualitative samples of the frozen 20B base model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Never contend with the continuation pilot for CUDA memory or download weights.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

EXPECTED = {
    "pytorch_model.bin": "bc7c43425cf538d5dcea333fdd1513faef0a994679acc08681505a879b3cd1f3",
    "config.json": "b9310f5a84d88f7a6dc84782fd6cec531f4d576c2881106dc1914e58ced67ebe",
    "model_config.yaml": "030d868464d4466007747b8f28c8fe5a43884b5aa4e28580d01b86df27b8ff8e",
    "tokenizer.json": "e755dfeeed1acab465839adc0988811983a3287db230a8d9bc0f74704208609c",
}
PROMPTS = (
    ("dialogue", "User: Hello! What can you help me with?\nAssistant:"),
    ("explanation", "Question: Why do we have day and night?\nAnswer:"),
    ("story", "The old lighthouse had been empty for fifty years. One stormy night, a light appeared in its highest window."),
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("cannot determine available RAM")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if available_gib() < 8:
        raise RuntimeError("CPU sampling requires at least 8 GiB available RAM")
    started = time.time()
    hashes = {name: digest(args.model_dir / name) for name in EXPECTED}
    if hashes != EXPECTED:
        raise RuntimeError(f"base model identity mismatch: {hashes}")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, local_files_only=True, attn_implementation="sdpa",
    ).eval()
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 1_100_048_384
    generation = GenerationConfig(
        max_new_tokens=80, do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None, repetition_penalty=1.0,
    )
    receipt = {
        "model": str(args.model_dir), "model_hashes": hashes,
        "model_parameters": 1_100_048_384, "parent_prediction_tokens": 19_999_703_040,
        "device": "cpu", "dtype": "bfloat16", "torch": torch.__version__,
        "transformers": transformers.__version__, "threads": 2, "seed": 42,
        "chat_template": False, "generation": generation.to_dict(),
        "script_sha256": digest(Path(__file__)), "started_unix": started,
        "notes": "Three prespecified prompts; one greedy generation each, no retries, edits, or candidate selection. Qualitative only; CPU timing is not GPU inference throughput.",
    }
    with args.output.open("x") as stream:
        def emit(record):
            line = json.dumps(record, ensure_ascii=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)
        emit({"stage": "loaded", **receipt, "load_seconds": time.time() - started})
        for name, prompt in PROMPTS:
            if available_gib() < 4:
                raise RuntimeError("stopping sampling to preserve training RAM")
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                               return_token_type_ids=False)
            tick = time.time()
            with torch.inference_mode():
                outputs = model.generate(**inputs, generation_config=generation)
            ids = outputs[0, inputs.input_ids.shape[1]:].tolist()
            ended = bool(ids) and ids[-1] == tokenizer.eos_token_id
            emit({
                "stage": "sample", "name": name, "prompt": prompt,
                "input_ids": inputs.input_ids[0].tolist(), "generated_ids": ids,
                "continuation": tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "display_continuation": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                "new_tokens": len(ids), "seconds": time.time() - tick,
                "finish_reason": "eos" if ended else "length", "time": time.time(),
            })
        emit({"stage": "complete", "samples": len(PROMPTS), "finished_unix": time.time(), "seconds": time.time() - started})


if __name__ == "__main__":
    main()
