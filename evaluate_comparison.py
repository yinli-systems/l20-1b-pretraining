#!/usr/bin/env python3
"""Complete missing evaluations against revision-pinned TinyLlama baselines.

Writes an independent evidence bundle. Existing final evaluations and checkpoints
are inputs only. Run with the same environment as evaluate_final.py.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path("/home/hhai/pretrain")
OUTPUT = ROOT / "evaluations/comparison-20260911"
OUR_MODEL = ROOT / "evaluations/final/hf"
CORE = (
    "hellaswag", "piqa", "winogrande", "openbookqa", "arc_easy",
    "arc_challenge", "boolq", "lambada_openai", "truthfulqa_mc2",
)
BASELINES = {
    "tinyllama_3t": (
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "59f6f375b26bde864a6ca194a9a3044570490064",
    ),
    "tinyllama_v1_1": (
        "TinyLlama/TinyLlama_v1.1",
        "ff3c701f2424c7625fdefb9dd470f45ef18b02d6",
    ),
}
ORIGINALS = {
    "zero_shot_core": "6fa3ecfa69d3bf0bad50fcfe26207379fc65ca3aac9cf8b75d38ad33094045df",
    "mmlu_5shot": "d506a9d5bc74c11b9752b20181c5b87f2ba685b05c3cb8d8eb1700880b98c5f0",
    "gsm8k_5shot": "46fa267b4af7dbab3ef4177343a994126f42f854200f71d72a39f570ef8c11e3",
}
PROTOCOL = {
    "lm_eval_version": "0.4.9",
    "dtype": "bfloat16",
    "batch_size": "auto:4",
    "device": "cuda",
    "max_length": 2048,
    "random_seed": 42,
    "numpy_random_seed": 42,
    "torch_random_seed": 42,
    # LitGPT supplied the three seeds above but left the harness default here.
    "fewshot_random_seed": 1234,
    "log_samples": True,
    "limit": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def jobs(phase: str) -> list[dict]:
    selected = []
    if phase in ("all", "core"):
        selected.append({"name": "ours_boolq", "model": str(OUR_MODEL),
                         "revision": None, "tasks": ["boolq"], "num_fewshot": 0})
        for name, (model, revision) in BASELINES.items():
            selected.append({"name": f"{name}_core", "model": model,
                             "revision": revision, "tasks": list(CORE), "num_fewshot": 0})
    if phase in ("all", "extended"):
        for name, (model, revision) in BASELINES.items():
            for task in ("mmlu", "gsm8k"):
                selected.append({"name": f"{name}_{task}_5shot", "model": model,
                                 "revision": revision, "tasks": [task], "num_fewshot": 5})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "core", "extended"), default="all")
    parser.add_argument("--plan", action="store_true", help="Print the work list without loading models")
    args = parser.parse_args()
    work = jobs(args.phase)
    if args.plan:
        print(json.dumps({"protocol": PROTOCOL, "jobs": work}, indent=2))
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        installed = importlib.metadata.version("lm_eval")
        if installed != PROTOCOL["lm_eval_version"]:
            raise RuntimeError(f"Expected lm_eval 0.4.9, found {installed}")
        if not (OUR_MODEL / "pytorch_model.bin").is_file():
            raise FileNotFoundError(OUR_MODEL)
        for name, digest in ORIGINALS.items():
            if sha256(ROOT / "evaluations/final" / f"{name}.json") != digest:
                raise RuntimeError(f"Original evaluation integrity check failed: {name}")

        manifest_path = OUTPUT / "receipt.json"
        receipt = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            "protocol": PROTOCOL, "original_result_sha256": ORIGINALS,
            "created_unix": time.time(), "jobs": {},
        }
        if receipt["protocol"] != PROTOCOL or receipt["original_result_sha256"] != ORIGINALS:
            raise RuntimeError("Existing receipt describes a different protocol")
        receipt["runner_sha256"] = sha256(Path(__file__))
        receipt["status"] = "running"
        save_json(manifest_path, receipt)

        import torch
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM

        for job in work:
            name = job["name"]
            destination = OUTPUT / f"{name}.json"
            previous = receipt["jobs"].get(name, {})
            if previous.get("status") == "complete":
                if previous.get("job") != job or sha256(destination) != previous.get("sha256"):
                    raise RuntimeError(f"Completed result changed: {name}")
                print(f"VERIFIED_EXISTING {name}", flush=True)
                continue
            if destination.exists():
                raise FileExistsError(f"Unreceipted result will not be overwritten: {destination}")
            started = time.time()
            receipt["jobs"][name] = {"status": "running", "started_unix": started, "job": job}
            receipt["current_job"] = name
            save_json(manifest_path, receipt)
            print(f"START {name} {json.dumps(job)}", flush=True)
            try:
                kwargs = {"pretrained": job["model"], "device": "cuda", "batch_size": "auto:4",
                          "dtype": "bfloat16", "max_length": 2048}
                if job["revision"] is not None:
                    kwargs["revision"] = job["revision"]
                model = HFLM(**kwargs)
                results = evaluator.simple_evaluate(
                    model=model, tasks=job["tasks"], num_fewshot=job["num_fewshot"],
                    batch_size="auto:4", device="cuda", limit=None,
                    random_seed=42, numpy_random_seed=42, torch_random_seed=42,
                    fewshot_random_seed=1234, log_samples=True,
                )
                if results is None or not results.get("results") or not results.get("samples"):
                    raise RuntimeError(f"Incomplete evaluation result: {name}")
                save_json(destination, results)
                receipt["jobs"][name].update({
                    "status": "complete", "completed_unix": time.time(),
                    "result": str(destination), "sha256": sha256(destination),
                    "results": results["results"], "sample_counts": results.get("n-samples"),
                    "actual_config": results.get("config"), "versions": results.get("versions"),
                })
                save_json(manifest_path, receipt)
                print(f"COMPLETE {name} {json.dumps(results['results'])}", flush=True)
                del results, model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                receipt["status"] = "failed"
                receipt["jobs"][name].update({"status": "failed", "error": repr(exc),
                                               "failed_unix": time.time()})
                save_json(manifest_path, receipt)
                raise
        receipt["status"] = "complete" if all(
            receipt["jobs"].get(job["name"], {}).get("status") == "complete" for job in jobs("all")
        ) else "phase_complete"
        receipt["completed_unix"] = time.time()
        receipt.pop("current_job", None)
        save_json(manifest_path, receipt)
        print(f"FINISHED {receipt['status']}", flush=True)


if __name__ == "__main__":
    main()
