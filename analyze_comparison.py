#!/usr/bin/env python3
"""Derive a compact, hash-bound comparison without copying benchmark texts.

Run beside evaluate_comparison.py in the GPU environment. Original evaluations
remain immutable. Confidence intervals describe test-sample uncertainty only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from evaluate_comparison import BASELINES, ORIGINALS, OUR_MODEL, OUTPUT, ROOT, save_json, sha256

PRIMARY = {
    "hellaswag": "acc_norm", "openbookqa": "acc_norm", "winogrande": "acc",
    "arc_challenge": "acc_norm", "arc_easy": "acc_norm", "boolq": "acc", "piqa": "acc_norm",
}
SEEDS = {"random_seed": 42, "numpy_seed": 42, "torch_seed": 42, "fewshot_seed": 1234}


def read_verified(path: Path, digest: str) -> dict:
    actual = sha256(path)
    if actual != digest:
        raise ValueError(f"Hash mismatch: {path}")
    result = json.loads(path.read_text())
    for key, value in SEEDS.items():
        if result["config"].get(key) != value:
            raise ValueError(f"Wrong seed {key}: {path}")
    for key, value in {"limit": None, "model_dtype": "torch.bfloat16", "batch_size": "auto:4"}.items():
        if key not in result["config"] or result["config"][key] != value:
            raise ValueError(f"Wrong config {key}: {path}")
    if result.get("lm_eval_version") != "0.4.9" or result.get("max_length") != 2048:
        raise ValueError(f"Wrong harness/context: {path}")
    return result


def samples_for(result: dict, task: str, metric: str, filter_name: str) -> dict:
    selected = {}
    for sample in result["samples"][task]:
        if sample["filter"] != filter_name:
            continue
        key = (sample["doc_id"], sample["doc_hash"])
        if key in selected:
            raise ValueError(f"Duplicate sample in {task}: {key}")
        value = float(sample[metric])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite metric in {task}: {key}")
        selected[key] = sample
    if not selected:
        raise ValueError(f"No samples for {task}/{filter_name}")
    expected = result["n-samples"][task]["effective"]
    if len(selected) != expected:
        raise ValueError(f"Incomplete sample set for {task}: {len(selected)} != {expected}")
    observed = np.mean([float(s[metric]) for s in selected.values()])
    reported = result["results"][task][f"{metric},{filter_name}"]
    if not np.isclose(observed, reported, rtol=0, atol=1e-10):
        raise ValueError(f"Sample mean differs from reported result: {task}")
    return selected


def align(ours: dict, baseline: dict, task: str, metric: str, filter_name: str = "none") -> tuple:
    for field in ("versions", "n-shot", "configs"):
        if ours[field][task] != baseline[field][task]:
            raise ValueError(f"Protocol mismatch: {task}/{field}")
    left = samples_for(ours, task, metric, filter_name)
    right = samples_for(baseline, task, metric, filter_name)
    if left.keys() != right.keys():
        raise ValueError(f"Different documents: {task}")
    pairs = []
    fingerprint = hashlib.sha256()
    for key in sorted(left):
        a, b = left[key], right[key]
        for field in ("prompt_hash", "target_hash"):
            if a[field] != b[field]:
                raise ValueError(f"Different {field}: {task}/{key}")
        fingerprint.update(json.dumps([key, a["prompt_hash"], a["target_hash"]]).encode())
        pairs.append((float(a[metric]), float(b[metric])))
    return np.asarray(pairs), fingerprint.hexdigest()


def bootstrap_difference(pairs: np.ndarray, rng: np.random.Generator, repetitions: int) -> np.ndarray:
    # Sampling the empirical difference histogram is exactly a paired bootstrap;
    # binary accuracy needs only three bins, rather than repetitions*n indices.
    differences = pairs[:, 0] - pairs[:, 1]
    values, counts = np.unique(differences, return_counts=True)
    output = np.empty(repetitions)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        draws = rng.multinomial(len(pairs), counts / len(pairs), size=size)
        output[start:start + size] = (draws @ values) / len(pairs)
    return output


def choice_diagnostics(result: dict, tasks: list[str], labels: list[str]) -> dict:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for task in tasks:
        samples = samples_for(result, task, "acc", "none")
        for sample in samples.values():
            target = int(sample["target"])
            prediction = int(np.argmax([response[0] for response in sample["filtered_resps"]]))
            if float(target == prediction) != sample["acc"]:
                raise ValueError(f"Cannot reconstruct {task} prediction")
            matrix[target, prediction] += 1
    n = int(matrix.sum())
    return {"n": n, "labels": labels,
            "confusion_rows_target_columns_prediction": matrix.tolist(),
            "target_counts": matrix.sum(axis=1).tolist(), "prediction_counts": matrix.sum(axis=0).tolist(),
            "majority_class_accuracy": float(matrix.sum(axis=1).max()/n),
            "accuracy": float(matrix.trace()/n)}


def boolq_diagnostics(result: dict) -> dict:
    return choice_diagnostics(result, ["boolq"], result["configs"]["boolq"]["doc_to_choice"])


def compare_task(ours: dict, baseline: dict, task: str, metric: str,
                 rng: np.random.Generator, repetitions: int, filter_name: str = "none") -> tuple:
    pairs, fingerprint = align(ours, baseline, task, metric, filter_name)
    draws = bootstrap_difference(pairs, rng, repetitions)
    differences = pairs[:, 0] - pairs[:, 1]
    result = {
        "metric": f"{metric},{filter_name}", "n": len(pairs),
        "ours": float(pairs[:, 0].mean()), "baseline": float(pairs[:, 1].mean()),
        "difference_pp": float(differences.mean() * 100),
        "paired_95_ci_pp": (np.quantile(draws, [0.025, 0.975]) * 100).tolist(),
        "ours_better_samples": int(np.sum(differences > 0)),
        "baseline_better_samples": int(np.sum(differences < 0)),
        "tied_samples": int(np.sum(differences == 0)),
        "document_prompt_target_fingerprint": fingerprint,
        "task_config_sha256": hashlib.sha256(json.dumps(ours["configs"][task], sort_keys=True).encode()).hexdigest(),
        "protocol_alignment": "PASS",
    }
    return result, draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--hash-model", action="store_true", help="Also bind the current local model/tokenizer artifacts")
    parser.add_argument("--output", type=Path, default=OUTPUT / "comparison-summary.json")
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1000:
        parser.error("Use at least 1000 bootstrap repetitions")
    receipt_bytes = (OUTPUT / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    originals = {name: read_verified(ROOT / "evaluations/final" / f"{name}.json", digest)
                 for name, digest in ORIGINALS.items()}
    completed = {name: read_verified(Path(job["result"]), job["sha256"])
                 for name, job in receipt["jobs"].items() if job["status"] == "complete"}
    for name, result in completed.items():
        job = receipt["jobs"][name]["job"]
        if result["config"]["model"] != job["model"]:
            raise ValueError(f"Model ID mismatch: {name}")
        if job["revision"] and result["config"]["model_revision"] != job["revision"]:
            raise ValueError(f"Model revision mismatch: {name}")
    if "ours_boolq" not in completed:
        raise RuntimeError("BoolQ is not complete")
    ours = {task: originals["zero_shot_core"] for task in PRIMARY if task != "boolq"}
    ours["boolq"] = completed["ours_boolq"]
    our_scores = {task: source["results"][task][f"{PRIMARY[task]},none"] for task, source in ours.items()}
    summary = {
        "created_unix": time.time(), "suite_status_at_analysis": receipt["status"],
        "evaluation_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "analysis_sha256": sha256(Path(__file__)), "runner_sha256": receipt["runner_sha256"],
        "original_result_sha256": ORIGINALS,
        "comparison_result_sha256": {name: receipt["jobs"][name]["sha256"] for name in completed},
        "protocol": receipt["protocol"], "baseline_revisions": BASELINES,
        "analysis_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                 "packages": {package: importlib.metadata.version(package)
                                              for package in ("lm_eval", "torch", "transformers", "datasets", "tokenizers")}},
        "uncertainty": {"method": "paired percentile bootstrap; primary macro stratified by task",
                        "repetitions": args.bootstrap_repetitions, "seed": args.seed,
                        "scope": "test-sample uncertainty, not training-seed uncertainty; no multiple-testing correction"},
        "ours_primary_scores": our_scores, "ours_primary_macro": float(np.mean(list(our_scores.values()))),
        "boolq_diagnostics": {"ours": boolq_diagnostics(completed["ours_boolq"])},
        "mmlu_choice_diagnostics": {"ours": choice_diagnostics(originals["mmlu_5shot"],
                                                               sorted(originals["mmlu_5shot"]["samples"]),
                                                               ["A", "B", "C", "D"])},
        "all_reported_metrics": {"ours_original": {name: r["results"] for name, r in originals.items()},
                                 **{name: r["results"] for name, r in completed.items()}},
        "comparisons": {},
        "limitations": ["Original 20B decontamination did not explicitly include BoolQ.",
                        "Same 2048-token cap; tokenizer-dependent truncation can differ.",
                        "Cross-tokenizer perplexities must not be ranked directly.",
                        "No Chat template or code-generation benchmark is included."],
    }
    rng = np.random.default_rng(args.seed)
    for name in BASELINES:
        comparison = {"primary": {}, "extended": {}}
        key = f"{name}_core"
        if key in completed:
            summary["boolq_diagnostics"][name] = boolq_diagnostics(completed[key])
            draws = []
            for task, metric in PRIMARY.items():
                result, bootstrap = compare_task(ours[task], completed[key], task, metric, rng, args.bootstrap_repetitions)
                comparison["primary"][task] = result
                draws.append(bootstrap)
            macro = np.mean(draws, axis=0)
            task_results = comparison["primary"].values()
            comparison["primary_macro"] = {
                "ours": summary["ours_primary_macro"],
                "baseline": float(np.mean([r["baseline"] for r in task_results])),
                "difference_pp": float(np.mean([r["difference_pp"] for r in task_results])),
                "paired_95_ci_pp": (np.quantile(macro, [0.025, 0.975]) * 100).tolist(),
            }
            without_boolq = [r for t, r in comparison["primary"].items() if t != "boolq"]
            comparison["six_task_sensitivity_excluding_boolq"] = {
                "ours": float(np.mean([r["ours"] for r in without_boolq])),
                "baseline": float(np.mean([r["baseline"] for r in without_boolq])),
                "difference_pp": float(np.mean([r["difference_pp"] for r in without_boolq])),
                "reason": "BoolQ was not explicitly covered by original 20B benchmark decontamination",
            }
            for task in ("lambada_openai", "truthfulqa_mc2"):
                result, _ = compare_task(originals["zero_shot_core"], completed[key], task, "acc", rng, args.bootstrap_repetitions)
                comparison["extended"][task] = result
        key = f"{name}_mmlu_5shot"
        if key in completed:
            summary["mmlu_choice_diagnostics"][name] = choice_diagnostics(
                completed[key], sorted(completed[key]["samples"]), ["A", "B", "C", "D"])
            subject_results, weighted_draws, total = {}, [], 0
            for task in sorted(originals["mmlu_5shot"]["samples"]):
                result, draws = compare_task(originals["mmlu_5shot"], completed[key], task, "acc", rng, args.bootstrap_repetitions)
                subject_results[task] = result
                weighted_draws.append(draws * result["n"])
                total += result["n"]
            pooled = np.sum(weighted_draws, axis=0) / total
            left = sum(r["ours"] * r["n"] for r in subject_results.values()) / total
            right = sum(r["baseline"] * r["n"] for r in subject_results.values()) / total
            for source, score in ((originals["mmlu_5shot"], left), (completed[key], right)):
                if not np.isclose(source["results"]["mmlu"]["acc,none"], score, rtol=0, atol=1e-10):
                    raise ValueError("MMLU group aggregation mismatch")
            comparison["extended"]["mmlu_5shot"] = {
                "ours": left, "baseline": right, "n": total, "difference_pp": (left-right)*100,
                "paired_95_ci_pp": (np.quantile(pooled, [0.025, 0.975])*100).tolist(),
                "aggregation": "sample-weighted; bootstrap stratified by subject", "subjects": subject_results,
            }
        key = f"{name}_gsm8k_5shot"
        if key in completed:
            for filter_name in ("strict-match", "flexible-extract"):
                result, _ = compare_task(originals["gsm8k_5shot"], completed[key], "gsm8k", "exact_match",
                                         rng, args.bootstrap_repetitions, filter_name)
                comparison["extended"][f"gsm8k_{filter_name}"] = result
        summary["comparisons"][name] = comparison
    if args.hash_model:
        artifacts = {}
        for path in sorted(OUR_MODEL.iterdir()):
            if not path.is_file():
                continue
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Model artifact changed during hashing: {path}")
            artifacts[path.name] = {"sha256": digest, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}
        summary["ours_model_artifacts"] = {"path": str(OUR_MODEL), "captured_unix": time.time(),
                                           "timing": "post-evaluation hash capture", "files": artifacts}
    save_json(args.output, summary)
    print(json.dumps({"status": summary["suite_status_at_analysis"], "ours_primary_macro": summary["ours_primary_macro"],
                      "comparisons": {name: {"primary_macro": c.get("primary_macro"), "extended_tasks": list(c["extended"])}
                                      for name, c in summary["comparisons"].items()}}, indent=2))


if __name__ == "__main__":
    main()
