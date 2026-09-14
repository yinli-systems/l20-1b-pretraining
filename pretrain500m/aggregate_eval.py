#!/usr/bin/env python3
"""Recompute the frozen seven-task aggregate from lm-eval sample logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_COUNTS = {
    "hellaswag": 10042,
    "piqa": 1838,
    "winogrande": 1267,
    "openbookqa": 500,
    "arc_easy": 2376,
    "arc_challenge": 1172,
    "boolq": 3270,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_samples(path: Path, metric: str, expected_count: int) -> np.ndarray:
    values: list[float] = []
    doc_ids: set[object] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if metric not in row:
                raise RuntimeError(f"{path}:{line_number} lacks primary metric {metric!r}")
            value = float(row[metric])
            if not np.isfinite(value):
                raise RuntimeError(f"{path}:{line_number} has a non-finite score")
            if value < 0.0 or value > 1.0:
                raise RuntimeError(f"{path}:{line_number} score {value} is outside [0, 1]")
            doc_id = row.get("doc_id")
            if doc_id in doc_ids:
                raise RuntimeError(f"{path}:{line_number} repeats doc_id {doc_id!r}")
            doc_ids.add(doc_id)
            values.append(value)
    if len(values) != expected_count:
        raise RuntimeError(
            f"{path} contains {len(values)} samples; frozen count is {expected_count}"
        )
    return np.asarray(values, dtype=np.float64)


def bootstrap_means(
    values: np.ndarray, replicates: int, rng: np.random.Generator, chunk_size: int = 256
) -> np.ndarray:
    output = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        output[start:stop] = values[indices].mean(axis=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    task_specs = protocol["tasks"]
    if set(task_specs) != set(EXPECTED_COUNTS):
        raise RuntimeError("protocol task set does not match the frozen seven-task aggregate")

    aggregate_spec = protocol["aggregate"]
    replicates = int(aggregate_spec["bootstrap_replicates"])
    seed = int(aggregate_spec["bootstrap_seed"])
    if replicates != 10000 or seed != 20260912:
        raise RuntimeError("bootstrap settings differ from the frozen protocol")

    name = args.results_json.name
    if not (name.startswith("results_") and name.endswith(".json")):
        raise RuntimeError("results filename must use lm-eval's results_<timestamp>.json form")
    timestamp = name[len("results_") : -len(".json")]
    result_tasks = results.get("results", {})
    if set(result_tasks) != set(EXPECTED_COUNTS):
        raise RuntimeError("lm-eval result task set does not match the frozen protocol")

    rng = np.random.Generator(np.random.PCG64(seed))
    task_output: dict[str, object] = {}
    replicate_sum = np.zeros(replicates, dtype=np.float64)
    point_sum = 0.0
    input_sha256: dict[str, str] = {
        "protocol": sha256(args.protocol),
        "results": sha256(args.results_json),
    }

    for task in task_specs:
        metric = task_specs[task]["primary_metric"]
        sample_path = args.results_json.parent / f"samples_{task}_{timestamp}.jsonl"
        if not sample_path.is_file():
            raise RuntimeError(f"missing sample log: {sample_path}")
        values = read_samples(sample_path, metric, EXPECTED_COUNTS[task])
        point = float(values.mean())
        lm_eval_key = f"{metric},none"
        reported = float(result_tasks[task][lm_eval_key])
        if not np.isclose(point, reported, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"{task} recomputed {metric}={point} but lm-eval reported {reported}"
            )
        draws = bootstrap_means(values, replicates, rng)
        replicate_sum += draws
        point_sum += point
        task_output[task] = {
            "primary_metric": metric,
            "samples": int(values.size),
            "score": point,
            "bootstrap_ci_95": [
                float(np.percentile(draws, 2.5, method="linear")),
                float(np.percentile(draws, 97.5, method="linear")),
            ],
            "sample_log": str(sample_path),
            "sample_log_sha256": sha256(sample_path),
        }
        input_sha256[f"samples_{task}"] = task_output[task]["sample_log_sha256"]

    aggregate_draws = replicate_sum / len(task_specs)
    output = {
        "status": "PASS_FROZEN_EVALUATION_AGGREGATE",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": input_sha256["protocol"],
        "aggregate": {
            "name": aggregate_spec["name"],
            "score": point_sum / len(task_specs),
            "bootstrap_ci_95": [
                float(np.percentile(aggregate_draws, 2.5, method="linear")),
                float(np.percentile(aggregate_draws, 97.5, method="linear")),
            ],
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "percentile_method": "linear",
            "task_weighting": "unweighted",
        },
        "tasks": task_output,
        "input_sha256": input_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
