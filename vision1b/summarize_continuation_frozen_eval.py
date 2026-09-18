#!/usr/bin/env python3
"""Compare continuation frozen-feature results with the frozen step-250 student baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASELINE = {
    "fashion_mnist.knn_top1": 0.671,
    "fashion_mnist.ridge_top1": 0.6991000175476074,
    "mnist.knn_top1": 0.6361,
    "mnist.ridge_top1": 0.7075999975204468,
}
BASELINE_MEAN = 0.6784500037670136
MAX_ABSOLUTE_REGRESSION = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def scores(path: Path) -> tuple[dict[str, float], dict]:
    report = json.loads(path.read_text())
    if report.get("status") != "COMPLETE" or set(report.get("results", {})) != {"student"}:
        raise RuntimeError(f"invalid student evaluation result: {path}")
    student = report["results"]["student"]
    values = {
        "fashion_mnist.knn_top1": student["fashion_mnist"]["knn"]["top1"],
        "fashion_mnist.ridge_top1": student["fashion_mnist"]["ridge_probe"]["top1"],
        "mnist.knn_top1": student["mnist"]["knn"]["top1"],
        "mnist.ridge_top1": student["mnist"]["ridge_probe"]["top1"],
    }
    return values, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step500", type=Path, required=True)
    parser.add_argument("--step1000", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    step500, report500 = scores(args.step500)
    step1000, report1000 = scores(args.step1000)
    mean500 = sum(step500.values()) / len(step500)
    mean1000 = sum(step1000.values()) / len(step1000)
    deltas = {key: step1000[key] - BASELINE[key] for key in BASELINE}
    promotion = mean1000 > BASELINE_MEAN and min(deltas.values()) >= -MAX_ABSOLUTE_REGRESSION
    output = {
        "baseline": {"scores": BASELINE, "student_mean_top1": BASELINE_MEAN, "step": 250},
        "claim_boundary": (
            "This diagnostic compares frozen-feature transfer on MNIST and Fashion-MNIST only. "
            "It does not retrospectively satisfy the continuation run's external zero-stderr gate, "
            "establish broad visual quality, or establish superiority over reference models."
        ),
        "gate": {
            "maximum_absolute_regression_per_primary_score": MAX_ABSOLUTE_REGRESSION,
            "requires_step1000_mean_strictly_above_step250": True,
        },
        "inputs": {
            "step500_results_sha256": sha256(args.step500),
            "step1000_results_sha256": sha256(args.step1000),
            "step500_protocol_sha256": report500["protocol_sha256"],
            "step1000_protocol_sha256": report1000["protocol_sha256"],
        },
        "step500": {"scores": step500, "student_mean_top1": mean500},
        "step1000": {
            "deltas_from_step250": deltas,
            "scores": step1000,
            "student_mean_top1": mean1000,
        },
        "status": "PROMOTE_STEP1000" if promotion else "RETAIN_STEP250",
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
