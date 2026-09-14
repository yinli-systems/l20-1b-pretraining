#!/usr/bin/env python3
"""Apply the frozen v3 learning-rate selection rule to completed pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


EXPECTED_PROTOCOL_SHA256 = "beec1dc00f73bcfdd9e21e2a12c452a2248ddf0258a2bf89391f5d9978d4134e"
EXPECTED_TRAIN_SHA256 = "27b5f998d5a48223db4a6eb6b245c133e5c13a02a5bac8a19d2b7078a794e29e"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def evaluate_candidate(directory: Path, peak_lr: float, rule: dict) -> dict:
    manifest_path = directory / "run-manifest.json"
    metrics_path = directory / "metrics.jsonl"
    status_path = directory / "training-status.json"
    manifest = load_json(manifest_path)
    status = load_json(status_path)
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    train_rows = [row for row in rows if "loss" in row]
    validation_rows = {int(row["step"]): row for row in rows if "val_loss" in row}
    reasons: list[str] = []

    if status.get("status") != "COMPLETED" or status.get("step") != 300:
        reasons.append(f"training status is {status}")
    if len(train_rows) != 300 or [row["step"] for row in train_rows] != list(range(1, 301)):
        reasons.append("training metrics do not contain exactly steps 1..300")
    required_steps = [int(value) for value in rule["qualification"]["required_validation_steps"]]
    if sorted(validation_rows) != required_steps:
        reasons.append(f"validation steps are {sorted(validation_rows)}, expected {required_steps}")
    if manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        reasons.append("protocol hash mismatch")
    if manifest.get("code_sha256", {}).get("train.py") != EXPECTED_TRAIN_SHA256:
        reasons.append("train.py hash mismatch")
    expected = rule["pilot"]
    checks = {
        "world_size": int(expected["gpus"]),
        "microbatch": int(expected["microbatch"]),
        "tokens_per_step": int(expected["global_prediction_tokens_per_step"]),
        "deterministic_algorithms": bool(expected["deterministic_algorithms"]),
        "ddp_gradient_reduction": expected["ddp_gradient_reduction"],
        "ddp_bucket_cap_mb": float(expected["ddp_bucket_cap_mb"]),
    }
    for key, value in checks.items():
        if manifest.get(key) != value:
            reasons.append(f"manifest {key}={manifest.get(key)!r}, expected {value!r}")
    if not math.isclose(float(manifest.get("peak_lr", -1)), peak_lr, rel_tol=0, abs_tol=1e-15):
        reasons.append("peak learning rate mismatch")

    numeric = []
    for row in train_rows:
        numeric.extend((row.get("loss"), row.get("grad_norm"), row.get("mfu")))
    numeric.extend(row.get("val_loss") for row in validation_rows.values())
    if any(value is None or not math.isfinite(float(value)) for value in numeric):
        reasons.append("a loss, gradient norm, validation loss, or MFU is non-finite")
    steady_mfu = [float(row["mfu"]) for row in train_rows if int(row["step"]) > 5]
    median_mfu = statistics.median(steady_mfu) if steady_mfu else float("nan")
    minimum = float(rule["qualification"]["post_compile_median_mfu_strictly_greater_than"])
    if not median_mfu > minimum:
        reasons.append(f"post-compile median MFU {median_mfu} is not greater than {minimum}")

    val_losses = {str(step): float(validation_rows[step]["val_loss"]) for step in required_steps if step in validation_rows}
    score = statistics.mean(val_losses.values()) if len(val_losses) == len(required_steps) else None
    return {
        "peak_lr": peak_lr,
        "qualified": not reasons,
        "rejection_reasons": reasons,
        "validation_loss": val_losses,
        "selection_score_mean_validation_loss": score,
        "post_compile_median_mfu": median_mfu,
        "post_compile_minimum_mfu": min(steady_mfu) if steady_mfu else None,
        "metrics_sha256": digest(metrics_path),
        "manifest_sha256": digest(manifest_path),
        "training_status_sha256": digest(status_path),
        "run_fingerprint_sha256": manifest.get("run_fingerprint_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-job", required=True)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rule = load_json(args.rule)
    if rule.get("status") != "FROZEN_BEFORE_LONG_PILOT_RESULTS":
        raise RuntimeError("selection rule is not frozen")
    candidates = []
    for peak_lr in [float(value) for value in rule["candidates"]]:
        directory = args.root / "pilots" / f"lr-long-v3-{args.pilot_job}" / f"lr-{peak_lr}"
        candidates.append(evaluate_candidate(directory, peak_lr, rule))
    qualified = [candidate for candidate in candidates if candidate["qualified"]]
    if not qualified:
        raise RuntimeError(f"no LR candidate qualified: {candidates}")
    selected = min(
        qualified,
        key=lambda candidate: (candidate["selection_score_mean_validation_loss"], candidate["peak_lr"]),
    )
    receipt = {
        "status": "PASS_FROZEN_LR_SELECTION",
        "rule_id": rule["rule_id"],
        "rule_sha256": digest(args.rule),
        "pilot_job": str(args.pilot_job),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "candidates": candidates,
        "selected_peak_lr": selected["peak_lr"],
        "selected_score": selected["selection_score_mean_validation_loss"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
