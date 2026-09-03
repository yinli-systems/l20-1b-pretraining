#!/usr/bin/env python3
"""Run and score the 1,000-step real-data training gate plus resume check."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/hhai/pretrain")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(command: list[str], log_path: Path, telemetry_path: Path | None = None) -> tuple[int, float]:
    telemetry = None
    telemetry_stream = None
    if telemetry_path is not None:
        telemetry_stream = telemetry_path.open("w")
        telemetry = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-l",
                "1",
            ],
            stdout=telemetry_stream,
            stderr=subprocess.STDOUT,
        )
    started = time.perf_counter()
    try:
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
        if telemetry_stream is not None:
            telemetry_stream.close()
    return result.returncode, time.perf_counter() - started


def numeric_after_commas(path: Path) -> list[tuple[float, float, float]]:
    values = []
    for line in path.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            values.append((float(fields[1]), float(fields[2]), float(fields[3])))
        except ValueError:
            continue
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batch-size", type=int, default=6)
    args = parser.parse_args()

    data_receipt = ROOT / "manifests/data-gate-receipt.json"
    if not data_receipt.is_file():
        raise SystemExit(f"missing gate data receipt: {data_receipt}")
    data = json.loads(data_receipt.read_text())
    if set(data.get("sources", {})) != {"web", "synthetic", "math", "code"}:
        raise SystemExit("gate data receipt is incomplete")

    project = ROOT / "src/hq-pretrain"
    command = [
        str(ROOT / ".venv/bin/python"),
        str(project / "run_pretrain.py"),
        "--mode",
        "gate",
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--no-resume",
        "--keep-step-checkpoints",
        "2",
    ]
    train_log = ROOT / "logs/train-gate.log"
    telemetry_log = ROOT / "logs/train-gate-gpu.csv"
    train_returncode, wall_seconds = run(command, train_log, telemetry_log)

    resume_command = command.copy()
    resume_command[resume_command.index("--no-resume")] = "--resume"
    resume_log = ROOT / "logs/train-gate-resume.log"
    resume_returncode, resume_seconds = run(resume_command, resume_log)

    text = train_log.read_text(errors="replace")
    iteration_ms = [float(value) for value in re.findall(r"iter time: ([0-9.]+) ms", text)]
    steady_ms = iteration_ms[100:] if len(iteration_ms) > 100 else iteration_ms
    median_iteration_ms = statistics.median(steady_ms) if steady_ms else math.inf
    steady_tokens_per_second = args.micro_batch_size * 2048 / (median_iteration_ms / 1000)
    val_losses = [float(value) for value in re.findall(r"val loss (?:: )?([0-9.]+)", text)]
    final_losses = [float(value) for value in re.findall(r"Final evaluation \| val loss: ([0-9.]+)", text)]

    telemetry = numeric_after_commas(telemetry_log)
    active = [row for row in telemetry if row[0] >= 10]
    active_gpu_mean = statistics.mean(row[0] for row in active) if active else 0.0
    peak_memory_mib = max((row[1] for row in telemetry), default=0.0)
    active_power_mean_w = statistics.mean(row[2] for row in active) if active else 0.0
    checkpoints = [
        {"path": str(path), "bytes": (path / "lit_model.pth").stat().st_size}
        for path in sorted((ROOT / "checkpoints/gate").glob("step-*"))
        if (path / "lit_model.pth").is_file()
    ]

    no_nonfinite_markers = re.search(r"\b(?:nan|inf)\b", text, flags=re.IGNORECASE) is None
    finite_losses = no_nonfinite_markers and all(math.isfinite(loss) for loss in val_losses + final_losses)
    passed = all(
        (
            train_returncode == 0,
            resume_returncode == 0,
            len(iteration_ms) >= 900,
            steady_tokens_per_second >= 10_000,
            active_gpu_mean >= 90,
            finite_losses,
            bool(final_losses),
            len(checkpoints) >= 1,
        )
    )
    receipt = {
        "passed": passed,
        "train_returncode": train_returncode,
        "resume_returncode": resume_returncode,
        "wall_seconds": wall_seconds,
        "resume_seconds": resume_seconds,
        "micro_batch_size": args.micro_batch_size,
        "trained_tokens": 1000 * args.micro_batch_size * 2048,
        "logged_iterations": len(iteration_ms),
        "median_steady_iteration_ms": median_iteration_ms,
        "steady_tokens_per_second": steady_tokens_per_second,
        "validation_losses": val_losses,
        "final_validation_losses": final_losses,
        "active_gpu_samples": len(active),
        "active_gpu_mean_percent": active_gpu_mean,
        "peak_memory_mib": peak_memory_mib,
        "active_power_mean_w": active_power_mean_w,
        "finite_losses": finite_losses,
        "no_nonfinite_markers": no_nonfinite_markers,
        "checkpoints": checkpoints,
        "data_receipt": str(data_receipt),
    }
    atomic_json(ROOT / "manifests/training-gate-receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("training gate failed; full data/training must not start")


if __name__ == "__main__":
    main()
