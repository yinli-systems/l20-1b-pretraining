#!/usr/bin/env python3
"""Fail-closed supervisor: gate data -> train gate -> full data -> full pretrain."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/hhai/pretrain")
PROJECT = ROOT / "src/hq-pretrain"
STATUS = ROOT / "manifests/production-status.json"
EXPECTED_GATE_SOURCES = {"web", "synthetic", "math", "code"}
EXPECTED_FULL_SOURCES = {"web", "dclm", "math", "code"}


def atomic_status(stage: str, **values) -> None:
    payload = {"stage": stage, "updated_unix": time.time(), **values}
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, STATUS)
    print(json.dumps(payload, sort_keys=True), flush=True)


def complete_data_receipt(
    path: Path, expected_sources: set[str], expected_total: int | None = None
) -> bool:
    if not path.is_file():
        return False
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if set(receipt.get("sources", {})) != expected_sources or "completed_unix" not in receipt:
        return False
    if expected_total is not None:
        rounded = expected_total // 2049 * 2049
        # Each source target is rounded independently, so allow at most one block per source.
        actual = int(receipt.get("total_train_tokens", -1))
        # A resumed source may already contain a safe excess reservoir from an
        # earlier recipe. It is valid as long as every requested source target
        # was met; training weights, not reservoir size, determine consumption.
        if actual < rounded - (len(expected_sources) - 1) * 2049:
            return False
    return True


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def main() -> None:
    gate_data = ROOT / "manifests/data-gate-receipt.json"
    while not complete_data_receipt(gate_data, EXPECTED_GATE_SOURCES):
        atomic_status("waiting_for_gate_data")
        time.sleep(30)

    gate_receipt = ROOT / "manifests/training-gate-receipt.json"
    gate_passed = False
    if gate_receipt.is_file():
        gate_passed = bool(json.loads(gate_receipt.read_text()).get("passed"))
    if not gate_passed:
        atomic_status("running_training_gate")
        run_logged(
            [str(ROOT / ".venv/bin/python"), str(PROJECT / "run_gate.py"), "--micro-batch-size", "6"],
            ROOT / "logs/production-gate-supervisor.log",
        )
    gate = json.loads(gate_receipt.read_text())
    if not gate.get("passed"):
        raise RuntimeError("training gate did not pass; refusing full build and training")

    full_data = ROOT / "manifests/data-full-receipt.json"
    if not complete_data_receipt(full_data, EXPECTED_FULL_SOURCES, expected_total=20_200_000_000):
        atomic_status("building_full_data", gate_receipt=str(gate_receipt))
        run_logged(
            [
                str(ROOT / ".venv/bin/python"),
                str(PROJECT / "build_data.py"),
                "--stage",
                "full",
                "--workers",
                "4",
                "--pack-workers",
                "6",
            ],
            ROOT / "logs/production-full-data-supervisor.log",
        )
    if not complete_data_receipt(full_data, EXPECTED_FULL_SOURCES, expected_total=20_200_000_000):
        raise RuntimeError("full data receipt failed validation; refusing to train")

    atomic_status(
        "training_full",
        gate_receipt=str(gate_receipt),
        full_data_receipt=str(full_data),
        from_scratch=True,
    )
    run_logged(
        [
            str(ROOT / ".venv/bin/python"),
            str(PROJECT / "run_pretrain.py"),
            "--mode",
            "full",
            "--micro-batch-size",
            "6",
            "--resume",
            "--keep-step-checkpoints",
            "2",
        ],
        ROOT / "logs/train-full.log",
    )
    full_checkpoint = ROOT / "checkpoints/full/final/lit_model.pth"
    atomic_status("training_complete_evaluating", full_checkpoint=str(full_checkpoint))
    try:
        run_logged(
            [str(ROOT / ".venv/bin/python"), str(PROJECT / "evaluate_final.py")],
            ROOT / "logs/evaluate-final.log",
        )
    except subprocess.CalledProcessError as error:
        atomic_status(
            "training_complete_evaluation_failed",
            full_checkpoint=str(full_checkpoint),
            evaluation_error=str(error),
        )
        return
    atomic_status(
        "complete",
        full_checkpoint=str(full_checkpoint),
        evaluation_receipt=str(ROOT / "manifests/final-evaluation-receipt.json"),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_status("failed", error_type=type(error).__name__, error=str(error))
        raise
