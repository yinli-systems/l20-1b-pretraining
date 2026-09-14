#!/usr/bin/env python3
"""Fail-closed formal-training preflight and Slurm dispatch."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
PROTOCOL_SHA256 = "1b44371442a5c418c64f012ec5d9864718448d7bc9f81e56e7e04f34d551b4f3"
TRAIN_SHA256 = "3d9469d6bd3cc844ee690f37730b08051a8a38c4b2cf2d1d1b6c8377cf9655cc"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def submit(arguments: list[str]) -> tuple[str, str]:
    launch = subprocess.check_output(["sbatch", *arguments], text=True).strip()
    match = re.fullmatch(r"Submitted batch job (\d+)", launch)
    if not match:
        raise RuntimeError(f"unexpected sbatch response: {launch}")
    return match.group(1), launch


def main() -> None:
    owner = (ROOT / "OWNER.txt").read_text().strip()
    if "owner task 01a09290-b43f-7431-be8a-412ea5d37954" not in owner:
        raise RuntimeError("owner marker mismatch")
    protocol = ROOT / "source" / "protocol-v2.json"
    train = ROOT / "source" / "train-v2" / "train.py"
    if digest(protocol) != PROTOCOL_SHA256 or digest(train) != TRAIN_SHA256:
        raise RuntimeError("formal protocol or training code hash mismatch")

    data_manifest_path = ROOT / "data" / "fineweb-edu-formal-v1" / "manifest.json"
    data_manifest = load(data_manifest_path)
    expected_data = {
        "status": "FROZEN_VERIFIED_PACK",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "block_size": 2049,
        "train_tokens": 8_999_998_914,
        "unique_prediction_tokens": 8_995_606_528,
        "document_exact_dedup": True,
    }
    for key, expected in expected_data.items():
        if data_manifest.get(key) != expected:
            raise RuntimeError(f"formal data {key}={data_manifest.get(key)!r}, expected {expected!r}")

    lr_receipt_path = ROOT / "formal" / "lr-selection-v2.json"
    lr_receipt = load(lr_receipt_path)
    if lr_receipt.get("status") != "PASS_FROZEN_LR_SELECTION":
        raise RuntimeError("frozen learning-rate selection did not pass")
    if lr_receipt.get("protocol_sha256") != PROTOCOL_SHA256 or lr_receipt.get("train_sha256") != TRAIN_SHA256:
        raise RuntimeError("learning-rate receipt code or protocol mismatch")
    peak_lr = float(lr_receipt["selected_peak_lr"])

    measured = []
    for receipt_path in sorted((ROOT / "pilots").glob("scale-v2-*-nodes-*/scale-receipt.json")):
        receipt = load(receipt_path)
        manifest_path = receipt_path.parent / "run-manifest.json"
        manifest = load(manifest_path)
        record = {
            "receipt": str(receipt_path),
            "receipt_sha256": digest(receipt_path),
            "status": receipt.get("status"),
            "gpus": receipt.get("gpus"),
            "median_mfu": receipt.get("median_mfu"),
            "median_tokens_per_second": receipt.get("median_tokens_per_second"),
            "train_sha256": manifest.get("code_sha256", {}).get("train.py"),
            "protocol_sha256": manifest.get("protocol_sha256"),
        }
        record["admitted"] = bool(
            record["status"] == "PASS_MFU_STRICTLY_GREATER_THAN_0.65"
            and float(record["median_mfu"]) > 0.65
            and record["train_sha256"] == TRAIN_SHA256
            and record["protocol_sha256"] == PROTOCOL_SHA256
        )
        measured.append(record)
    admitted = [record for record in measured if record["admitted"]]
    if not admitted:
        raise RuntimeError(f"no current-code scale receipt passes the strict MFU gate: {measured}")
    selected = max(admitted, key=lambda record: float(record["median_tokens_per_second"]))
    gpus = int(selected["gpus"])
    if gpus % 4 or 256 % gpus:
        raise RuntimeError(f"selected GPU count is incompatible with the frozen batch: {gpus}")
    nodes = gpus // 4

    active = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%j|%T"], text=True
    ).splitlines()
    if any("|p529m-formal-v2|" in line for line in active):
        raise RuntimeError(f"a formal training job is already active: {active}")
    run_root = ROOT / "formal" / "run-v2"
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"formal output already contains files: {run_root}")
    for output_name in ("hf-v2", "eval-v2"):
        output_root = ROOT / "formal" / output_name
        if output_root.exists() and any(output_root.iterdir()):
            raise RuntimeError(f"formal output already contains files: {output_root}")

    training_job, training_launch = submit(
        [
            f"--nodes={nodes}",
            f"--export=ALL,P500M_PEAK_LR={peak_lr:.10g}",
            str(ROOT / "source" / "train-v2" / "train_ngpu.sbatch"),
        ]
    )
    export_job, export_launch = submit(
        [
            f"--dependency=afterok:{training_job}",
            str(ROOT / "source" / "train-v2" / "export_hf.sbatch"),
        ]
    )
    evaluation_job, evaluation_launch = submit(
        [
            f"--dependency=afterok:{export_job}",
            str(ROOT / "source" / "train-v2" / "evaluate_4gpu.sbatch"),
        ]
    )
    receipt = {
        "status": "DISPATCHED_FORMAL_PIPELINE",
        "unix": time.time(),
        "owner_marker": owner,
        "protocol_sha256": PROTOCOL_SHA256,
        "train_sha256": TRAIN_SHA256,
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": digest(data_manifest_path),
        "lr_selection_receipt": str(lr_receipt_path),
        "lr_selection_receipt_sha256": digest(lr_receipt_path),
        "selected_peak_lr": peak_lr,
        "measured_scale_candidates": measured,
        "selected_scale": selected,
        "selected_nodes": nodes,
        "selected_gpus": gpus,
        "active_jobs_at_dispatch": active,
        "training_job": training_job,
        "training_sbatch_response": training_launch,
        "export_job": export_job,
        "export_sbatch_response": export_launch,
        "evaluation_job": evaluation_job,
        "evaluation_sbatch_response": evaluation_launch,
    }
    output = ROOT / "formal" / "formal-dispatch-v2.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
