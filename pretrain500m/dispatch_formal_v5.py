#!/usr/bin/env python3
"""Freeze the capacity-corrected protocol and dispatch formal training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(os.environ.get("P500M_ROOT", "/ssd/scxi253/pretrain500m-20260912-v1"))
BASE_PROTOCOL_SHA256 = "38a8adbbba3ca19d2cd59718412fb6671925b6b123350d25a8755eed8661b893"
PILOT_PROTOCOL_SHA256 = "1b44371442a5c418c64f012ec5d9864718448d7bc9f81e56e7e04f34d551b4f3"
TRAIN_SHA256 = "3d9469d6bd3cc844ee690f37730b08051a8a38c4b2cf2d1d1b6c8377cf9655cc"
MINIMUM_MFU = 0.50


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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
    base_protocol_path = ROOT / "source" / "protocol-v3.json"
    train = ROOT / "source" / "train-v2" / "train.py"
    if digest(base_protocol_path) != BASE_PROTOCOL_SHA256 or digest(train) != TRAIN_SHA256:
        raise RuntimeError("base protocol or training code hash mismatch")

    data_manifest_path = ROOT / "data" / "fineweb-edu-formal-v1" / "manifest.json"
    data_manifest = load(data_manifest_path)
    expected_data = {
        "status": "FROZEN_VERIFIED_PACK",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "block_size": 2049,
        "document_exact_dedup": True,
    }
    for key, expected in expected_data.items():
        if data_manifest.get(key) != expected:
            raise RuntimeError(f"formal data {key}={data_manifest.get(key)!r}, expected {expected!r}")
    train_tokens = int(data_manifest.get("train_tokens", -1))
    unique_prediction_tokens = int(data_manifest.get("unique_prediction_tokens", -1))
    if not 8_000_000_000 <= train_tokens < 9_000_000_000 or train_tokens % 2049:
        raise RuntimeError(f"capacity-corrected raw token count is invalid: {train_tokens}")
    expected_unique = train_tokens // 2049 * 2048
    if unique_prediction_tokens != expected_unique:
        raise RuntimeError(
            f"unique prediction tokens are {unique_prediction_tokens}, expected {expected_unique}"
        )
    if sum(int(record["tokens"]) for record in data_manifest.get("train_shards", [])) != train_tokens:
        raise RuntimeError("formal train-shard token sum does not match the data manifest")
    if int(data_manifest.get("validation_tokens", 0)) <= 0:
        raise RuntimeError("formal validation data is empty")

    optimizer_prediction_tokens = 15_999_172_608
    effective_data_epochs = optimizer_prediction_tokens / unique_prediction_tokens
    protocol = load(base_protocol_path)
    protocol["protocol_id"] = "p500m-english-base-v5"
    protocol["supersedes"] = {
        "protocol_id": "p500m-english-base-v3",
        "sha256": BASE_PROTOCOL_SHA256,
        "reason": (
            "Capacity correction before formal training: the fully scanned, filtered, "
            "decontaminated, exact-deduplicated sample-10BT snapshot contains fewer than "
            "the previously planned 8,999,998,914 raw packed tokens. Use every verified "
            "block-aligned training token and retain the optimizer-token target."
        ),
    }
    protocol["training_data"]["planned_raw_packed_train_tokens"] = train_tokens
    protocol["training_data"]["planned_unique_prediction_tokens"] = unique_prediction_tokens
    protocol["training_data"]["planned_effective_prediction_tokens"] = optimizer_prediction_tokens
    protocol["training_data"]["planned_data_epochs"] = effective_data_epochs
    protocol["training_data"]["frozen_data_manifest_sha256"] = digest(data_manifest_path)
    protocol["training_data"]["capacity_policy"] = (
        "all verified block-aligned train shards after complete-source filtering, benchmark "
        "decontamination, within-part exact deduplication, and cross-part exact deduplication"
    )
    protocol_path = ROOT / "source" / "protocol-v5.json"
    if protocol_path.exists():
        existing = load(protocol_path)
        frozen_at_utc = existing.get("frozen_at_utc")
        protocol["frozen_at_utc"] = frozen_at_utc
        if not isinstance(frozen_at_utc, str) or existing != protocol:
            raise RuntimeError("protocol-v5 already exists with different frozen contents")
        protocol = existing
    else:
        protocol["frozen_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(protocol_path, protocol)
    protocol_sha256 = digest(protocol_path)

    # LR selection was frozen and launched under v2. Protocol v3 changes only
    # the formal scale gate, so the independently frozen v2 LR receipt remains authoritative.
    lr_receipt_path = ROOT / "formal" / "lr-selection-v2.json"
    lr_receipt = load(lr_receipt_path)
    if lr_receipt.get("status") != "PASS_FROZEN_LR_SELECTION":
        raise RuntimeError("frozen learning-rate selection did not pass")
    if (
        lr_receipt.get("protocol_sha256") != PILOT_PROTOCOL_SHA256
        or lr_receipt.get("train_sha256") != TRAIN_SHA256
    ):
        raise RuntimeError("learning-rate receipt code or pilot protocol mismatch")
    peak_lr = float(lr_receipt["selected_peak_lr"])

    measured = []
    for receipt_path in sorted((ROOT / "pilots").glob("scale-v*-*-nodes-*/scale-receipt.json")):
        receipt = load(receipt_path)
        manifest_path = receipt_path.parent / "run-manifest.json"
        manifest = load(manifest_path)
        record = {
            "receipt": str(receipt_path),
            "receipt_sha256": digest(receipt_path),
            "status": receipt.get("status"),
            "gpus": receipt.get("gpus"),
            "samples": receipt.get("samples"),
            "global_prediction_tokens_per_step": receipt.get("global_prediction_tokens_per_step"),
            "median_mfu": receipt.get("median_mfu"),
            "minimum_mfu": receipt.get("minimum_mfu"),
            "median_tokens_per_second": receipt.get("median_tokens_per_second"),
            "train_sha256": manifest.get("code_sha256", {}).get("train.py"),
            "protocol_sha256": manifest.get("protocol_sha256"),
        }
        complete = bool(
            int(record["samples"] or 0) >= 10
            and int(record["global_prediction_tokens_per_step"] or 0) == 2_097_152
            and int(record["gpus"] or 0) > 0
            and float(record["median_tokens_per_second"] or 0) > 0
        )
        record["admitted"] = bool(
            complete
            and float(record["median_mfu"] or 0) > MINIMUM_MFU
            and record["train_sha256"] == TRAIN_SHA256
            and record["protocol_sha256"] in {PILOT_PROTOCOL_SHA256, BASE_PROTOCOL_SHA256}
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
    if any("|p529m-formal-v" in line for line in active):
        raise RuntimeError(f"a formal training job is already active: {active}")
    run_root = ROOT / "formal" / "run-v5"
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"formal output already contains files: {run_root}")
    for output_name in ("hf-v5", "eval-v5"):
        output_root = ROOT / "formal" / output_name
        if output_root.exists() and any(output_root.iterdir()):
            raise RuntimeError(f"formal output already contains files: {output_root}")

    training_job, training_launch = submit(
        [
            f"--nodes={nodes}",
            f"--export=ALL,P500M_PEAK_LR={peak_lr:.10g},P500M_PROTOCOL_SHA256={protocol_sha256}",
            str(ROOT / "source" / "train-v2" / "train_ngpu_v5.sbatch"),
        ]
    )
    export_job, export_launch = submit(
        [
            f"--dependency=afterok:{training_job}",
            str(ROOT / "source" / "train-v2" / "export_hf_v5.sbatch"),
        ]
    )
    evaluation_job, evaluation_launch = submit(
        [
            f"--dependency=afterok:{export_job}",
            str(ROOT / "source" / "train-v2" / "evaluate_4gpu_v5.sbatch"),
        ]
    )
    receipt = {
        "status": "DISPATCHED_FORMAL_PIPELINE",
        "unix": time.time(),
        "owner_marker": owner,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "pilot_protocol_sha256": PILOT_PROTOCOL_SHA256,
        "train_sha256": TRAIN_SHA256,
        "minimum_mfu_strictly_greater_than": MINIMUM_MFU,
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": digest(data_manifest_path),
        "train_tokens": train_tokens,
        "unique_prediction_tokens": unique_prediction_tokens,
        "optimizer_prediction_tokens": optimizer_prediction_tokens,
        "effective_data_epochs": effective_data_epochs,
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
    output = ROOT / "formal" / "formal-dispatch-v5.json"
    atomic_json(output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
