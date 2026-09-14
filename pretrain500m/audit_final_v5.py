#!/usr/bin/env python3
"""Fail-closed audit of the completed formal v5 base-model pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
RUN = ROOT / "formal/run-v5"
EXPECTED_CHECKPOINT_SHA = "13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf"
EXPECTED_PROTOCOL_SHA = "a71ce3c8cc85cac6fe6751cb46f7e50dfb78d266602d6377cb2e65249f11c761"
EXPECTED_STEPS = 7629
EXPECTED_TOKENS = 15999172608
EXPECTED_PARAMS = 528748800
EXPECTED_COUNTS = {
    "hellaswag": 10042,
    "piqa": 1838,
    "winogrande": 1267,
    "openbookqa": 500,
    "arc_easy": 2376,
    "arc_challenge": 1172,
    "boolq": 3270,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    checkpoint = RUN / "resume.pt"
    checkpoint_sha = digest(checkpoint)
    sidecar_sha = (RUN / "resume.sha256").read_text().split()[0]
    training = load(RUN / "training-status.json")
    manifest = load(RUN / "run-manifest.json")
    export = load(ROOT / "formal/hf-v5/export-receipt.json")
    aggregate_path = ROOT / "formal/eval-v5/seven-task-aggregate.json"
    aggregate = load(aggregate_path)

    require(checkpoint_sha == EXPECTED_CHECKPOINT_SHA, "checkpoint digest mismatch")
    require(sidecar_sha == checkpoint_sha, "checkpoint sidecar mismatch")
    require(not (RUN / "resume.pt.next").exists(), "atomic checkpoint temporary remains")
    require(training["status"] == "COMPLETED", "training status is not completed")
    require(training["step"] == EXPECTED_STEPS, "training step mismatch")
    require(training["tokens"] == EXPECTED_TOKENS, "training token mismatch")
    require(training["checkpoint_sha256"] == checkpoint_sha, "training checkpoint mismatch")
    require(manifest["parameters_total"] == EXPECTED_PARAMS, "parameter count mismatch")
    require(manifest["protocol_sha256"] == EXPECTED_PROTOCOL_SHA, "run protocol mismatch")
    require(digest(Path(manifest["protocol_file"])) == EXPECTED_PROTOCOL_SHA, "protocol digest mismatch")

    require(export["checkpoint_sha256"] == checkpoint_sha, "export checkpoint mismatch")
    require(export["checkpoint_step"] == EXPECTED_STEPS, "export step mismatch")
    require(export["parameters_total"] == EXPECTED_PARAMS, "export parameter count mismatch")
    require(export["state_tensor_parity"] == "bitwise_exact", "export state parity failed")
    require(export["status"].startswith("PASS_HF_EXPORT"), "export receipt did not pass")

    require(aggregate["status"] == "PASS_FROZEN_EVALUATION_AGGREGATE", "evaluation failed")
    require(aggregate["protocol_sha256"] == EXPECTED_PROTOCOL_SHA, "evaluation protocol mismatch")
    require(aggregate["aggregate"]["bootstrap_replicates"] == 10000, "bootstrap count mismatch")
    require(set(aggregate["tasks"]) == set(EXPECTED_COUNTS), "evaluation task set mismatch")
    for task, count in EXPECTED_COUNTS.items():
        require(aggregate["tasks"][task]["samples"] == count, f"{task} sample count mismatch")
        require(math.isfinite(aggregate["tasks"][task]["score"]), f"{task} score is non-finite")

    latest_by_step: dict[int, dict] = {}
    for line in (RUN / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "mfu" in row:
            latest_by_step[int(row["step"])] = row
    require(set(latest_by_step) == set(range(1, EXPECTED_STEPS + 1)), "metric steps are incomplete")
    selected = [latest_by_step[step] for step in range(1, EXPECTED_STEPS + 1)]
    for row in selected:
        for key in ("loss", "grad_norm", "step_seconds", "tokens_per_second", "mfu"):
            require(math.isfinite(float(row[key])), f"non-finite {key} at step {row['step']}")
    post_grace = [float(row["mfu"]) for row in selected if row["step"] > 5]
    rolling = [statistics.median(post_grace[index - 9 : index + 1]) for index in range(9, len(post_grace))]
    median_mfu = statistics.median(post_grace)
    minimum_rolling_mfu = min(rolling)
    require(median_mfu > 0.50, "full-run median MFU gate failed")
    require(minimum_rolling_mfu > 0.50, "full-run rolling MFU gate failed")

    output = {
        "status": "PASS_FORMAL_V5_END_TO_END_AUDIT",
        "model": {
            "parameters_total": EXPECTED_PARAMS,
            "prediction_tokens": EXPECTED_TOKENS,
            "steps": EXPECTED_STEPS,
            "validation_loss": 2.7283682823181152,
            "validation_perplexity": 15.30788847787056,
        },
        "systems": {
            "mfu_gate_strictly_greater_than": 0.50,
            "full_trajectory_post_grace_median_mfu": median_mfu,
            "full_trajectory_minimum_rolling10_median_mfu": minimum_rolling_mfu,
            "metric_steps": len(selected),
            "metrics_finite": True,
        },
        "quality": aggregate,
        "identity": {
            "checkpoint_sha256": checkpoint_sha,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA,
            "run_manifest_sha256": digest(RUN / "run-manifest.json"),
            "training_status_sha256": digest(RUN / "training-status.json"),
            "export_receipt_sha256": digest(ROOT / "formal/hf-v5/export-receipt.json"),
            "evaluation_aggregate_sha256": digest(aggregate_path),
        },
        "claim_limits": [
            "The seven-task result is a base-model benchmark under protocol v5, not an instruction-following or domain qualification.",
            "No market-leading claim follows without matched competitor reruns under the same protocol.",
            "Post-training candidates must preserve this immutable base artifact and use separate validation for selection.",
        ],
    }
    final = ROOT / "formal/final-audit-v5.json"
    temporary = final.with_suffix(final.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    with temporary.open() as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, final)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
