#!/usr/bin/env python3
"""Summarize one matched-pair task in the frozen margin-aware export grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--array-task-id", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    pair = plan["checkpoint_pairs"][args.array_task_id]
    rows = []
    missing = []
    for sample_seed in plan["sample_seeds"]:
        for checkpoint in pair["checkpoints"]:
            cell_id = f"sample{sample_seed}-{checkpoint['id']}"
            diagnostic_path = args.output_root / cell_id / "diagnostic.json"
            receipt_path = args.output_root / cell_id / "export-receipt.json"
            if not diagnostic_path.is_file():
                missing.append(cell_id)
                continue
            diagnostic = json.loads(diagnostic_path.read_text())
            if diagnostic["sample_seed"] != sample_seed:
                raise RuntimeError(f"sample seed mismatch for {cell_id}")
            if diagnostic["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint mismatch for {cell_id}")
            rows.append({
                "cell_id": cell_id,
                "sample_seed": sample_seed,
                "checkpoint_id": checkpoint["id"],
                "checkpoint_role": checkpoint["role"],
                "status": diagnostic["status"],
                "state_tensor_parity": diagnostic["state_tensor_parity"],
                "argmax_matches": diagnostic["argmax_matches"],
                "argmax_total": diagnostic["argmax_total"],
                "stable_position_matches": diagnostic["stable_position_matches"],
                "stable_position_total": diagnostic["stable_position_total"],
                "max_abs_logit_difference": diagnostic["max_abs_logit_difference"],
                "mean_abs_logit_difference": diagnostic["mean_abs_logit_difference"],
                "diagnostic_sha256": digest(diagnostic_path),
                "export_receipt_written": receipt_path.is_file(),
                "export_receipt_sha256": digest(receipt_path) if receipt_path.is_file() else None,
            })
    passes = sum(row["status"] == "PASS_FROZEN_MARGIN_AWARE_GATE" for row in rows)
    status = (
        "INCOMPLETE_TASK_GRID"
        if missing
        else "PASS_ALL_EIGHT_MARGIN_AWARE_CELLS"
        if passes == 8
        else "ONE_OR_MORE_MARGIN_AWARE_CELLS_FAILED"
    )
    result = {
        "schema": "p529m-f2-margin-aware-export-task-result-v1",
        "status": status,
        "array_job_id": int(args.array_job_id),
        "array_task_id": args.array_task_id,
        "pair_seed": pair["pair_seed"],
        "plan_sha256": digest(args.plan),
        "expected_cells": 8,
        "completed_cells": len(rows),
        "pass_cells": passes,
        "fail_cells": len(rows) - passes,
        "missing_cells": missing,
        "cells": rows,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output_root / "task-result.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
