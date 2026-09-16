#!/usr/bin/env python3
"""Summarize the complete frozen checkpoint-by-sample diagnostic grid."""

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
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "p529m-f2-export-parity-independent-samples-plan-v1":
        raise RuntimeError("unexpected plan schema")

    cells = []
    missing = []
    for sample_seed in plan["sample_seeds"]:
        for checkpoint in plan["checkpoint_grid"]:
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
            cells.append({
                "cell_id": cell_id,
                "sample_seed": sample_seed,
                "checkpoint_id": checkpoint["id"],
                "checkpoint_seed": checkpoint["seed"],
                "status": diagnostic["status"],
                "state_tensor_parity": diagnostic["state_tensor_parity"],
                "argmax_matches": diagnostic["argmax_matches"],
                "argmax_total": diagnostic["argmax_total"],
                "argmax_agreement": diagnostic["argmax_agreement"],
                "max_abs_logit_difference": diagnostic["max_abs_logit_difference"],
                "mean_abs_logit_difference": diagnostic["mean_abs_logit_difference"],
                "mismatch_count": len(diagnostic["mismatches"]),
                "diagnostic_sha256": digest(diagnostic_path),
                "formal_export_receipt_written": receipt_path.is_file(),
                "formal_export_receipt_sha256": digest(receipt_path) if receipt_path.is_file() else None,
            })

    statuses = [cell["status"] for cell in cells]
    all_pass = len(cells) == 4 and all(status == "PASS_ORIGINAL_FROZEN_GATE" for status in statuses)
    status = (
        "INCOMPLETE_DIAGNOSTIC_GRID"
        if missing
        else "PASS_ALL_UNCHANGED_ORIGINAL_GATES"
        if all_pass
        else "ONE_OR_MORE_UNCHANGED_ORIGINAL_GATES_FAILED"
    )
    result = {
        "schema": "p529m-f2-export-parity-independent-samples-result-v1",
        "status": status,
        "job_id": int(args.job_id),
        "plan_sha256": digest(args.plan),
        "expected_cells": 4,
        "completed_cells": len(cells),
        "missing_cells": missing,
        "pass_cells": sum(value == "PASS_ORIGINAL_FROZEN_GATE" for value in statuses),
        "fail_cells": sum(value == "FAIL_ORIGINAL_FROZEN_GATE" for value in statuses),
        "cells": cells,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output_root / "result.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
