#!/usr/bin/env python3
"""Run and remove two bounded Stage-D trainer smokes after preserving a receipt."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "authorized_stage_d_lr_microsearch":
        raise RuntimeError("smoke requires the frozen Stage-D LR-screen protocol")
    results = []
    paths = []
    for arm in ("F_matched_196", "A_spatial_49"):
        run = ROOT / "runs" / f"stage-d-smoke-{arm}-v1"
        if run.exists():
            raise FileExistsError(run)
        paths.append(run)
        subprocess.run([
            sys.executable,
            str(ROOT / "train_stage_c_counterfactual.py"),
            "--protocol", str(args.protocol),
            "--arm", arm,
            "--seed", str(protocol["optimization"]["seed"]),
            "--learning-rate-multiplier", "1.0",
            "--max-optimizer-steps", "2",
            "--max-wall-seconds", "600",
            "--num-workers", "0",
            "--output", str(run),
        ], cwd=ROOT, check=True)
        state = json.loads((run / "run.json").read_text())
        checkpoint = run / "step-000002"
        result = {
            "arm": arm,
            "status": state["status"],
            "optimizer_step": state["optimizer_step"],
            "output_visual_tokens": state["output_visual_tokens"],
            "teacher_cache_sha256": state["teacher_cache_sha256"],
            "finite_total_loss": math.isfinite(float(state["total_loss_mean"])),
            "bridge_sha256": sha256_file(checkpoint / "bridge.safetensors"),
            "adapter_model_sha256": sha256_file(checkpoint / "language_adapter" / "adapter_model.safetensors"),
        }
        expected_tokens = 196 if arm == "F_matched_196" else 49
        if not (
            result["status"] == "complete"
            and result["optimizer_step"] == 2
            and result["output_visual_tokens"] == expected_tokens
            and result["teacher_cache_sha256"] is None
            and result["finite_total_loss"]
        ):
            raise RuntimeError(f"Stage-D smoke gate failed: {result}")
        results.append(result)
    for path in paths:
        resolved = path.resolve()
        if resolved.parent != (ROOT / "runs").resolve() or not resolved.name.startswith("stage-d-smoke-"):
            raise RuntimeError(f"refusing unsafe smoke cleanup target: {resolved}")
        shutil.rmtree(resolved)
    receipt = {
        "schema_version": "2026-09-13-v1",
        "status": "pass",
        "completed_at": utc_now(),
        "protocol_sha256": sha256_file(args.protocol),
        "results": results,
        "smoke_directories_removed": all(not path.exists() for path in paths),
        "prior_test_predictions_produced": 0,
        "new_iid_test_predictions_produced": 0,
        "ood_test_predictions_produced": 0,
        "claim_boundary": "Two optimizer steps validate wiring only; they provide no efficacy evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
