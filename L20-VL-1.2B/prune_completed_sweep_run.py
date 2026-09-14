#!/usr/bin/env python3
"""Retain only a selected sweep checkpoint after auditable completion."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


def checkpoint_record(path: Path) -> dict:
    files = {}
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        relative = str(file.relative_to(path))
        record = {"bytes": file.stat().st_size}
        if file.name != "optimizer.pt":
            record["sha256"] = sha256_file(file)
        files[relative] = record
    return {"path": str(path), "bytes": sum(item["bytes"] for item in files.values()), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-optimizer-steps", type=int, default=875)
    args = parser.parse_args()
    if args.expected_optimizer_steps < 1:
        raise SystemExit("expected optimizer steps must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    run_root = args.run.resolve()
    run = json.loads((run_root / "run.json").read_text())
    selection = json.loads(args.selection.read_text())
    if (
        run.get("status") != "complete"
        or run.get("optimizer_step") != args.expected_optimizer_steps
    ):
        raise RuntimeError(
            "only a complete run at the declared optimizer-step boundary may be pruned"
        )
    selected = Path(selection["selected_checkpoint"]).resolve()
    if selected.parent != run_root or not selected.is_dir():
        raise RuntimeError("selected checkpoint is not an immediate child of the declared run")
    if selection.get("protocol_arm") != run.get("arm"):
        raise RuntimeError("selection arm does not match run arm")
    if sha256_file(selected / "bridge.safetensors") != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("selected bridge hash mismatch before pruning")
    if sha256_file(selected / "language_adapter" / "adapter_model.safetensors") != selection["selected_language_adapter_model_sha256"]:
        raise RuntimeError("selected adapter hash mismatch before pruning")
    checkpoints = sorted(path for path in run_root.glob("step-*") if path.is_dir())
    records = [checkpoint_record(path) for path in checkpoints]
    bytes_before = sum(item["bytes"] for item in records)
    deleted = []
    for path in checkpoints:
        if path == selected:
            optimizer = path / "optimizer.pt"
            if optimizer.exists():
                deleted.append({"path": str(optimizer), "bytes": optimizer.stat().st_size})
                optimizer.unlink()
        else:
            deleted.append({
                "path": str(path),
                "bytes": next(item["bytes"] for item in records if item["path"] == str(path)),
            })
            shutil.rmtree(path)
    remaining = sorted(path for path in run_root.glob("step-*") if path.is_dir())
    if remaining != [selected] or (selected / "optimizer.pt").exists():
        raise RuntimeError("post-prune checkpoint layout is not exact")
    if sha256_file(selected / "bridge.safetensors") != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("selected bridge changed during pruning")
    if sha256_file(selected / "language_adapter" / "adapter_model.safetensors") != selection["selected_language_adapter_model_sha256"]:
        raise RuntimeError("selected adapter changed during pruning")
    bytes_after = sum(file.stat().st_size for file in selected.rglob("*") if file.is_file())
    receipt = {
        "schema_version": "2026-09-13-v1",
        "status": "complete",
        "completed_at": utc_now(),
        "run": str(run_root),
        "arm": run["arm"],
        "expected_optimizer_steps": args.expected_optimizer_steps,
        "selection_receipt": str(args.selection),
        "selection_receipt_sha256": sha256_file(args.selection),
        "selected_checkpoint": str(selected),
        "selected_step": selection["selected_step"],
        "selected_bridge_sha256": selection["selected_checkpoint_sha256"],
        "selected_adapter_model_sha256": selection["selected_language_adapter_model_sha256"],
        "checkpoint_inventory_before": records,
        "deleted": deleted,
        "checkpoint_bytes_before": bytes_before,
        "checkpoint_bytes_after": bytes_after,
        "bytes_reclaimed": bytes_before - bytes_after,
        "postcondition": "only the selected checkpoint remains and its optimizer state is removed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, receipt)
    print(json.dumps({
        "arm": receipt["arm"],
        "selected_step": receipt["selected_step"],
        "bytes_reclaimed": receipt["bytes_reclaimed"],
        "status": "complete",
    }, indent=2))


if __name__ == "__main__":
    main()
