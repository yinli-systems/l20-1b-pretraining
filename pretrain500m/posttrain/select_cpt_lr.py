"""Select a continued-pretraining LR pilot using only fixed development loss."""
import argparse
import json
import math
import statistics
from pathlib import Path


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-mfu", type=float, default=0.50)
    args = parser.parse_args()

    expected = ("0.00003", "0.0001", "0.0003")
    candidates = []
    for lr in expected:
        run = args.pilot_root / f"lr-{lr}"
        status = load_json(run / "training-status.json")
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        train = [row for row in rows if "mfu" in row]
        validation = [row for row in rows if "val_loss" in row]
        if status.get("status") != "COMPLETED":
            raise RuntimeError(f"candidate {lr} did not complete: {status}")
        if len(validation) < 2 or validation[0].get("phase") != "before_training":
            raise RuntimeError(f"candidate {lr} lacks pre/post validation")
        numeric = [value for row in rows for value in row.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError(f"candidate {lr} contains non-finite metrics")
        steady_mfu = statistics.median(row["mfu"] for row in train[5:])
        if steady_mfu <= args.minimum_mfu:
            raise RuntimeError(f"candidate {lr} median MFU {steady_mfu} did not pass")
        candidates.append({
            "learning_rate": float(lr),
            "initial_val_loss": validation[0]["val_loss"],
            "final_val_loss": validation[-1]["val_loss"],
            "final_step": status["step"],
            "median_mfu_after_grace": steady_mfu,
            "output_directory": str(run),
        })

    initial = [candidate["initial_val_loss"] for candidate in candidates]
    if max(initial) - min(initial) > 1e-7:
        raise RuntimeError(f"initial validation mismatch across candidates: {initial}")
    selected = min(candidates, key=lambda candidate: candidate["final_val_loss"])
    improvement = initial[0] - selected["final_val_loss"]
    receipt = {
        "candidates": candidates,
        "development_loss_improvement": improvement,
        "frozen_seven_task_test_used_for_selection": False,
        "selected_learning_rate": selected["learning_rate"] if improvement > 0 else None,
        "status": "PASS_CPT_PILOT" if improvement > 0 else "NO_CPT_PROMOTION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
