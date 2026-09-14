"""Compare same-allocation ABBA DDP throughput trials without a checkpoint."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


LABELS = ("B_false_static_1", "A_true_dynamic_1", "A_true_dynamic_2", "B_false_static_2")
NUMERIC_KEYS = ("loss", "lr", "grad_norm")


def atomic_json(value, path):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_run(path):
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    train = [row for row in rows if isinstance(row.get("mfu"), (int, float))]
    validation = [row for row in rows if isinstance(row.get("val_loss_equal_domain"), (int, float))]
    status = json.loads((path / "training-status.json").read_text())
    manifest = json.loads((path / "run-manifest.json").read_text())
    if len(train) != 20 or not validation:
        raise ValueError(f"incomplete benchmark run: {path}")
    if status.get("status") != "QUALIFICATION_COMPLETED" or status.get("step") != 20:
        raise ValueError(f"bad benchmark status: {path}")
    steady = [float(row["mfu"]) for row in train if row["step"] > 5]
    return {
        "train": train,
        "validation": validation[-1],
        "manifest": manifest,
        "steady_mfu_median": statistics.median(steady),
        "final_mfu10_median": statistics.median(steady[-10:]),
    }


def max_numeric_difference(left, right):
    if len(left["train"]) != len(right["train"]):
        return math.inf
    maximum = 0.0
    for a, b in zip(left["train"], right["train"]):
        if a["step"] != b["step"] or a["prediction_tokens"] != b["prediction_tokens"]:
            return math.inf
        for key in NUMERIC_KEYS:
            maximum = max(maximum, abs(float(a[key]) - float(b[key])))
    maximum = max(
        maximum,
        abs(float(left["validation"]["val_loss_equal_domain"]) -
            float(right["validation"]["val_loss_equal_domain"])),
    )
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {label: read_run(args.run_dir / label) for label in LABELS}
    a = [runs["A_true_dynamic_1"]["steady_mfu_median"], runs["A_true_dynamic_2"]["steady_mfu_median"]]
    b = [runs["B_false_static_1"]["steady_mfu_median"], runs["B_false_static_2"]["steady_mfu_median"]]
    a_center = statistics.median(a)
    b_center = statistics.median(b)
    speedup = b_center / a_center
    differences = {
        "A_true_dynamic_1_vs_B_false_static_1": max_numeric_difference(runs["A_true_dynamic_1"], runs["B_false_static_1"]),
        "A_true_dynamic_2_vs_B_false_static_2": max_numeric_difference(runs["A_true_dynamic_2"], runs["B_false_static_2"]),
    }
    numerical_match = max(differences.values()) <= 1e-12
    mfu_pass = all(run["final_mfu10_median"] > 0.50 for run in runs.values())
    material_gain = speedup > 1.005
    adopt = numerical_match and mfu_pass and material_gain
    result = {
        "schema_version": 1,
        "status": "PASS_ADOPT" if adopt else
                  ("FAIL_NUMERICAL" if not numerical_match else
                   ("FAIL_MFU" if not mfu_pass else "NO_MATERIAL_GAIN")),
        "design": "same-allocation BAAB, identical seed/data/base/schedule, 20 steps each",
        "variant_A": "find_unused_parameters=true, static_graph=false",
        "variant_B": "find_unused_parameters=false, static_graph=true",
        "steady_steps": "6..20",
        "per_run": {label: {
            "steady_mfu_median": run["steady_mfu_median"],
            "final_mfu10_median": run["final_mfu10_median"],
            "val_loss_equal_domain": run["validation"]["val_loss_equal_domain"],
            "run_fingerprint_sha256": run["manifest"]["run_fingerprint_sha256"],
        } for label, run in runs.items()},
        "variant_A_median_mfu": a_center,
        "variant_B_median_mfu": b_center,
        "B_over_A_speedup": speedup,
        "minimum_material_speedup": 1.005,
        "max_numerical_difference": max(differences.values()),
        "pair_differences": differences,
        "numerical_match_at_1e-12": numerical_match,
        "all_final_mfu10_strictly_above_0_50": mfu_pass,
        "adopt_find_unused_false_static_graph_true": adopt,
        "claim_boundary": "qualification-only throughput result; no quality or promotion claim",
    }
    atomic_json(result, args.run_dir / "comparison.json")
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    print("COMPARISON_SHA256", hashlib.sha256((payload + "\n").encode()).hexdigest())


if __name__ == "__main__":
    main()
