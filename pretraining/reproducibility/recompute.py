#!/usr/bin/env python3
"""Fail closed while independently recomputing the checked-in evidence bundle."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from datetime import datetime

getcontext().prec = 50


class VerificationError(ValueError):
    """Raised when a bound input or recomputed result does not match."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}: {exc}") from exc


def require_close(name: str, actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise VerificationError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")


def verify_bound_files(root: Path, manifest: dict[str, Any]) -> None:
    for relative, expected_hash in manifest["bound_files"].items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise VerificationError(f"bound file missing or not regular: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise VerificationError(f"SHA-256 mismatch for {relative}")


def verify_training_sources(root: Path) -> dict[str, int]:
    manifest = load_json(root / "reproducibility/training-source-manifest.json")
    checked = 0
    for relative, expected in manifest["tracked_execution_files"].items():
        if sha256_file(root / relative) != expected:
            raise VerificationError(f"executed source hash mismatch: {relative}")
        checked += 1
    for relative, expected in manifest["snapshot_files"].items():
        if sha256_file(root / relative) != expected:
            raise VerificationError(f"execution snapshot hash mismatch: {relative}")
        checked += 1
    if not manifest["pre_run_environment_receipt_mismatches"]:
        raise VerificationError("source reconstruction unexpectedly hides pre-run receipt drift")
    return {"verified_source_files": checked, "pre_run_receipt_mismatches": len(manifest["pre_run_environment_receipt_mismatches"])}


def recompute_telemetry(root: Path, accounting: dict[str, Any]) -> dict[str, float | int]:
    path = root / accounting["full_run_telemetry"]["source_export"]
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise VerificationError("telemetry export is empty")
    total_steps = int(accounting["training"]["optimizer_steps"])
    tokens_per_step = int(accounting["training"]["tokens_per_step"])
    expected_window = 1
    samples = 0
    sum_flops = Decimal(0)
    reciprocal_flops = Decimal(0)
    covered_seconds = Decimal(0)
    previous_end = 0
    for row in rows:
        window = int(row["window"])
        declared_start = int(row["declared_step_start"])
        declared_end = int(row["declared_step_end"])
        if window != expected_window or declared_start != previous_end + 1 or declared_end < declared_start:
            raise VerificationError("telemetry windows are not contiguous")
        expected_window += 1
        previous_end = declared_end
        count = int(row["sample_count"])
        covered_start = int(row["covered_step_start"])
        covered_end = int(row["covered_step_end"])
        if count <= 0 or count != covered_end - covered_start + 1:
            raise VerificationError("telemetry window sample range is inconsistent")
        if not (declared_start <= covered_start <= covered_end <= declared_end):
            raise VerificationError("telemetry coverage falls outside its declared window")
        window_sum_flops = Decimal(row["sum_model_flops_per_second"])
        window_reciprocal_flops = Decimal(row["sum_reciprocal_model_flops_per_second"])
        window_mean_flops = Decimal(row["mean_model_flops_per_second"])
        window_harmonic_flops = Decimal(row["harmonic_model_flops_per_second"])
        window_sum_tokens_per_second = Decimal(row["sum_tokens_per_second"])
        window_harmonic_tokens_per_second = Decimal(row["harmonic_tokens_per_second"])
        window_logger_seconds = Decimal(row["covered_logger_seconds"])
        if min(window_sum_flops, window_reciprocal_flops, window_sum_tokens_per_second, window_logger_seconds) <= 0:
            raise VerificationError("telemetry window contains a non-positive aggregate")
        require_close(
            "window arithmetic FLOP/s",
            float(window_mean_flops),
            float(window_sum_flops / count),
            tolerance=2e-15,
        )
        require_close(
            "window harmonic FLOP/s",
            float(window_harmonic_flops),
            float(Decimal(count) / window_reciprocal_flops),
            tolerance=2e-15,
        )
        expected_harmonic_tokens = Decimal(count * tokens_per_step) / window_logger_seconds
        require_close(
            "window harmonic token/s",
            float(window_harmonic_tokens_per_second),
            float(expected_harmonic_tokens),
            tolerance=2e-12,
        )
        if window_sum_tokens_per_second / count < window_harmonic_tokens_per_second:
            raise VerificationError("window arithmetic token/s is below harmonic token/s")
        samples += count
        sum_flops += window_sum_flops
        reciprocal_flops += window_reciprocal_flops
        covered_seconds += window_logger_seconds

    if int(accounting["training"]["prediction_tokens"]) != total_steps * tokens_per_step:
        raise VerificationError("training prediction-token accounting does not multiply out")
    peak = float(accounting["full_run_telemetry"]["peak_denominator_flops_per_second"])
    step_weighted_flops = float(sum_flops / samples)
    time_weighted_flops = float(Decimal(samples) / reciprocal_flops)
    result = {
        "windows": len(rows),
        "samples": samples,
        "coverage": samples / total_steps,
        "step_weighted_model_flops_per_second": step_weighted_flops,
        "step_weighted_mfu": step_weighted_flops / peak,
        "time_weighted_model_flops_per_second": time_weighted_flops,
        "time_weighted_mfu": time_weighted_flops / peak,
        "covered_logger_seconds": float(covered_seconds),
        "aggregate_tokens_per_second": float(Decimal(samples * tokens_per_step) / covered_seconds),
    }
    expected = accounting["full_run_telemetry"]
    checks = {
        "telemetry sample count": (result["samples"], expected["model_flops_samples"]),
        "telemetry coverage": (result["coverage"], expected["sample_coverage"]),
        "step-weighted MFU": (result["step_weighted_mfu"], expected["mean_mfu_step_weighted"]),
        "time-weighted MFU": (result["time_weighted_mfu"], expected["aggregate_mfu_time_weighted"]),
        "aggregate tokens/s": (result["aggregate_tokens_per_second"], expected["aggregate_tokens_per_second_over_covered_logger_intervals"]),
        "covered logger seconds": (result["covered_logger_seconds"], expected["covered_logger_seconds"]),
    }
    for name, (actual, wanted) in checks.items():
        require_close(name, float(actual), float(wanted))
    return result


def verify_timing(accounting: dict[str, Any]) -> dict[str, float]:
    def parse(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    training = accounting["training"]
    elapsed = (parse(training["training_complete_utc"]) - parse(training["successful_supervisor_start_utc"])).total_seconds()
    require_close("successful supervisor elapsed", elapsed, training["successful_supervisor_elapsed_seconds"])
    require_close("successful supervisor hours", elapsed / 3600, training["successful_supervisor_elapsed_hours"])
    require_close("single-L20 elapsed GPU-hours", elapsed / 3600, training["single_l20_elapsed_gpu_hours"])

    data = accounting["data_preparation"]
    total = (parse(data["full_data_receipt_utc"]) - parse(data["first_full_data_supervisor_attempt_utc"])).total_seconds() / 3600
    admitted = (parse(data["full_data_receipt_utc"]) - parse(data["final_admitted_attempt_start_utc"])).total_seconds() / 3600
    require_close("data preparation calendar hours", total, data["first_attempt_to_receipt_calendar_hours"])
    require_close("final admitted data attempt hours", admitted, data["final_admitted_attempt_calendar_hours"])
    require_close("prior data attempt overhead", total - admitted, data["prior_attempt_and_wait_calendar_hours"])
    return {
        "successful_training_supervisor_hours": elapsed / 3600,
        "data_first_attempt_to_receipt_hours": total,
        "data_final_admitted_attempt_hours": admitted,
    }


def verify_final_metrics(root: Path, accounting: dict[str, Any]) -> dict[str, float | int]:
    validation_rows = list(csv.DictReader((root / "reports/metrics/validation-loss.csv").open(encoding="utf-8")))
    final = validation_rows[-1]
    final_step = int(final["step"])
    final_loss = float(final["loss"])
    final_ppl = float(final["ppl"])
    require_close("final step", final_step, accounting["training"]["optimizer_steps"])
    # The compact CSV intentionally stores nine decimal places; the accounting
    # receipt retains the source float32 values from the event file.
    require_close(
        "final validation loss",
        final_loss,
        accounting["training"]["final_validation_loss"],
        tolerance=5e-10,
    )
    require_close(
        "final validation perplexity",
        final_ppl,
        accounting["training"]["final_validation_perplexity"],
        tolerance=5e-10,
    )
    # val_ppl was emitted after float32 tensor exponentiation, so its relative
    # agreement with a host float64 exp is bounded at float32 precision.
    require_close(
        "perplexity equals exp(loss)",
        math.exp(float(accounting["training"]["final_validation_loss"])),
        float(accounting["training"]["final_validation_perplexity"]),
        tolerance=3e-8,
    )

    data = load_json(root / "reports/receipts/data-full-receipt.json")
    train_tokens = sum(source["actual_train_tokens"] for source in data["sources"].values())
    validation_tokens = sum(source["actual_validation_tokens"] for source in data["sources"].values())
    if train_tokens != data["total_train_tokens"] or validation_tokens != data["total_validation_tokens"]:
        raise VerificationError("data receipt source totals do not add up")
    if train_tokens != accounting["data_preparation"]["total_train_reservoir_tokens"]:
        raise VerificationError("accounting data token total mismatch")

    benchmarks = load_json(root / "reports/metrics/final-benchmarks.json")
    zero = benchmarks["zero_shot_core"]
    selected_scores = [value for name, value in zero.items() if not name.endswith("_perplexity")]
    if len(selected_scores) != 8:
        raise VerificationError("expected exactly eight selected zero-shot task metrics")
    recomputed_core = sum(selected_scores) / len(selected_scores)
    require_close("eight-task core mean", recomputed_core, benchmarks["selected_metric_core_mean"])
    return {
        "final_step": final_step,
        "final_validation_loss": final_loss,
        "final_validation_perplexity": final_ppl,
        "train_reservoir_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "eight_task_core_mean": recomputed_core,
    }


def verify_release_binding(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    receipt = load_json(root / "reports/receipts/base-release-verification-20260913.json")
    expected = manifest["model_release"]
    hub = receipt["hub"]
    if hub["repository"] != expected["repository"] or hub["revision"] != expected["revision"]:
        raise VerificationError("Hugging Face release identity mismatch")
    if not hub["byte_exact_manifest_match"] or not receipt["structural_verification"]["valid"]:
        raise VerificationError("Hugging Face release verification is not valid")
    return {
        "repository": hub["repository"],
        "revision": hub["revision"],
        "parameter_count": receipt["structural_verification"]["parameter_count"],
        "tensor_count": receipt["structural_verification"]["tensor_count"],
    }


def verify_efficiency_result(root: Path) -> dict[str, Any]:
    bundle = load_json(root / "reports/metrics/efficiency-all-results-final-20260912.json")
    if bundle["status"] != "verified_complete" or bundle["checkpoint_count"] != 36:
        raise VerificationError("efficiency result bundle is not the complete frozen comparison")
    if bundle["task_comparison_count"] != 36 * 7:
        raise VerificationError("efficiency task-comparison count is inconsistent")
    candidates = [item for item in bundle["checkpoints"] if item["job"] == "tinyllama_1t"]
    if len(candidates) != 1:
        raise VerificationError("expected exactly one TinyLlama 1T comparison")
    comparison = candidates[0]
    plan_path = root / "reports/final-efficiency-20260912/efficiency-evaluation-plan-20260912-v5.json"
    verification_path = root / "reports/final-efficiency-20260912/final-verification-20260912.json"
    summary_path = root / "reports/final-efficiency-20260912/results/tinyllama_1t-summary.json"
    verification = load_json(verification_path)
    summary = load_json(summary_path)
    binding = verification["result_bindings"]["tinyllama_1t"]
    if verification["status"] != "verified_complete" or verification["complete_jobs"] != 36:
        raise VerificationError("frozen efficiency verification is incomplete")
    if sha256_file(plan_path) != bundle["plan_sha256"] or verification["plan_sha256"] != bundle["plan_sha256"]:
        raise VerificationError("frozen efficiency plan binding mismatch")
    if sha256_file(verification_path) != bundle["final_verification_sha256"]:
        raise VerificationError("frozen efficiency verification hash mismatch")
    if sha256_file(summary_path) != binding["summary_sha256"]:
        raise VerificationError("TinyLlama summary binding mismatch")
    if summary["result_sha256"] != binding["result_sha256"] or summary["result_sha256"] != comparison["result_sha256"]:
        raise VerificationError("TinyLlama raw-result binding mismatch")
    if summary["tasks"] != comparison["tasks"]:
        raise VerificationError("TinyLlama compact summary changed during publication")
    task_ours: list[float] = []
    task_baseline: list[float] = []
    for task, result in comparison["tasks"].items():
        sample_count = int(result["n"])
        ours_better = int(result["ours_better_samples"])
        baseline_better = int(result["baseline_better_samples"])
        tied = int(result["tied_samples"])
        if ours_better + baseline_better + tied != sample_count:
            raise VerificationError(f"paired outcome counts do not add up for {task}")
        ours_successes = round(float(result["ours"]) * sample_count)
        baseline_successes = round(float(result["baseline"]) * sample_count)
        require_close(f"ours score grid for {task}", ours_successes / sample_count, result["ours"])
        require_close(
            f"baseline score grid for {task}",
            baseline_successes / sample_count,
            result["baseline"],
        )
        both_correct = ours_successes - ours_better
        both_wrong = tied - both_correct
        if min(both_correct, both_wrong) < 0:
            raise VerificationError(f"paired sufficient statistics are impossible for {task}")
        require_close(
            f"difference for {task}",
            (result["ours"] - result["baseline"]) * 100,
            result["difference_pp"],
        )
        task_ours.append(float(result["ours"]))
        task_baseline.append(float(result["baseline"]))

    macro = comparison["seven_task_macro"]
    ours_macro = sum(task_ours) / len(task_ours)
    baseline_macro = sum(task_baseline) / len(task_baseline)
    require_close("TinyLlama comparison ours macro", ours_macro, macro["ours"])
    require_close("TinyLlama comparison baseline macro", baseline_macro, macro["baseline"])
    require_close(
        "TinyLlama comparison macro difference",
        (ours_macro - baseline_macro) * 100,
        macro["ours_minus_baseline_pp"],
    )
    return {
        "baseline": comparison["repo"],
        "revision": comparison["revision"],
        "tasks": len(task_ours),
        "ours_macro": ours_macro,
        "baseline_macro": baseline_macro,
        "difference_percentage_points": macro["ours_minus_baseline_pp"],
        "reported_paired_95_ci_percentage_points": macro["paired_95_ci_pp"],
        "ci_boundary": "Use recompute_efficiency_ci.py with NumPy 2.5.2 for exact statistical replay. Independent prediction verification still requires raw or regenerated item outputs.",
    }


def verify_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = load_json(root / "reproducibility/manifest.json")
    verify_bound_files(root, manifest)
    accounting = load_json(root / "reports/receipts/full-training-accounting-v1.json")
    if accounting["energy"]["full_training_energy_kwh"] is not None:
        raise VerificationError("bundle must not infer unmeasured full-run energy")
    if accounting["point_in_time_snapshot"]["mfu"] == accounting["full_run_telemetry"]["aggregate_mfu_time_weighted"]:
        raise VerificationError("snapshot MFU was incorrectly substituted for full-run MFU")
    return {
        "status": "verified",
        "source": verify_training_sources(root),
        "timing": verify_timing(accounting),
        "training_telemetry": recompute_telemetry(root, accounting),
        "final_metrics": verify_final_metrics(root, accounting),
        "efficiency_result": verify_efficiency_result(root),
        "model_release": verify_release_binding(root, manifest),
        "energy": {
            "status": accounting["energy"]["status"],
            "full_training_energy_kwh": None,
        },
        "claim_boundary": manifest["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = verify_bundle(args.root)
    except VerificationError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
