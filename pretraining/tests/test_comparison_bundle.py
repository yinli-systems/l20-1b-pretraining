"""Check the published compact evidence bundle without downloading models."""

import hashlib
import json
from pathlib import Path

import pytest

from evaluate_comparison import BASELINES, ORIGINALS, PROTOCOL, jobs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle():
    receipt_path = ROOT / "reports/receipts/tinyllama-comparison-20260911.json"
    summary = json.loads((ROOT / "reports/metrics/tinyllama-comparison-20260911.json").read_text())
    receipt = json.loads(receipt_path.read_text())
    return receipt_path, receipt, summary


def test_completed_bundle_and_hashes(bundle):
    receipt_path, receipt, summary = bundle
    assert receipt["status"] == summary["suite_status_at_analysis"] == "complete"
    assert summary["evaluation_receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert summary["analysis_sha256"] == hashlib.sha256((ROOT / "analyze_comparison.py").read_bytes()).hexdigest()
    assert receipt["runner_sha256"] == summary["runner_sha256"] == hashlib.sha256((ROOT / "evaluate_comparison.py").read_bytes()).hexdigest()
    assert receipt["original_result_sha256"] == summary["original_result_sha256"] == ORIGINALS
    old_receipt = json.loads((ROOT / "reports/receipts/final-evaluation-receipt.json").read_text())
    assert {name: row["sha256"] for name, row in old_receipt["suites"].items()} == ORIGINALS
    assert set(receipt["jobs"]) == {job["name"] for job in jobs("all")}
    for name, job in receipt["jobs"].items():
        assert job["status"] == "complete"
        assert summary["comparison_result_sha256"][name] == job["sha256"]


def test_protocol_and_model_revisions(bundle):
    _, receipt, summary = bundle
    assert receipt["protocol"] == summary["protocol"] == PROTOCOL
    assert {name: tuple(value) for name, value in summary["baseline_revisions"].items()} == BASELINES
    for row in receipt["jobs"].values():
        config = row["actual_config"]
        assert config["random_seed"] == config["numpy_seed"] == config["torch_seed"] == 42
        assert config["fewshot_seed"] == 1234
        assert config["model_dtype"] == "torch.bfloat16"
        assert config["model_num_parameters"] == 1100048384
        assert config["limit"] is None
        if row["job"]["revision"]:
            assert config["model_revision"] == row["job"]["revision"]


def test_full_comparisons_and_macro_arithmetic(bundle):
    _, _, summary = bundle
    assert len(summary["ours_primary_scores"]) == 7
    for comparison in summary["comparisons"].values():
        assert len(comparison["primary"]) == 7
        assert len(comparison["extended"]) == 5
        for side in ("ours", "baseline"):
            observed = sum(row[side] for row in comparison["primary"].values()) / 7
            assert comparison["primary_macro"][side] == pytest.approx(observed, abs=1e-12)
        assert len(comparison["extended"]["mmlu_5shot"]["subjects"]) == 57
        assert comparison["extended"]["mmlu_5shot"]["n"] == 14042
        for row in comparison["primary"].values():
            assert row["protocol_alignment"] == "PASS"
            assert row["ours_better_samples"] + row["baseline_better_samples"] + row["tied_samples"] == row["n"]


def test_report_table_matches_measured_values(bundle):
    _, _, summary = bundle
    report = (ROOT / "reports/research/tinyllama-comparison-20260911.md").read_text()
    rows = {parts[1].strip(): [cell.strip() for cell in parts[2:-1]]
            for line in report.splitlines() if line.startswith("|")
            for parts in [line.split("|")]}
    names = {"HellaSwag acc_norm": "hellaswag", "OpenBookQA acc_norm": "openbookqa",
             "Winogrande acc": "winogrande", "ARC-Challenge acc_norm": "arc_challenge",
             "ARC-Easy acc_norm": "arc_easy", "BoolQ acc": "boolq", "PIQA acc_norm": "piqa"}
    for label, task in names.items():
        values = [summary["ours_primary_scores"][task]] + [
            summary["comparisons"][model]["primary"][task]["baseline"] for model in BASELINES]
        assert rows[label][1:] == [f"{value*100:.2f}" for value in values]
    for label, task in {"LAMBADA acc": "lambada_openai", "TruthfulQA MC2": "truthfulqa_mc2",
                        "MMLU 5-shot acc": "mmlu_5shot",
                        "GSM8K 5-shot flexible extraction": "gsm8k_flexible-extract",
                        "GSM8K 5-shot strict match": "gsm8k_strict-match"}.items():
        values = [summary["comparisons"]["tinyllama_3t"]["extended"][task]["ours"]] + [
            summary["comparisons"][model]["extended"][task]["baseline"] for model in BASELINES]
        assert rows[label][1:] == [f"{value*100:.2f}" for value in values]
    values = [summary["ours_primary_macro"]] + [summary["comparisons"][model]["primary_macro"]["baseline"] for model in BASELINES]
    assert rows["七项等权均分"][1:] == [f"{value*100:.2f}" for value in values]
