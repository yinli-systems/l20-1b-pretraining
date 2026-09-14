from pathlib import Path

import pytest

from reproducibility.recompute import (
    VerificationError,
    require_close,
    verify_bound_files,
    verify_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_bundle_recomputes() -> None:
    result = verify_bundle(ROOT)
    assert result["status"] == "verified"
    assert result["training_telemetry"]["samples"] == 19144
    assert result["training_telemetry"]["time_weighted_mfu"] == pytest.approx(0.7105589798862282)
    assert result["final_metrics"]["final_validation_perplexity"] == pytest.approx(11.299003601)


def test_snapshot_is_not_full_run_average() -> None:
    result = verify_bundle(ROOT)
    assert result["training_telemetry"]["time_weighted_mfu"] != pytest.approx(0.7117272727272727)


def test_unmeasured_energy_remains_null() -> None:
    result = verify_bundle(ROOT)
    assert result["energy"] == {
        "status": "not_measured_for_full_training",
        "full_training_energy_kwh": None,
    }


def test_numeric_verification_fails_closed() -> None:
    with pytest.raises(VerificationError, match="tampered mismatch"):
        require_close("tampered", 1.0, 2.0)


def test_bound_file_tampering_fails_closed(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("changed\n", encoding="utf-8")
    manifest = {"bound_files": {"evidence.txt": "0" * 64}}
    with pytest.raises(VerificationError, match="SHA-256 mismatch"):
        verify_bound_files(tmp_path, manifest)
