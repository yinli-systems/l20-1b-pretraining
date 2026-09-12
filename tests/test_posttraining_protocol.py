import copy
import hashlib
import json
from pathlib import Path

from validate_posttraining_protocol import EXPECTED_STAGE_IDS, validate


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT / "posttraining_protocol.json").read_text())


def test_posttraining_protocol_is_fail_closed():
    result = validate(protocol())
    assert result == {
        "valid": True,
        "protocol_version": "2026-09-12-v1",
        "stage_count": len(EXPECTED_STAGE_IDS),
        "errors": [],
    }


def test_protocol_rejects_benchmark_training_and_auto_promotion():
    value = copy.deepcopy(protocol())
    value["automatic_promotion"] = True
    value["global_data_rules"]["benchmark_seeded_training_data"] = "allowed"
    result = validate(value)
    assert not result["valid"]
    assert "automatic_promotion must be false" in result["errors"]
    assert "benchmark_seeded_training_data must be forbidden" in result["errors"]


def test_protocol_rejects_unbounded_single_gpu_rlvr():
    value = copy.deepcopy(protocol())
    value["stages"][4]["single_l20_episode_ladder"].append(2_000_000)
    result = validate(value)
    assert not result["valid"]
    assert any("capped at 100k" in error for error in result["errors"])


def test_protocol_rejects_unpinned_or_unlicensed_sources():
    value = copy.deepcopy(protocol())
    value["candidate_source_registry"][0]["revision"] = "main"
    value["candidate_source_registry"][1]["license"] = ""
    result = validate(value)
    assert not result["valid"]
    assert any("not a pinned commit" in error for error in result["errors"])
    assert any("lacks license" in error for error in result["errors"])


def test_validation_receipt_binds_protocol_artifacts():
    receipt = json.loads(
        (ROOT / "reports/receipts/posttraining-protocol-validation-20260912.json").read_text()
    )
    for relative_path, expected in receipt["files"].items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected
    assert receipt["local_validation"]["protocol_valid"] is True
    assert receipt["github_actions"]["classification"] == "infrastructure_not_executed"
    assert receipt["github_actions"]["steps_started"] == 0
