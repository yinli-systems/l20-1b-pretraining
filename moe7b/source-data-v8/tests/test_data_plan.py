import copy
import json
from pathlib import Path

from scripts.validate_moe7b_data_plan import DEFAULT_PLAN, validate
from scripts.make_moe7b_audit_acquisition_plan import select_records


def load_plan() -> dict:
    return json.loads(Path(DEFAULT_PLAN).read_text(encoding="utf-8"))


def test_frozen_150b_plan_passes() -> None:
    result = validate(load_plan())
    assert result["status"] == "PASS", result["errors"]
    assert result["target_exposed_tokens"] == 150_000_000_000
    assert result["minimum_unique_tokens"] == 142_500_000_000


def test_stage_drift_fails_closed() -> None:
    plan = copy.deepcopy(load_plan())
    plan["stages"][2]["source_quotas"]["finemath"] += 1
    result = validate(plan)
    assert result["status"] == "FAIL"
    assert any("stage C" in error for error in result["errors"])


def test_source_revision_must_be_immutable() -> None:
    plan = copy.deepcopy(load_plan())
    plan["sources"][0]["revision"] = "main"
    result = validate(plan)
    assert result["status"] == "FAIL"
    assert any("revision" in error for error in result["errors"])


def test_admission_status_cannot_be_predeclared() -> None:
    plan = copy.deepcopy(load_plan())
    plan["status"] = "ADMITTED"
    result = validate(plan)
    assert result["status"] == "FAIL"
    assert any("fail-closed" in error for error in result["errors"])


def test_stack_audit_selection_spans_frozen_inventory() -> None:
    records = [
        {
            "path": f"data/part-{index:05d}.parquet",
            "size": 100 + index,
            "lfs_sha256": f"{index:064x}",
        }
        for index in range(16)
    ]
    selected = select_records("stack_v3_permissive", records, stack_shards=4)
    assert [record["path"] for record in selected] == [
        "data/part-00000.parquet",
        "data/part-00005.parquet",
        "data/part-00010.parquet",
        "data/part-00015.parquet",
    ]
