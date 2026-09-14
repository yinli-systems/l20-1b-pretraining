import gzip
import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import intake


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    table = pa.table({"text": [f"owned document {i}" for i in range(12)], "url": [f"https://fixture/{i}" for i in range(12)]})
    sink = io.BytesIO()
    pq.write_table(table, sink, row_group_size=2)
    body = sink.getvalue()

    class MemoryHTTP(io.BytesIO):
        def __init__(self, _url):
            super().__init__(body)
            self.total, self.transferred, self.ranges = len(body), 0, []

        def read(self, size=-1):
            data = super().read(size)
            self.transferred += len(data)
            return data

    root = tmp_path
    output = root / "data/intake-expansion-v1"
    output.mkdir(parents=True)
    prior_dir = root / "data/diverse-intake-v1"
    prior_dir.mkdir()
    prior_output = prior_dir / "dclm.jsonl.gz"
    with gzip.open(prior_output, "wt") as handle:
        handle.write(json.dumps({"text": "prior"}) + "\n")
    prior = {
        "id": "dclm", "repo_id": "owned/fixture", "revision": "v1", "path": "file.parquet",
        "status": "RAW_SEGMENT_READY", "groups": [{"row_group": 1}], "output_sha256": hashlib.sha256(prior_output.read_bytes()).hexdigest(),
    }
    prior_path = prior_dir / "dclm.receipt.json"
    prior_path.write_text(json.dumps(prior))
    monkeypatch.setattr(intake, "ROOT", root)
    monkeypatch.setattr(intake, "OUTPUT", output)
    monkeypatch.setattr(intake, "MIN_FREE", 0)
    monkeypatch.setattr(intake.http_ranges, "BoundedHTTPFile", MemoryHTTP)
    segment = {
        "segment_id": "dclm_extra", "logical_source_id": "dclm", "repo_id": "owned/fixture", "revision": "v1",
        "path": "file.parquet", "bytes": len(body), "lfs_sha256": "0" * 64, "declared_dataset_license": "test",
        "history": [{"path": str(prior_path), "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest()}],
        "exclude_row_groups": [1], "additional_row_groups": 3, "parquet_row_groups": 6,
        "max_rows": 6, "max_text_bytes": 1024**2, "max_parquet_network_bytes": 1024**2,
        "max_code_attempts": 0, "max_code_payload_bytes": 0, "training_admitted": False,
    }
    return segment, prior_path, output


def test_history_is_excluded_and_provenance_is_bound(fixture):
    segment, _, output = fixture
    result = intake.acquire(segment)
    assert result["status"] == "RAW_SEGMENT_READY"
    assert len(result["groups"]) == 3
    assert 1 not in {x["row_group"] for x in result["groups"]}
    with gzip.open(output / "dclm_extra.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 6
    assert all(row["_provenance"]["source_id"] == "dclm" for row in rows)
    assert all(row["_provenance"]["segment_id"] == "dclm_extra" for row in rows)


def test_tampered_history_fails_closed(fixture):
    segment, prior, output = fixture
    prior.write_text(prior.read_text() + " ")
    result = intake.acquire(segment)
    assert result["status"] == "BLOCKED"
    assert "history receipt changed" in result["error"]
    assert not (output / "dclm_extra.jsonl.gz").exists()


def test_completed_segment_is_idempotent(fixture):
    segment, _, _ = fixture
    first = intake.acquire(segment)
    second = intake.acquire(segment)
    assert first["output_sha256"] == second["output_sha256"]
    assert first["completed_utc"] == second["completed_utc"]


def test_plan_covers_both_confirmation_candidates():
    plan = intake.SOURCE / "plan.json"
    data = json.loads(plan.read_text())
    assert data["source_repeat_cap"] == 2.0
    assert data["training_admitted"] is False
    assert len(data["segments"]) == 16
    assert sum(x["additional_row_groups"] for x in data["segments"] if x["logical_source_id"] == "finemath4") == 138
    assert {x["source_id"] for x in data["sources"]} == {
        "code_cpp", "code_java", "code_javascript", "code_python", "code_typescript", "dclm", "finemath4",
        "infiwebmath4", "pdf_en", "multilingual_arb_Arab", "multilingual_cmn_Hani", "multilingual_deu_Latn",
        "multilingual_fra_Latn", "multilingual_jpn_Jpan", "multilingual_spa_Latn",
    }
