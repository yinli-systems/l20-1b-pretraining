import gzip
import json
from pathlib import Path

import pytest

from audit_raw_intake_v2 import preflight, sha
import audit_incremental


def write_old(root, source_id, group):
    raw = root / f"{source_id}.jsonl.gz"
    receipt = {
        "id": source_id, "status": "RAW_SEGMENT_READY", "repo_id": "owned/repo", "revision": "v1",
        "path": "shard-a.parquet", "groups": [{"row_group": group, "physical_row_offset": group * 10, "rows_read": 1}],
        "rows_written": 1,
    }
    with gzip.open(raw, "wt") as handle:
        handle.write(json.dumps({"text": "old", "_provenance": {"source_id": source_id}}) + "\n")
    receipt.update(output_sha256=sha(raw), output_bytes=raw.stat().st_size)
    path = root / f"{source_id}.receipt.json"
    path.write_text(json.dumps(receipt))
    (root / "intake-summary.json").write_text(json.dumps({"status": "RAW_INTAKE_FINISHED_NOT_ADMITTED", "sources_blocked": 0, "sources_ready": 1, "rows": 1}))
    return path


def write_generalized(root, old_receipt, excluded=(0,), group=1):
    raw = root / "dclm_extra.jsonl.gz"
    old = json.loads(old_receipt.read_text())
    receipt = {
        "segment_id": "dclm_extra", "logical_source_id": "dclm", "status": "RAW_SEGMENT_READY",
        "repo_id": old["repo_id"], "revision": old["revision"], "path": old["path"],
        "history": [{"path": str(old_receipt), "sha256": sha(old_receipt)}],
        "exclude_row_groups": list(excluded),
        "groups": [{"row_group": group, "physical_row_offset": group * 10, "rows_read": 1}], "rows_written": 1,
    }
    with gzip.open(raw, "wt") as handle:
        handle.write(json.dumps({"text": "new", "_provenance": {"source_id": "dclm"}}) + "\n")
    receipt.update(output_sha256=sha(raw), output_bytes=raw.stat().st_size)
    (root / "dclm_extra.receipt.json").write_text(json.dumps(receipt))
    (root / "intake-summary.json").write_text(json.dumps({"status": "RAW_INTAKE_FINISHED_NOT_ADMITTED", "sources_blocked": 0, "sources_ready": 1, "rows": 1}))


def test_generalized_history_and_logical_source(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    receipt = write_old(old, "dclm", 0)
    write_generalized(new, receipt)
    files, _, roots = preflight([old, new])
    assert [x["source_id"] for x in files] == ["dclm", "dclm"]
    assert [x["file_id"] for x in files] == ["dclm", "dclm_extra"]
    assert [x["tranche"] for x in files] == [0, 1]
    assert sum(x["rows"] for x in roots) == 2


def test_generalized_history_union_is_fail_closed(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    receipt = write_old(old, "dclm", 0)
    write_generalized(new, receipt, excluded=())
    with pytest.raises(ValueError, match="generalized history group union mismatch"):
        preflight([old, new])


def test_incremental_history_maps_multiple_same_source_files_by_path(tmp_path):
    root=tmp_path/'root';root.mkdir()
    files=[];prior=[];bindings={}
    for tranche,name in enumerate(['a.jsonl.gz','b.jsonl.gz']):
        path=root/name;path.write_bytes(name.encode());digest=audit_incremental.sha(path)
        files.append({'source_id':'dclm','tranche':tranche,'root_index':0,'path':path,
                      'receipt':{'rows_written':tranche+1}})
        bindings[path]=digest
        prior.append({'source_id':'dclm','tranche':tranche,'path':str(path),
                      'rows':tranche+1,'raw_file_sha256':digest})
    mapped=audit_incremental.map_historical_files(files,prior,1,bindings)
    assert mapped[('dclm',0)]['path']==Path(root/'a.jsonl.gz')
    assert mapped[('dclm',1)]['path']==Path(root/'b.jsonl.gz')
