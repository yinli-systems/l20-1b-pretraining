import json
from pathlib import Path

import pipeline_watchdog


def test_progress_signature_tracks_data_without_volatile_status_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline_watchdog, "PRODUCTION_STATUS", tmp_path / "manifests/production-status.json")
    (tmp_path / "manifests").mkdir()
    (tmp_path / "data/full-npy/code").mkdir(parents=True)
    (tmp_path / "manifests/production-status.json").write_text(
        json.dumps({"stage": "building_full_data", "updated_unix": 123})
    )
    (tmp_path / "data/full-npy/code/progress.json").write_text(
        json.dumps({"train_tokens": 42, "val_tokens": 7, "seen": 99})
    )

    signature = pipeline_watchdog.progress_signature()

    assert signature["production_stage"] == "building_full_data"
    assert signature["sources"]["code"] == {"train_tokens": 42, "val_tokens": 7}
    assert "updated_unix" not in signature


def test_atomic_status_is_durable_json(tmp_path, monkeypatch):
    status = tmp_path / "watchdog-status.json"
    monkeypatch.setattr(pipeline_watchdog, "STATUS", status)

    pipeline_watchdog.atomic_status("monitoring", restart_count=2)

    payload = json.loads(status.read_text())
    assert payload["stage"] == "monitoring"
    assert payload["restart_count"] == 2
    assert not status.with_suffix(".json.tmp").exists()
