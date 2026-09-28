from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_loss_bundle_is_complete_and_hash_bound() -> None:
    training = _rows("reports/metrics/training-loss-100-step.csv")
    validation = _rows("reports/metrics/validation-loss.csv")
    receipt = json.loads((ROOT / "reports/receipts/loss-curve-final-en.json").read_text())

    assert len(training) == 192
    assert training[0]["step_start"] == "1"
    assert training[-1]["step_end"] == "19148"
    assert len(validation) == 39
    assert validation[-1]["step"] == "19148"
    assert validation[-1]["tokens_b"] == "19.999703"
    assert validation[-1]["loss"] == "2.424714565"
    assert validation[-1]["ppl"] == "11.299003601"

    assert receipt["language"] == "en"
    assert receipt["measurement_boundary"] == "all plotted curves are measured; no extrapolation"
    assert receipt["source_event"]["scalar_counts"]["loss"] == 19148
    assert receipt["source_event"]["scalar_counts"]["val_loss"] == 39
    for item in receipt["files"].values():
        path = ROOT / item["path"]
        assert path.is_file()
        assert _sha256(path) == item["sha256"]


def test_loss_png_has_publication_dimensions() -> None:
    png = ROOT / "reports/figures/loss-curve-final-en.png"
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (3200, 1500)
