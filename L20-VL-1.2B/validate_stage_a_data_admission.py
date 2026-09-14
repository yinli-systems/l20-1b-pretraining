#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
ADMISSION = ROOT / "stage_a_data_admission.json"


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    limits = admission["limits"]
    sources = admission["sources"]
    clevr = sources["clevr_v1_0"]
    narratives = sources["open_images_localized_narratives"]
    declared = (
        clevr["expected_bytes"]
        + narratives["captions_expected_bytes"]
        + narratives["image_metadata_expected_bytes"]
        + narratives["maximum_image_bytes"]
    )
    assert admission["authorized_by_user"] is True
    assert admission["download_authorized"] is True
    assert admission["audit_authorized"] is True
    assert admission["formal_release_authorized"] is False
    assert admission["automatic_training_start"] is False
    assert declared <= limits["download_bytes_max"]
    assert limits["stage_a_gpu_hours_max"] <= 12.0
    assert limits["destination"] == "/home/hhai/l20-vl-1.2b/data/stage-a-v1"
    assert clevr["license"] == "CC-BY-4.0"
    assert clevr["allowed_splits"] == ["train", "val"]
    assert clevr["prohibited_split"] == "test"
    assert narratives["caption_license"] == "CC-BY-4.0"
    assert narratives["required_image_license"] == "https://creativecommons.org/licenses/by/2.0/"
    allowed_hosts = {
        "cs.stanford.edu",
        "dl.fbaipublicfiles.com",
        "google.github.io",
        "storage.googleapis.com",
        "open-images-dataset.s3.amazonaws.com",
    }
    urls = [
        clevr["official_page"],
        clevr["artifact_url"],
        narratives["official_page"],
        narratives["captions_url"],
        narratives["image_metadata_url"],
        narratives["image_url_template"].format(image_id="0" * 16),
    ]
    assert all(urlparse(url).scheme == "https" for url in urls)
    assert all(urlparse(url).hostname in allowed_hosts for url in urls)
    required_gates = admission["quality_gates"]
    assert required_gates and all(required_gates.values())
    print(
        f"PASS: bounded Stage-A acquisition declared={declared} bytes <= "
        f"{limits['download_bytes_max']}; training remains conditional on every audit gate"
    )


if __name__ == "__main__":
    main()
