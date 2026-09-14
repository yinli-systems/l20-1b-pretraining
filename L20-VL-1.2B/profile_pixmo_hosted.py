#!/usr/bin/env python3
"""Profile the AI2-hosted PixMo-Cap subset without emitting raw corpus rows."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq

from audit_pixmo_metadata import (
    BENCHMARK_MARKERS,
    EMAIL_RE,
    PHONE_RE,
    REFUSAL_MARKERS,
    ascii_fraction,
    benchmark_markers,
    normalized_words,
)


ROOT = Path(__file__).resolve().parent
ADMISSION = ROOT / "pixmo_cap_metadata_admission.json"
AUDIT = ROOT / "evidence" / "pixmo-cap-metadata-audit.json"
OUTPUT = ROOT / "evidence" / "pixmo-cap-hosted-profile.json"
HOST = "pixmo.s3.us-west-2.amazonaws.com"


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    audit = json.loads(AUDIT.read_text())
    if audit.get("metadata_integrity_status") != "pass":
        raise SystemExit("full metadata audit has not passed")
    if admission.get("image_download_authorized") is not False:
        raise SystemExit("image acquisition is not authorized by this profiler")

    families = Counter()
    rows = 0
    candidate_rows = 0
    benchmark_rows = 0
    rejected = Counter()
    root = Path(admission["destination"])
    for relative in admission["files"]:
        parquet = pq.ParquetFile(root / relative)
        for batch in parquet.iter_batches(
            batch_size=8192, columns=["image_url", "caption", "transcripts"]
        ):
            columns = batch.to_pydict()
            for url, caption, transcripts in zip(
                columns["image_url"], columns["caption"], columns["transcripts"], strict=True
            ):
                parsed = urlparse(url)
                if (parsed.hostname or "").lower() != HOST:
                    continue
                rows += 1
                path_parts = [part for part in parsed.path.split("/") if part]
                families[path_parts[0].lower() if path_parts else "<root>"] += 1
                words = normalized_words(caption)
                transcript_words = normalized_words(
                    " ".join(item for item in transcripts if isinstance(item, str))
                )
                combined = caption + "\n" + " ".join(transcripts)
                markers = benchmark_markers(url)
                benchmark_rows += bool(markers)
                reasons = []
                if parsed.scheme.lower() != "https":
                    reasons.append("non_https")
                if markers:
                    reasons.append("benchmark_url_marker")
                if not 60 <= len(words) <= 350:
                    reasons.append("caption_length")
                if not transcript_words:
                    reasons.append("empty_transcript")
                if ascii_fraction(caption) < 0.985:
                    reasons.append("caption_non_ascii")
                if EMAIL_RE.search(combined):
                    reasons.append("email_regex")
                if PHONE_RE.search(combined):
                    reasons.append("phone_regex")
                if any(marker in combined.lower() for marker in REFUSAL_MARKERS):
                    reasons.append("refusal_marker")
                if reasons:
                    rejected.update(set(reasons))
                else:
                    candidate_rows += 1

    report = {
        "schema_version": "2026-09-13-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "repo": admission["repo"],
        "revision": admission["revision"],
        "host": HOST,
        "rows": rows,
        "path_family_counts": dict(families.most_common()),
        "unique_path_families": len(families),
        "benchmark_marker_rows": benchmark_rows,
        "conservative_candidate_rows": candidate_rows,
        "rejection_signal_counts_not_mutually_exclusive": dict(rejected.most_common()),
        "image_downloaded": False,
        "training_started": False,
        "training_admission": "blocked_pending_bounded_image_audit_and_family_rights_review",
        "claim_boundary": (
            "AI2-controlled hosting improves reproducibility but does not by itself establish the "
            "underlying image license, image-caption correctness, or evaluation non-overlap."
        ),
        "benchmark_markers": sorted(BENCHMARK_MARKERS),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
