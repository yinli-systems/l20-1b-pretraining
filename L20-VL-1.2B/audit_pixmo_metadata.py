#!/usr/bin/env python3
"""Stream the pinned PixMo-Cap metadata and produce a corpus-level admission audit."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
ADMISSION = ROOT / "pixmo_cap_metadata_admission.json"
ACQUISITION = ROOT / "evidence" / "pixmo-cap-metadata-acquisition.json"
OUTPUT = ROOT / "evidence" / "pixmo-cap-metadata-audit.json"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*)?(?:\d[\s().-]*){7,14}(?!\d)")
BENCHMARK_MARKERS = {
    "ai2d",
    "chartqa",
    "coco",
    "docvqa",
    "gqa",
    "infographicvqa",
    "infovqa",
    "mathvista",
    "mmmu",
    "okvqa",
    "scienceqa",
    "textvqa",
    "vizwiz",
    "vqa",
}
REFUSAL_MARKERS = (
    "i cannot see",
    "i can't see",
    "unable to determine",
    "cannot determine",
    "not possible to tell",
)
FILLERS = {"uh", "um", "erm", "hmm"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[int | float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def distribution(values: list[int | float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "p01": percentile(values, 0.01),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def normalized_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def ascii_fraction(text: str) -> float:
    return sum(ord(char) < 128 for char in text) / max(len(text), 1)


def sha_text(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def benchmark_markers(url: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", url.lower())
    tokens = set(normalized.split())
    return sorted(marker for marker in BENCHMARK_MARKERS if marker in tokens)


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    acquisition = json.loads(ACQUISITION.read_text())
    if acquisition.get("status") != "complete":
        raise SystemExit("metadata acquisition receipt is not complete")
    if admission.get("image_download_authorized") is not False:
        raise SystemExit("image download must remain blocked")
    if admission.get("training_authorized") is not False:
        raise SystemExit("training must remain blocked")

    root = Path(admission["destination"])
    rows = 0
    null_counts = Counter()
    schemes = Counter()
    hosts = Counter()
    benchmark_counts = Counter()
    caption_word_counts: list[int] = []
    caption_char_counts: list[int] = []
    transcript_word_counts: list[int] = []
    transcript_counts: list[int] = []
    caption_ascii_fractions: list[float] = []
    transcript_ascii_fractions: list[float] = []
    transcript_filler_fractions: list[float] = []
    caption_transcript_jaccards: list[float] = []
    duplicate_urls = 0
    duplicate_captions = 0
    seen_urls: set[bytes] = set()
    seen_captions: set[bytes] = set()
    email_candidates = 0
    phone_candidates = 0
    refusal_candidates = 0
    short_caption_rows = 0
    long_caption_rows = 0
    empty_transcript_rows = 0
    candidate_rows = 0

    for relative, expected in admission["files"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"missing or wrong-size shard: {relative}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow.names != ["image_url", "caption", "transcripts"]:
            raise SystemExit(f"unexpected schema: {relative}: {parquet.schema_arrow}")
        for batch in parquet.iter_batches(batch_size=4096):
            columns = batch.to_pydict()
            for url, caption, transcripts in zip(
                columns["image_url"], columns["caption"], columns["transcripts"], strict=True
            ):
                rows += 1
                if not isinstance(url, str) or not url.strip():
                    null_counts["image_url"] += 1
                    continue
                if not isinstance(caption, str) or not caption.strip():
                    null_counts["caption"] += 1
                    continue
                if not isinstance(transcripts, list):
                    null_counts["transcripts"] += 1
                    continue

                parsed = urlparse(url)
                schemes[parsed.scheme.lower()] += 1
                host = (parsed.hostname or "").lower()
                hosts[host] += 1
                markers = benchmark_markers(url)
                for marker in markers:
                    benchmark_counts[marker] += 1

                url_hash = sha_text(url.strip())
                caption_hash = sha_text(" ".join(caption.split()).lower())
                if url_hash in seen_urls:
                    duplicate_urls += 1
                else:
                    seen_urls.add(url_hash)
                if caption_hash in seen_captions:
                    duplicate_captions += 1
                else:
                    seen_captions.add(caption_hash)

                caption_words = normalized_words(caption)
                transcript_text = " ".join(item for item in transcripts if isinstance(item, str))
                transcript_words = normalized_words(transcript_text)
                caption_word_counts.append(len(caption_words))
                caption_char_counts.append(len(caption))
                transcript_word_counts.append(len(transcript_words))
                transcript_counts.append(len(transcripts))
                caption_ascii = ascii_fraction(caption)
                transcript_ascii = ascii_fraction(transcript_text)
                caption_ascii_fractions.append(caption_ascii)
                transcript_ascii_fractions.append(transcript_ascii)
                transcript_filler_fractions.append(
                    sum(word in FILLERS for word in transcript_words) / max(len(transcript_words), 1)
                )
                caption_set = set(caption_words)
                transcript_set = set(transcript_words)
                union = caption_set | transcript_set
                caption_transcript_jaccards.append(
                    len(caption_set & transcript_set) / max(len(union), 1)
                )

                combined = f"{caption}\n{transcript_text}"
                has_email = bool(EMAIL_RE.search(combined))
                has_phone = bool(PHONE_RE.search(combined))
                has_refusal = any(marker in combined.lower() for marker in REFUSAL_MARKERS)
                email_candidates += has_email
                phone_candidates += has_phone
                refusal_candidates += has_refusal
                short_caption_rows += len(caption_words) < 40
                long_caption_rows += len(caption_words) > 500
                empty_transcript_rows += not transcript_words

                if (
                    parsed.scheme.lower() == "https"
                    and host
                    and not markers
                    and 60 <= len(caption_words) <= 350
                    and transcript_words
                    and caption_ascii >= 0.985
                    and not has_email
                    and not has_phone
                    and not has_refusal
                ):
                    candidate_rows += 1

    expected_rows = admission["declared_examples"]
    integrity_status = "pass" if rows == expected_rows and not null_counts else "fail"
    report = {
        "schema_version": "2026-09-13-v1",
        "completed_at": utc_now(),
        "repo": admission["repo"],
        "revision": admission["revision"],
        "rows": rows,
        "expected_rows": expected_rows,
        "metadata_integrity_status": integrity_status,
        "null_or_invalid_rows": dict(null_counts),
        "url_schemes": dict(schemes.most_common()),
        "top_100_hosts": dict(hosts.most_common(100)),
        "unique_hosts": len(hosts),
        "unique_urls": len(seen_urls),
        "duplicate_url_rows": duplicate_urls,
        "unique_captions": len(seen_captions),
        "duplicate_caption_rows": duplicate_captions,
        "caption_words": distribution(caption_word_counts),
        "caption_characters": distribution(caption_char_counts),
        "transcript_words": distribution(transcript_word_counts),
        "transcripts_per_row": distribution(transcript_counts),
        "caption_ascii_fraction": distribution(caption_ascii_fractions),
        "transcript_ascii_fraction": distribution(transcript_ascii_fractions),
        "transcript_filler_fraction": distribution(transcript_filler_fractions),
        "caption_transcript_token_set_jaccard": distribution(caption_transcript_jaccards),
        "benchmark_marker_counts_from_url_only": dict(benchmark_counts.most_common()),
        "email_regex_candidate_rows": email_candidates,
        "phone_regex_candidate_rows": phone_candidates,
        "refusal_marker_rows": refusal_candidates,
        "caption_rows_under_40_words": short_caption_rows,
        "caption_rows_over_500_words": long_caption_rows,
        "empty_transcript_rows": empty_transcript_rows,
        "conservative_metadata_candidate_rows": candidate_rows,
        "candidate_definition": {
            "https_only": True,
            "benchmark_url_markers_excluded": sorted(BENCHMARK_MARKERS),
            "caption_word_range": [60, 350],
            "caption_ascii_fraction_min": 0.985,
            "requires_nonempty_human_transcript": True,
            "regex_email_phone_and_refusal_candidates_excluded": True,
        },
        "image_downloaded": False,
        "training_started": False,
        "training_admission": "blocked_pending_source_family_rights_opt_out_image_integrity_contamination_and_human_review",
        "claim_boundary": (
            "This report audits frozen metadata, not image availability, image-caption factuality, "
            "underlying image rights, row-level opt-out status, or benchmark pixel overlap."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if integrity_status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
