from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from PIL import Image

from vision1b.openimages_cvdf_admission import combined_dhash_128, pixel_sha256


SCRIPT = Path(__file__).parents[1] / "vision1b" / "openimages_cvdf_admission.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jpeg(seed: int) -> bytes:
    image = Image.new("RGB", (256, 256))
    pixels = image.load()
    for y in range(256):
        for x in range(256):
            pixels[x, y] = (
                (x * (seed + 3) + y * 7) % 256,
                (y * (seed + 5) + x * 11) % 256,
                ((x ^ y) * (seed + 1)) % 256,
            )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def write_archive(path: Path, rows: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for image_id, payload in rows:
            member = tarfile.TarInfo(f"nested/{image_id}.jpg")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def run(*arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(argument) for argument in arguments)],
        check=check,
        text=True,
        capture_output=True,
    )


def test_shard_parallel_admission_is_identity_bound_and_fail_closed(tmp_path: Path) -> None:
    ids = [f"{index:016x}" for index in range(1, 5)]
    payload_a = jpeg(1)
    payload_b = jpeg(2)
    payload_c = jpeg(3)
    archive_0 = tmp_path / "train_0.tar.gz"
    archive_1 = tmp_path / "train_1.tar.gz"
    write_archive(archive_0, [(ids[0], payload_a), (ids[1], payload_a)])
    write_archive(archive_1, [(ids[2], payload_b), (ids[3], payload_c)])

    metadata = tmp_path / "metadata.csv"
    fields = [
        "ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
        "Author", "AuthorProfileURL", "Title",
    ]
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id in ids:
            writer.writerow(
                {
                    "ImageID": image_id,
                    "Subset": "train",
                    "OriginalURL": f"https://example.invalid/{image_id}.jpg",
                    "OriginalLandingURL": f"https://example.invalid/{image_id}",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                    "Author": "test-author",
                    "AuthorProfileURL": "https://example.invalid/author",
                    "Title": image_id,
                }
            )

    protocol = json.loads(
        (Path(__file__).parents[1] / "vision1b" / "openimages_cvdf_admission_protocol_v1.json").read_text()
    )
    protocol["acquisition"]["expected_archive_count"] = 2
    protocol["acquisition"]["protocol_sha256"] = "synthetic-acquisition"
    protocol["metadata"]["expected_bytes"] = metadata.stat().st_size
    protocol["metadata"]["sha256"] = digest(metadata)
    protocol["audit"]["images"] = 2
    protocol["storage"]["minimum_free_bytes"] = 0
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True))

    acquisition = {
        "archives": [
            {"bytes": path.stat().st_size, "filename": path.name, "sha256": digest(path), "url": "synthetic"}
            for path in (archive_0, archive_1)
        ],
        "protocol_sha256": "synthetic-acquisition",
        "status": "DOWNLOADED_NOT_ADMITTED",
    }
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition))

    database = tmp_path / "metadata.sqlite"
    metadata_receipt = tmp_path / "metadata-receipt.json"
    run(
        "build-metadata", "--protocol", protocol_path, "--metadata", metadata,
        "--output", database, "--receipt", metadata_receipt,
    )

    index_receipts = []
    for number, archive in enumerate((archive_0, archive_1)):
        manifest = tmp_path / f"index-{number}.jsonl"
        receipt = tmp_path / f"index-{number}-receipt.json"
        run(
            "index-shard", "--protocol", protocol_path,
            "--acquisition-receipt", acquisition_path, "--archive", archive,
            "--metadata-database", database, "--metadata-receipt", metadata_receipt,
            "--output-manifest", manifest, "--output-receipt", receipt,
        )
        index_receipts.append(receipt)

    with Image.open(io.BytesIO(payload_b)) as image:
        downstream = {
            "dhash128": combined_dhash_128(image),
            "file_sha256": hashlib.sha256(payload_b).hexdigest(),
            "image_id": "downstream:test:00000",
            "pixel_sha256": pixel_sha256(image),
            "source": "downstream:test:00000",
        }
    downstream_manifest = tmp_path / "downstream.jsonl"
    downstream_manifest.write_text(json.dumps(downstream, sort_keys=True) + "\n")
    downstream_receipt = tmp_path / "downstream-receipt.json"
    downstream_receipt.write_text(
        json.dumps(
            {
                "manifest_sha256": digest(downstream_manifest),
                "rows": 1,
                "status": "FROZEN_DOWNSTREAM_DENYLIST",
            }
        )
    )

    reduced = tmp_path / "reduced"
    run(
        "reduce", "--protocol", protocol_path, "--index-receipts", *index_receipts,
        "--downstream-manifest", downstream_manifest,
        "--downstream-receipt", downstream_receipt, "--output-root", reduced,
    )
    reduction = json.loads((reduced / "reduction-receipt.json").read_text())
    assert reduction["accepted_rows"] == 2
    assert reduction["rejected"] == {
        "downstream_exact_overlap": 1,
        "global_exact_duplicate": 1,
    }

    audit_root = tmp_path / "audit"
    run(
        "render-audit", "--protocol", protocol_path,
        "--reduction-receipt", reduced / "reduction-receipt.json",
        "--archive-root", tmp_path, "--output-root", audit_root,
    )
    audit_index = audit_root / "audit-index.json"
    audit = json.loads(audit_index.read_text())
    assert len(audit["image_ids"]) == 2

    pack_receipts = []
    for number, archive in enumerate((archive_0, archive_1)):
        output_tar = tmp_path / f"packed-{number}.tar"
        output_receipt = tmp_path / f"packed-{number}.json"
        run(
            "pack-shard", "--protocol", protocol_path,
            "--reduction-receipt", reduced / "reduction-receipt.json",
            "--archive", archive, "--output-tar", output_tar,
            "--output-receipt", output_receipt,
        )
        pack_receipts.append(output_receipt)

    audit_decision = tmp_path / "audit-decision.json"
    audit_decision.write_text(json.dumps({"status": "PASS", "audit_index_sha256": digest(audit_index)}))
    safety = tmp_path / "safety.json"
    safety.write_text(
        json.dumps(
            {
                "candidate_manifest_sha256": reduction["manifest"]["sha256"],
                "policy_sha256": "synthetic-policy",
                "scanned_rows": 2,
                "status": "FAIL",
            }
        )
    )
    final = tmp_path / "training-admission.json"
    failed = run(
        "finalize", "--protocol", protocol_path,
        "--reduction-receipt", reduced / "reduction-receipt.json",
        "--audit-index", audit_index, "--audit-decision", audit_decision,
        "--safety-receipt", safety, "--pack-receipts", *pack_receipts,
        "--output-receipt", final, check=False,
    )
    assert failed.returncode != 0
    assert not final.exists()

    safety.write_text(
        json.dumps(
            {
                "candidate_manifest_sha256": reduction["manifest"]["sha256"],
                "policy_sha256": "synthetic-policy",
                "scanned_rows": 2,
                "status": "PASS",
            }
        )
    )
    with (tmp_path / "packed-0.tar").open("ab") as handle:
        handle.write(b"drift")
    drifted = run(
        "finalize", "--protocol", protocol_path,
        "--reduction-receipt", reduced / "reduction-receipt.json",
        "--audit-index", audit_index, "--audit-decision", audit_decision,
        "--safety-receipt", safety, "--pack-receipts", *pack_receipts,
        "--output-receipt", final, check=False,
    )
    assert drifted.returncode != 0
    assert not final.exists()
    run(
        "pack-shard", "--protocol", protocol_path,
        "--reduction-receipt", reduced / "reduction-receipt.json",
        "--archive", archive_0, "--output-tar", tmp_path / "packed-0.tar",
        "--output-receipt", pack_receipts[0],
    )
    run(
        "finalize", "--protocol", protocol_path,
        "--reduction-receipt", reduced / "reduction-receipt.json",
        "--audit-index", audit_index, "--audit-decision", audit_decision,
        "--safety-receipt", safety, "--pack-receipts", *pack_receipts,
        "--output-receipt", final,
    )
    final_receipt = json.loads(final.read_text())
    assert final_receipt["status"] == "TRAINING_ADMITTED"
    assert final_receipt["training_admitted_count"] == 2
