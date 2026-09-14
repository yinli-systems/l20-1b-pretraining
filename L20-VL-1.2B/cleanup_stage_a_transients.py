#!/usr/bin/env python3
"""Remove only receipted Stage-A transients and the prohibited CLEVR test split."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACQUISITION = ROOT / "evidence" / "stage-a-source-acquisition.json"
RECEIPT = ROOT / "evidence" / "stage-a-transient-cleanup.json"
SOURCE_ROOT = Path("/home/hhai/l20-vl-1.2b/data/stage-a-v1/sources")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict:
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink target: {path}")
    if path.is_file():
        return {"path": str(path), "kind": "file", "files": 1, "bytes": path.stat().st_size}
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "path": str(path),
        "kind": "directory",
        "files": len(files),
        "bytes": sum(candidate.stat().st_size for candidate in files),
    }


def main() -> None:
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    acquisition = json.loads(ACQUISITION.read_text())
    if acquisition.get("status") != "complete_pending_content_audit":
        raise SystemExit("source acquisition is not complete")
    archive = SOURCE_ROOT / "CLEVR_v1.0.zip"
    expected_archive = acquisition["artifacts"]["clevr_archive"]
    if archive.stat().st_size != expected_archive["bytes"] or sha256_file(archive) != expected_archive["sha256"]:
        raise RuntimeError("CLEVR archive does not match the acquisition receipt")
    clevr = SOURCE_ROOT / "CLEVR_v1.0"
    targets = [
        archive,
        clevr / "images" / "test",
        clevr / "questions" / "CLEVR_test_questions.json",
    ]
    if any(not target.exists() for target in targets):
        raise RuntimeError("one or more exact cleanup targets are absent")
    before = [inventory(target) for target in targets]
    archive.unlink()
    shutil.rmtree(targets[1])
    targets[2].unlink()
    if any(target.exists() for target in targets):
        raise RuntimeError("cleanup absence verification failed")
    train_count = len(list((clevr / "images" / "train").glob("*.png")))
    validation_count = len(list((clevr / "images" / "val").glob("*.png")))
    if train_count != 70_000 or validation_count != 15_000:
        raise RuntimeError("retained CLEVR split verification failed")
    result = {
        "schema_version": "2026-09-13-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_acquisition_sha256": sha256_file(ACQUISITION),
        "removed": before,
        "removed_bytes": sum(item["bytes"] for item in before),
        "absence_verified": True,
        "retained_train_images": train_count,
        "retained_validation_images": validation_count,
        "license_file_retained": (clevr / "LICENSE.txt").exists(),
        "reason": "The archive was hash-receipted and extracted; the test split is prohibited by the frozen admission and is not used for training or development.",
        "recoverability": "The exact official archive can be reacquired from the frozen URL and verified against the recorded SHA-256.",
    }
    RECEIPT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
