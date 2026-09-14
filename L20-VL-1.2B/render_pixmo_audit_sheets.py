#!/usr/bin/env python3
"""Re-render readable PixMo audit sheets from the frozen remote-only manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from acquire_pixmo_image_audit import contact_sheet


ROOT = Path(__file__).resolve().parent
ADMISSION = ROOT / "pixmo_image_audit_admission.json"
RECEIPT = ROOT / "evidence" / "pixmo-cap-image-audit-acquisition.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("status") != "complete" or receipt.get("downloaded_images") != 288:
        raise SystemExit("complete 288-image receipt required")
    manifest_path = Path(receipt["manifest_path"])
    if sha256_file(manifest_path) != receipt["manifest_sha256"]:
        raise SystemExit("manifest hash mismatch")
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    results = {item["row_id"]: item for item in receipt["results"]}
    sheets = []
    for family in admission["path_families"]:
        family_records = [row for row in records if row["family"] == family]
        output = ROOT / "evidence" / "human-audit" / f"pixmo-{family}-sample-sheet.png"
        contact_sheet(family, family_records, results, output)
        sheets.append({"family": family, "path": str(output), "sha256": sha256_file(output)})
    receipt["contact_sheets"] = sheets
    receipt["sheet_layout"] = "4 columns; row id and three wrapped caption-prefix lines"
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "contact_sheets": sheets}, indent=2))


if __name__ == "__main__":
    main()
