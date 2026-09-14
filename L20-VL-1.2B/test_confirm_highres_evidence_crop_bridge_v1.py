#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from confirm_highres_evidence_crop_bridge_v1 import load_sealed_rows


class CropBridgeConfirmationTests(unittest.TestCase):
    def test_loader_returns_only_sorted_sealed_rows(self) -> None:
        rows = [
            {"split": "train", "family_id": "a", "variant": "base"},
            {"split": "sealed_test", "family_id": "z", "variant": "base"},
            {"split": "sealed_test", "family_id": "z", "variant": "answer_change"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            selected = load_sealed_rows(path)
        self.assertEqual([row["split"] for row in selected], ["sealed_test", "sealed_test"])
        self.assertEqual([row["variant"] for row in selected], ["answer_change", "base"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
