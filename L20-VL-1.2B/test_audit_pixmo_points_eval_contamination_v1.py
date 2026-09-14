import json
from pathlib import Path
import tempfile
import unittest

from audit_pixmo_points_eval_contamination_v1 import audit_manifest, extract_image_hashes


class PixMoPointsEvalContaminationTests(unittest.TestCase):
    def test_nested_hash_extraction_ignores_non_image_hashes(self):
        a, b = "a" * 64, "b" * 64
        row = {
            "image_a": {"image_sha256": a, "selection_sha256": "c" * 64},
            "controls": [{"edited_image_sha256": b}],
            "caption_sha256": "d" * 64,
        }
        self.assertEqual(extract_image_hashes(row), {a, b})

    def test_manifest_overlap(self):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "manifest.jsonl"
        path.write_text("\n".join([
            json.dumps({"image_sha256": "a" * 64}),
            json.dumps({"base_image_sha256": "b" * 64, "selection_sha256": "a" * 64}),
        ]) + "\n")
        try:
            result = audit_manifest(path, {"a" * 64, "c" * 64})
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["unique_image_hashes"], 2)
            self.assertEqual(result["overlap_sha256"], ["a" * 64])
        finally:
            directory.cleanup()

    def test_invalid_image_hash_is_not_treated_as_evidence(self):
        self.assertEqual(extract_image_hashes({"image_sha256": "bad"}), set())


if __name__ == "__main__":
    unittest.main()
