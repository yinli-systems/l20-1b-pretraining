import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_openimages_attribute_candidates as module


class OpenImagesAttributeAuditTests(unittest.TestCase):
    def test_threshold_summary_requires_cross_attribute_class_cell(self):
        base = {
            "split": "train", "family": "posture", "class_name": "Person",
            "box_width": 0.2, "box_height": 0.3, "box_area": 0.06,
            "family_conflict": False,
        }
        rows = [
            {**base, "image_id": "a", "attribute_name": "Stand"},
            {**base, "image_id": "b", "attribute_name": "Sit"},
            {**base, "image_id": "c", "attribute_name": "Stand"},
            {**base, "image_id": "d", "class_name": "Dog", "attribute_name": "Sit"},
        ]
        result = module.threshold_summary(rows, 0.1, 0.02)
        self.assertEqual(result["pairable_annotations"], 3)
        self.assertEqual(result["same_attribute_no_op_ready_annotations"], 2)
        self.assertEqual(result["cross_attribute_pair_count"], 2)
        self.assertEqual(result["pairable_class_family_details"], [{
            "split": "train",
            "family": "posture",
            "class_name": "Person",
            "attribute_counts": {"Sit": 1, "Stand": 2},
            "annotations": 3,
            "images": 3,
        }])

    def test_load_manifest_rejects_duplicate_image_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text('\n'.join([
                json.dumps({"image_id": "x", "split": "train"}),
                json.dumps({"image_id": "x", "split": "development"}),
            ]) + '\n')
            with patch.object(module, "sha256_file", return_value=module.EXPECTED_MANIFEST_SHA256):
                with self.assertRaisesRegex(RuntimeError, "duplicate"):
                    module.load_manifest(path)

    def test_transparent_is_not_in_contrastive_family(self):
        self.assertIsNone(module.family_for_attribute("Transparent"))


if __name__ == "__main__":
    unittest.main()
