import tempfile
from pathlib import Path
import unittest

from build_openimages_posture_confirmation_candidates_v1 import (
    assign_one_target_per_image,
    geometry_passes,
    reject_development_overlap,
    select_for_download,
)


FILTERS = {
    "Sit": {"min_width": 0.25, "min_height": 0.25, "min_area": 0.06},
    "Stand": {
        "min_width": 0.15,
        "max_width": 0.65,
        "min_height": 0.35,
        "min_area": 0.04,
        "min_bbox_height_to_width": 1.0,
    },
}


class OpenImagesConfirmationBuilderTests(unittest.TestCase):
    def test_posture_specific_geometry(self):
        self.assertTrue(geometry_passes("Sit", [0.0, 0.3, 0.0, 0.3], FILTERS))
        self.assertFalse(geometry_passes("Sit", [0.0, 0.2, 0.0, 0.8], FILTERS))
        self.assertTrue(geometry_passes("Stand", [0.0, 0.2, 0.0, 0.8], FILTERS))
        self.assertFalse(geometry_passes("Stand", [0.0, 0.7, 0.0, 0.8], FILTERS))

    def test_one_target_per_image_is_deterministic(self):
        rows = [
            {"image_id": "a", "class_name": "Boy", "attribute_name": "Sit", "bbox_key": ("0",)},
            {"image_id": "a", "class_name": "Man", "attribute_name": "Stand", "bbox_key": ("1",)},
            {"image_id": "b", "class_name": "Girl", "attribute_name": "Sit", "bbox_key": ("2",)},
        ]
        left, conflicts = assign_one_target_per_image(rows, 7)
        right, _ = assign_one_target_per_image(list(reversed(rows)), 7)
        self.assertEqual(left, right)
        self.assertEqual(conflicts, 1)
        self.assertEqual(len(left), 2)

    def test_selection_is_balanced_and_image_unique(self):
        rows = []
        for class_name in ("Boy", "Girl", "Man", "Woman"):
            for attribute in ("Sit", "Stand"):
                for index in range(3):
                    rows.append(
                        {
                            "image_id": f"{class_name}-{attribute}-{index}",
                            "class_name": class_name,
                            "attribute_name": attribute,
                            "bbox_key": (str(index),),
                        }
                    )
        protocol = {
            "seed": 11,
            "sampling": {"download_candidates_per_stratum": {"Sit": 2, "Stand": 2}},
        }
        selected = select_for_download(protocol, rows)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({row["image_id"] for row in selected}), 16)

    def test_development_exact_and_near_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "development.jsonl"
            manifest.write_text(
                '{"image_a":{"image_id":"old-a","image_sha256":"sha-a","dhash64":"0000000000000000"},'
                '"image_b":{"image_id":"old-b","image_sha256":"sha-b","dhash64":"ffffffffffffffff"}}\n'
            )
            rows = [
                {"image_id": "new-a", "image_sha256": "sha-a", "dhash64": "1234567890abcdef"},
                {"image_id": "new-b", "image_sha256": "new", "dhash64": "0000000000000001"},
                {"image_id": "new-c", "image_sha256": "fresh", "dhash64": "5555555555555555"},
            ]
            accepted, rejected = reject_development_overlap(rows, manifest, 4)
            self.assertEqual([row["image_id"] for row in accepted], ["new-c"])
            self.assertEqual({row["reason"] for row in rejected}, {"exact_sha256", "development_dhash_hamming_1"})


if __name__ == "__main__":
    unittest.main()
