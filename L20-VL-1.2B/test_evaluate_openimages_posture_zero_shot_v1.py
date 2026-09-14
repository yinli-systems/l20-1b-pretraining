import tempfile
import unittest
from pathlib import Path

from PIL import Image

import evaluate_openimages_posture_zero_shot_v1 as module


def image_record(path: Path, image_id: str, answer: str) -> dict:
    return {
        "image_id": image_id,
        "image_path": str(path),
        "image_sha256": "unused",
        "answer": answer,
        "bbox": [0.25, 0.75, 0.2, 0.8],
    }


class PostureZeroShotTests(unittest.TestCase):
    def test_flatten_pairs_preserves_atomic_donors(self):
        pair = {
            "pair_id": "p",
            "class_name": "Boy",
            "crossfit": {"fold": 2},
            "image_a": {"answer": "sitting"},
            "image_b": {"answer": "standing"},
        }
        rows = module.flatten_pairs([pair])
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["expected"] for row in rows}, {"sitting", "standing"})
        self.assertTrue(all(row["donor"]["answer"] != row["expected"] for row in rows))

    def test_bbox_grid_target_and_region(self):
        center, region = module.bbox_grid_targets([0.25, 0.75, 0.2, 0.8], grid=7)
        self.assertEqual(center, 24)
        self.assertIn(center, region)
        self.assertGreater(len(region), 1)

    def test_render_views_do_not_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (100, 80), (10, 20, 30)).save(path)
            before = path.read_bytes()
            record = image_record(path, "a", "sitting")
            full = module.render_view(record, "full", 0.1)
            marked = module.render_view(record, "marked", 0.1)
            crop = module.render_view(record, "crop", 0.1)
            self.assertEqual(full.size, (100, 80))
            self.assertEqual(marked.size, (100, 80))
            self.assertEqual(crop.width, crop.height)
            self.assertNotEqual(marked.getpixel((25, 16)), full.getpixel((25, 16)))
            self.assertEqual(path.read_bytes(), before)

    def test_pair_joint_is_strict(self):
        values, clusters = module.pair_joint(
            [
                {"pair_id": "a", "correct": True},
                {"pair_id": "a", "correct": False},
                {"pair_id": "b", "correct": True},
                {"pair_id": "b", "correct": True},
            ]
        )
        self.assertEqual(values, [0.0, 100.0])
        self.assertEqual(clusters, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
