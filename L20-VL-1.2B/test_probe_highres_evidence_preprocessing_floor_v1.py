import unittest

import numpy as np

from probe_highres_evidence_preprocessing_floor_v1 import (
    digit_patch_indices,
    load_rows,
    metrics,
    paired_family_bootstrap,
    patch_index_at,
)


class HighresEvidencePreprocessingFloorTests(unittest.TestCase):
    def test_patch_index_row_major_and_bounded(self):
        self.assertEqual(patch_index_at(0, 0, 100, 100, 10), 0)
        self.assertEqual(patch_index_at(99, 99, 100, 100, 10), 99)
        with self.assertRaisesRegex(ValueError, "invalid"):
            patch_index_at(100, 99, 100, 100, 10)

    def test_digit_indices_differ_between_global_and_crop_coordinate_frames(self):
        row = {"target_crop_box_xyxy_normalized": [0.25, 0.25, 0.5, 0.5]}
        global_index, crop_index = digit_patch_indices(row, 1024, 14)
        self.assertNotEqual(global_index, crop_index)
        self.assertTrue(0 <= global_index < 196)
        self.assertTrue(0 <= crop_index < 196)

    def test_metrics_are_family_joint(self):
        rows = [
            {"family_id": "a", "answer": "1"},
            {"family_id": "a", "answer": "2"},
            {"family_id": "b", "answer": "3"},
            {"family_id": "b", "answer": "4"},
        ]
        result = metrics(rows, np.asarray(["1", "0", "3", "4"]))
        self.assertEqual(result["row_accuracy"], 0.75)
        self.assertEqual(result["family_joint_accuracy"], 0.5)

    def test_paired_bootstrap_uses_family_deltas(self):
        rows = [
            {"family_id": "a", "answer": "1"},
            {"family_id": "a", "answer": "2"},
            {"family_id": "b", "answer": "3"},
            {"family_id": "b", "answer": "4"},
        ]
        result = paired_family_bootstrap(
            rows,
            np.asarray(["1", "2", "3", "4"]),
            np.asarray(["0", "0", "3", "0"]),
            samples=1000,
            seed=1,
        )
        self.assertEqual(result["point_estimate"], 0.75)
        self.assertEqual(result["cluster_unit"], "scene_family")


if __name__ == "__main__":
    unittest.main()
