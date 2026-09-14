#!/usr/bin/env python3
from __future__ import annotations

import unittest

from probe_binding_representations import (
    build_samples,
    diagnose_local_information,
    grid_index,
)


class BindingRepresentationProbeTests(unittest.TestCase):
    def test_build_samples_uses_one_family_and_swapped_target_colors(self) -> None:
        base = {
            "split": "train",
            "scene_pair_id": "pair-a",
            "question_index": 0,
            "challenge": "shape_color_swap",
            "scene_state": {
                "objects": [
                    {"id": "target", "color": "red", "x": 32, "y": 64},
                    {"id": "partner", "color": "blue", "x": 192, "y": 64},
                ]
            },
            "base_image_path": "/tmp/base.png",
            "base_image_sha256": "base",
            "edited_image_path": "/tmp/edited.png",
            "edited_image_sha256": "edited",
        }
        duplicate_question = {**base, "question_index": 1}
        samples = build_samples([duplicate_question, base])
        self.assertEqual(len(samples), 2)
        self.assertEqual([sample["target_color"] for sample in samples], ["red", "blue"])
        self.assertTrue(all(sample["target_x"] == 32 for sample in samples))

    def test_grid_index_is_normalized_and_bounded(self) -> None:
        self.assertEqual(grid_index(0, 14), 0)
        self.assertEqual(grid_index(128, 14), 7)
        self.assertEqual(grid_index(255, 14), 13)
        self.assertEqual(grid_index(256, 14), 13)

    def test_diagnosis_localizes_first_unavailable_stage(self) -> None:
        def metric(accuracy: float, lower: float) -> dict:
            return {
                "mechanism_dev": {
                    "overall": {
                        "accuracy_percent": accuracy,
                        "minus_chance": {"lower_95_ci_pp": lower},
                    }
                }
            }

        metrics = {
            "vision_196_local": metric(95.0, 70.0),
            "compressed_49_local": metric(68.0, 45.0),
            "projected_49_local": metric(80.0, 50.0),
        }
        result = diagnose_local_information(metrics, 70.0, 40.0)
        self.assertEqual(result["localization"], "196_to_49_compression_bottleneck")


if __name__ == "__main__":
    unittest.main(verbosity=2)
