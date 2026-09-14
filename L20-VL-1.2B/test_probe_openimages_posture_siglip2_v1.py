import unittest

import numpy as np

import probe_openimages_posture_siglip2_v1 as module


class FrozenSiglip2PostureProbeTests(unittest.TestCase):
    def test_patch_region_indices_falls_back_to_center(self):
        indices = module.patch_region_indices([0.01, 0.02, 0.01, 0.02], grid=14)
        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0], 0)

    def test_roi_pool_uses_only_bbox_cells(self):
        features = np.arange(4, dtype=np.float32).reshape(1, 4, 1)
        pooled = module.roi_pool(features, [[0.0, 0.5, 0.0, 1.0]], grid=2)
        self.assertEqual(float(pooled[0, 0]), 1.0)

    def test_evaluate_predictions_requires_balanced_pairs(self):
        protocol = {
            "statistics": {
                "bootstrap_resamples": 10,
                "confidence": 0.95,
                "bootstrap_seed": 1,
            }
        }
        targets = [
            {"pair_id": "p", "class_name": "Boy", "expected": "sitting"},
            {"pair_id": "p", "class_name": "Boy", "expected": "sitting"},
        ]
        with self.assertRaisesRegex(ValueError, "invalid paired labels"):
            module.evaluate_predictions(
                targets,
                np.asarray([0, 0]),
                np.asarray([-1.0, -0.5]),
                protocol,
            )


if __name__ == "__main__":
    unittest.main()
