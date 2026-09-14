import math
import unittest

import numpy as np

from audit_openimages_posture_with_vitpose import angle_degrees, segment_verticality, summarize_pose


class PoseFeatureTests(unittest.TestCase):
    def test_geometry_primitives(self):
        self.assertAlmostEqual(segment_verticality(np.array([0.0, 0.0]), np.array([0.0, 2.0])), 1.0)
        self.assertAlmostEqual(segment_verticality(np.array([0.0, 0.0]), np.array([2.0, 0.0])), 0.0)
        self.assertAlmostEqual(
            angle_degrees(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])),
            90.0,
        )

    def test_summarize_straight_visible_leg(self):
        points = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros(17, dtype=np.float32)
        points[11], points[13], points[15] = (10, 10), (10, 30), (10, 50)
        scores[[11, 13, 15]] = 0.9
        result = summarize_pose(points, scores, [0, 0, 20, 100], 0.5)
        self.assertEqual(result["complete_visible_legs"], 1)
        self.assertEqual(result["visible_lower_body_keypoints"], 3)
        self.assertAlmostEqual(result["median_thigh_verticality"], 1.0)
        self.assertAlmostEqual(result["median_shank_verticality"], 1.0)
        self.assertAlmostEqual(result["median_knee_angle_degrees"], 180.0)

    def test_summarize_missing_leg_abstains(self):
        points = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros(17, dtype=np.float32)
        result = summarize_pose(points, scores, [0, 0, 20, 100], 0.5)
        self.assertEqual(result["complete_visible_legs"], 0)
        self.assertIsNone(result["median_knee_angle_degrees"])
        self.assertTrue(math.isfinite(result["mean_lower_body_score"]))


if __name__ == "__main__":
    unittest.main()
