import unittest

from evaluate_openimages_evidence_router_zero_shot_v1 import bbox_targets, localization_flags, wilson_interval


class OpenImagesEvidenceRouterZeroShotTests(unittest.TestCase):
    def test_bbox_targets_use_normalized_xy_ranges(self):
        center, region = bbox_targets([0.25, 0.50, 0.50, 0.75], 14)
        self.assertEqual(center, 8 * 14 + 5)
        self.assertIn(center, region)

    def test_localization_flags(self):
        result = localization_flags(15, 0, {15}, 14)
        self.assertFalse(result["exact_center"])
        self.assertTrue(result["within_one_center"])
        self.assertTrue(result["inside_bbox"])

    def test_wilson_interval_contains_estimate(self):
        interval = wilson_interval(8, 10)
        self.assertLess(interval["lower_95_ci"], interval["estimate"])
        self.assertGreater(interval["upper_95_ci"], interval["estimate"])


if __name__ == "__main__":
    unittest.main()
