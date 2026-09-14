import unittest

from build_openimages_evidence_router_pilot_v1 import center_patch, expanded_xyxy, stable_key


class BuildOpenImagesEvidenceRouterPilotTests(unittest.TestCase):
    def test_center_patch_uses_xmin_xmax_ymin_ymax(self):
        self.assertEqual(center_patch([0.25, 0.50, 0.50, 0.75], 14), 8 * 14 + 5)

    def test_expansion_is_clipped(self):
        actual = expanded_xyxy([0.0, 0.2, 0.8, 1.0], 0.5)
        for value, expected in zip(actual, [0.0, 0.7, 0.3, 1.0]):
            self.assertAlmostEqual(value, expected)

    def test_stable_key_is_deterministic(self):
        self.assertEqual(stable_key(7, "a", 1), stable_key(7, "a", 1))
        self.assertNotEqual(stable_key(7, "a", 1), stable_key(8, "a", 1))


if __name__ == "__main__":
    unittest.main()
