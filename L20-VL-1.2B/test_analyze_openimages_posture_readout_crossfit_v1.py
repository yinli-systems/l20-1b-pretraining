import unittest

from analyze_openimages_posture_readout_crossfit_v1 import pair_joint_map


class AnalyzePostureReadoutTests(unittest.TestCase):
    def test_pair_joint_map(self):
        rows = [
            {"pair_id": "a", "correct": True},
            {"pair_id": "a", "correct": True},
            {"pair_id": "b", "correct": True},
            {"pair_id": "b", "correct": False},
        ]
        self.assertEqual(pair_joint_map(rows), {"a": 100.0, "b": 0.0})

    def test_pair_joint_rejects_incomplete_pair(self):
        with self.assertRaisesRegex(ValueError, "incomplete pair"):
            pair_joint_map([{"pair_id": "a", "correct": True}])


if __name__ == "__main__":
    unittest.main()
