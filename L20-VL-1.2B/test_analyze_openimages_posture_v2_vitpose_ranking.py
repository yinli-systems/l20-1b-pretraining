import unittest

import analyze_openimages_posture_v2_vitpose_ranking as module


def row(label, value):
    return {"human_decision": label, "pose_features": {"mean_lower_body_score": value}}


class V2VitPoseRankingAnalysisTests(unittest.TestCase):
    def test_auc_perfect_and_reversed(self):
        values = [row("accept", 0.9), row("accept", 0.8), row("reject", 0.2), row("reject", 0.1)]
        self.assertEqual(module.auc(values), 1.0)
        self.assertEqual(module.auc(values, lambda item: -module.score(item)), 0.0)

    def test_auc_tie(self):
        self.assertEqual(module.auc([row("accept", 0.5), row("reject", 0.5)]), 0.5)

    def test_wilson_interval_contains_observed_fraction(self):
        lower, upper = module.wilson_interval(7, 8)
        self.assertLess(lower, 7 / 8)
        self.assertGreater(upper, 7 / 8)


if __name__ == "__main__":
    unittest.main()
