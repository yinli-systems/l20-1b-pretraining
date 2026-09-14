#!/usr/bin/env python3
from __future__ import annotations

import unittest

from run_binding_tuned_ce_evaluation import compact_metrics


class TunedCEEvaluationTests(unittest.TestCase):
    def test_screen_metrics_do_not_require_control_conditions(self) -> None:
        result = {
            "binding_selective_metrics": {
                "per_condition": {
                    "true_image": {
                        "affected_question_joint_accuracy_percent": 25.0,
                        "invariant_question_joint_accuracy_percent": 50.0,
                        "scene_pair_selective_all_six_correct_percent": 5.0,
                    }
                }
            }
        }
        metrics = compact_metrics(result)
        self.assertEqual(metrics["affected_question_joint_accuracy_percent"], 25.0)
        self.assertNotIn("visual_floor", metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
