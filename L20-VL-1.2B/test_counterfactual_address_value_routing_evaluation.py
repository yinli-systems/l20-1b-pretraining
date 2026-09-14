#!/usr/bin/env python3
from __future__ import annotations

import unittest

from evaluate_stage_a_visual_floor import BINDING_DEVELOPMENT_STATUSES
from run_counterfactual_address_value_routing_evaluation import select_checkpoint


class CAVREvaluationTests(unittest.TestCase):
    def test_v2_binding_protocol_uses_scene_pair_evaluation_path(self) -> None:
        self.assertTrue({
            "authorized_binding_answer_only_development_v2",
            "authorized_query_ce_gpu_time_matched_evaluation_v1",
        }.issubset(BINDING_DEVELOPMENT_STATUSES))

    def test_selection_uses_primary_then_strict_then_earliest(self) -> None:
        screens = [
            {"step": 100, "affected_question_joint_accuracy_percent": 20.0,
             "scene_pair_selective_all_six_correct_percent": 1.0},
            {"step": 150, "affected_question_joint_accuracy_percent": 25.0,
             "scene_pair_selective_all_six_correct_percent": 0.0},
            {"step": 200, "affected_question_joint_accuracy_percent": 25.0,
             "scene_pair_selective_all_six_correct_percent": 2.0},
            {"step": 250, "affected_question_joint_accuracy_percent": 25.0,
             "scene_pair_selective_all_six_correct_percent": 2.0},
        ]
        self.assertEqual(select_checkpoint(screens)["step"], 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
