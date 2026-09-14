#!/usr/bin/env python3
from __future__ import annotations

import unittest

from evaluate_highres_evidence_router_v1 import router_metrics, select_checkpoint


class EvidenceRouterEvaluationTests(unittest.TestCase):
    def test_router_metrics_distinguish_action_exact_and_near(self) -> None:
        rows = [
            {"expected_action": "POINT", "target_patch_index_14x14": 15},
            {"expected_action": "POINT", "target_patch_index_14x14": 30},
            {"expected_action": "STOP", "target_patch_index_14x14": None},
        ]
        metrics = router_metrics(rows, [15, 31, 196], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0], 196)
        self.assertEqual(metrics["action_accuracy"], 1.0)
        self.assertEqual(metrics["exact_patch_accuracy"], 0.5)
        self.assertEqual(metrics["within_one_patch_accuracy"], 1.0)
        self.assertEqual(metrics["stop_accuracy"], 1.0)

    def test_selection_prefers_gate_then_accuracy_then_earlier(self) -> None:
        def item(step, gate, action, near, exact, mae):
            return {"step": step, "gate_passed": gate, "metrics": {
                "action_accuracy": action,
                "within_one_patch_accuracy": near,
                "exact_patch_accuracy": exact,
                "gain_mae": mae,
            }}
        rows = [item(210, True, 1, 1, .8, .5), item(420, True, 1, 1, .8, .5), item(630, False, 1, 1, 1, 0)]
        self.assertEqual(select_checkpoint(rows)["step"], 210)

    def test_stop_on_point_target_is_not_a_near_hit(self) -> None:
        rows = [{"expected_action": "POINT", "target_patch_index_14x14": 195}]
        metrics = router_metrics(rows, [196], [0.0], [2.0], 196)
        self.assertEqual(metrics["within_one_patch_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
