#!/usr/bin/env python3
from __future__ import annotations

import unittest

from evaluate_highres_evidence_crop_bridge_v1 import select_checkpoint


def result(step, gate, joint, row, mismatch):
    return {
        "step": step,
        "gate_passed": gate,
        "metrics": {"global_crop": {"family_joint_accuracy": joint, "row_accuracy": row}},
        "paired_deltas": {"global_crop_minus_mismatched_crop": {"point_estimate": mismatch}},
    }


class CropBridgeEvaluationTests(unittest.TestCase):
    def test_selection_ignores_failed_gate(self) -> None:
        rows = [result(70, False, 1.0, 1.0, 1.0), result(140, True, 0.8, 0.9, 0.7)]
        self.assertEqual(select_checkpoint(rows)["step"], 140)

    def test_selection_prefers_joint_then_earlier_step(self) -> None:
        rows = [
            result(70, True, 0.8, 0.9, 0.7),
            result(140, True, 0.9, 0.9, 0.7),
            result(210, True, 0.9, 0.9, 0.7),
        ]
        self.assertEqual(select_checkpoint(rows)["step"], 140)

    def test_no_passing_checkpoint_returns_none(self) -> None:
        self.assertIsNone(select_checkpoint([result(70, False, 1.0, 1.0, 1.0)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
