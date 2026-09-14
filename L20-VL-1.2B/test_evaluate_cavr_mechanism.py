#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from evaluate_cavr_mechanism import candidate_attention_gap, summarize_records


class CAVRMechanismEvaluationTests(unittest.TestCase):
    def test_candidate_attention_gap_reduces_variant_and_token_axes(self) -> None:
        attention = torch.zeros(2, 3, 2, 49)
        attention[1, 2, 1, 7] = 0.25
        self.assertEqual(candidate_attention_gap(attention).tolist(), [0.0, 0.25])

    def test_summary_clusters_and_causal_metrics(self) -> None:
        base = {
            "question_role": "affected_target",
            "affected": True,
            "candidate_attention_max_gap": 0.0,
            "address_variant_accuracy": 1.0,
            "address_all_variants": True,
            "address_target_probability": 0.9,
            "address_base_edit_js": 0.01,
            "normal_base_edit_joint": True,
            "swapped_base_edit_joint": True,
            "invariant_normal_correct": True,
            "invariant_swapped_correct": True,
            "invariant_prediction_stable": True,
            "causal_six_way": True,
        }
        records = [
            {**base, "scene_pair_id": "pair-a", "scene_family_id": "a-1"},
            {**base, "scene_pair_id": "pair-a", "scene_family_id": "a-2", "causal_six_way": False},
            {**base, "scene_pair_id": "pair-b", "scene_family_id": "b-1"},
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["scene_pair_clusters"], 2)
        self.assertAlmostEqual(summary["affected_causal_six_way_percent"], 200.0 / 3.0)
        self.assertEqual(
            summary["cluster_bootstrap_95_ci"]["causal_six_way"]["scene_pair_clusters"],
            2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
