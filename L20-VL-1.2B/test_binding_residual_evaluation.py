#!/usr/bin/env python3
from __future__ import annotations

import unittest

from run_binding_residual_evaluation import paired_difference


class BindingResidualEvaluationTests(unittest.TestCase):
    def test_paired_difference_uses_affected_rows_and_scene_clusters(self) -> None:
        def result(candidate: bool) -> dict:
            rows = []
            for pair in ("a", "b"):
                rows.append({
                    "scene_family_id": f"{pair}-affected",
                    "statistical_cluster_id": pair,
                    "question_role": "affected_positive_to_negative",
                    "expected": {"base": "yes", "edited": "no", "invariant": "yes"},
                    "predictions": {
                        "true_image": {
                            "base": "yes",
                            "edited": "no" if candidate else "yes",
                            "invariant": "yes",
                        }
                    },
                })
                rows.append({
                    "scene_family_id": f"{pair}-control",
                    "statistical_cluster_id": pair,
                    "question_role": "invariant_control_positive",
                    "expected": {"base": "yes", "edited": "yes", "invariant": "yes"},
                    "predictions": {
                        "true_image": {"base": "yes", "edited": "yes", "invariant": "yes"}
                    },
                })
            return {"predictions": rows}

        comparison = paired_difference(result(True), result(False))
        self.assertEqual(comparison["estimate_pp"], 100.0)
        self.assertEqual(comparison["clusters"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
