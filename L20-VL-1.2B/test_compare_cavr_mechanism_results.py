#!/usr/bin/env python3
from __future__ import annotations

import unittest

from compare_cavr_mechanism_results import paired_mechanism_difference


class CompareCAVRMechanismTests(unittest.TestCase):
    def test_paired_difference_clusters_scene_pairs(self) -> None:
        def payload(values):
            return {
                "records": [
                    {
                        "scene_family_id": family,
                        "scene_pair_id": pair,
                        "affected": True,
                        "causal_six_way": value,
                    }
                    for family, pair, value in values
                ]
            }

        cavr = payload([("a1", "a", 1), ("a2", "a", 1), ("b1", "b", 1)])
        query = payload([("a1", "a", 0), ("a2", "a", 1), ("b1", "b", 0)])
        result = paired_mechanism_difference(cavr, query, "causal_six_way")
        self.assertEqual(result["scene_pair_clusters"], 2)
        self.assertEqual(result["question_families"], 3)
        self.assertAlmostEqual(result["estimate_pp"], 75.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
