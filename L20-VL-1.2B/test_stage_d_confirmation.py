#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from stage_d_confirmation_contract import (
    confirmation_execution_order,
    crossed_bootstrap_interval,
    family_joint,
    paired_t_interval,
)


class StageDConfirmationTests(unittest.TestCase):
    def test_execution_order_covers_each_cell_once(self) -> None:
        seeds = [1, 2, 3, 4, 5]
        order = confirmation_execution_order(seeds)
        cells = {(row["arm"], row["seed"], row["split"]) for row in order}
        self.assertEqual(len(order), 20)
        self.assertEqual(len(cells), 20)
        self.assertEqual(sum(row["arm"] == "A_spatial_49" for row in order), 10)
        self.assertEqual(sum(row["split"] == "iid_test" for row in order), 10)

    def test_family_joint_requires_base_and_edited(self) -> None:
        row = {
            "expected": {"base": "yes", "edited": "no"},
            "predictions": {"true_image": {"base": "yes", "edited": "yes"}},
        }
        self.assertEqual(family_joint(row), 0.0)
        row["predictions"]["true_image"]["edited"] = "no"
        self.assertEqual(family_joint(row), 1.0)

    def test_crossed_bootstrap_is_deterministic_and_paired(self) -> None:
        matrix = np.ones((5, 20), dtype=np.float64) * 0.02
        first = crossed_bootstrap_interval(matrix, resamples=200, seed=7)
        second = crossed_bootstrap_interval(matrix, resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate_pp"], 2.0)
        self.assertAlmostEqual(first["lower_95_ci_pp"], 2.0)
        self.assertAlmostEqual(first["upper_95_ci_pp"], 2.0)

    def test_paired_t_interval_contains_mean(self) -> None:
        result = paired_t_interval([-0.45, 0.0, -0.40, 0.50, 1.10])
        self.assertAlmostEqual(result["estimate_pp"], 0.15)
        self.assertLess(result["lower_95_ci_pp"], result["estimate_pp"])
        self.assertGreater(result["upper_95_ci_pp"], result["estimate_pp"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
