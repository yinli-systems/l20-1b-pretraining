#!/usr/bin/env python3
from __future__ import annotations

import unittest

from run_query_ce_time_matched_evaluation import decision_flags


class TimeMatchedEvaluationTests(unittest.TestCase):
    def test_decision_requires_compute_match_effect_and_ci(self) -> None:
        passed = decision_flags(
            {"estimate_pp": 20.0, "lower_95_ci_pp": 10.0}, 1.02, [0.9, 1.1], 5.0
        )
        self.assertTrue(all(passed.values()))
        failed = decision_flags(
            {"estimate_pp": 20.0, "lower_95_ci_pp": -1.0}, 1.2, [0.9, 1.1], 5.0
        )
        self.assertFalse(all(failed.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
