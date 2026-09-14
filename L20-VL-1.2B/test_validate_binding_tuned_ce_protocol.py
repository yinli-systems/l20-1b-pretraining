#!/usr/bin/env python3
from __future__ import annotations

import unittest

from validate_binding_tuned_ce_protocol import expected_optimizer_steps


class TunedCEProtocolTests(unittest.TestCase):
    def test_expected_optimizer_steps(self) -> None:
        self.assertEqual(expected_optimizer_steps(4800, 12, 1), 400)

    def test_rejects_partial_effective_batch(self) -> None:
        with self.assertRaises(ValueError):
            expected_optimizer_steps(4801, 12, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
