#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from train_highres_evidence_router_v1 import balanced_epoch_batches, balanced_router_loss


class EvidenceRouterTrainingTests(unittest.TestCase):
    def test_balanced_batches_cover_each_row_once(self) -> None:
        rows = [
            {"expected_action": action}
            for action in ("POINT", "STOP")
            for _ in range(8)
        ]
        batches = balanced_epoch_batches(rows, 4, 7, 0)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(16)))
        for batch in batches:
            self.assertEqual(sum(rows[index]["expected_action"] == "POINT" for index in batch), 2)

    def test_loss_balances_point_and_stop_examples(self) -> None:
        logits = torch.zeros(4, 5)
        gains = torch.zeros(4)
        targets = torch.tensor([1, 2, 4, 4])
        point = torch.tensor([True, True, False, False])
        loss, parts = balanced_router_loss(logits, gains, targets, gains, point, 0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(parts["point_pointer_ce"]), float(parts["stop_pointer_ce"]), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
