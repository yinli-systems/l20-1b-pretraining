#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from train_counterfactual_address_value_router import (
    address_objective,
    counterfactual_value_features,
    grid_index,
    value_swap_objective,
)


class CounterfactualAddressValueRouterTests(unittest.TestCase):
    def test_grid_index_is_bounded(self) -> None:
        self.assertEqual(grid_index(0), 0)
        self.assertEqual(grid_index(128), 3)
        self.assertEqual(grid_index(255), 6)
        self.assertEqual(grid_index(256), 6)

    def test_address_objective_rewards_correct_consistent_attention(self) -> None:
        target = torch.tensor([5, 9])
        supervised = torch.tensor([True, True])
        uniform = torch.full((2, 3, 2, 49), 1 / 49)
        focused = uniform.clone()
        for family, index in enumerate(target):
            focused[family, :, :, index] = 0.9
            remainder = (1 - 0.9) / 48
            for token in range(49):
                if token != index:
                    focused[family, :, :, token] = remainder
        uniform_result = address_objective(uniform, target, supervised)
        focused_result = address_objective(focused, target, supervised)
        self.assertLess(focused_result["address_loss"], uniform_result["address_loss"])
        self.assertEqual(float(focused_result["address_accuracy"]), 1.0)
        self.assertLess(float(focused_result["address_pair_js"]), 1e-7)

    def test_candidate_gap_is_audited(self) -> None:
        attention = torch.full((1, 3, 2, 49), 1 / 49)
        attention[0, 0, 1, 0] += 0.1
        result = address_objective(attention, torch.tensor([0]), torch.tensor([True]))
        self.assertAlmostEqual(float(result["candidate_attention_max_gap"]), 0.1)

    def test_counterfactual_values_swap_only_base_and_edit(self) -> None:
        features = torch.arange(1 * 3 * 2, dtype=torch.float32).reshape(6, 1, 1)
        swapped = counterfactual_value_features(features, families=1).flatten().tolist()
        self.assertEqual(swapped, [2.0, 3.0, 0.0, 1.0, 4.0, 5.0])

    def test_value_swap_objective_uses_opposite_variant_labels(self) -> None:
        targets = torch.tensor([[0, 1, 0], [1, 0, 1]])
        supervised = torch.tensor([True, False])
        correct = torch.tensor(
            [[[0.0, 8.0], [8.0, 0.0], [8.0, 0.0]],
             [[8.0, 0.0], [0.0, 8.0], [0.0, 8.0]]]
        )
        wrong = -correct
        correct_result = value_swap_objective(correct, targets, supervised)
        wrong_result = value_swap_objective(wrong, targets, supervised)
        self.assertEqual(float(correct_result["value_swap_accuracy"]), 1.0)
        self.assertLess(
            float(correct_result["value_swap_loss"]),
            float(wrong_result["value_swap_loss"]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
