#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from train_binding_residual import intervention_feature_pairs, intervention_target_pairs


class BindingResidualTrainingTests(unittest.TestCase):
    def test_interchange_pairs_swap_only_residual_source(self) -> None:
        features = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        cores, residuals = intervention_feature_pairs(features)
        torch.testing.assert_close(cores[:, 0], features[:, 0])
        torch.testing.assert_close(cores[:, 1], features[:, 1])
        torch.testing.assert_close(residuals[:, 0], features[:, 1])
        torch.testing.assert_close(residuals[:, 1], features[:, 0])

    def test_interchange_targets_follow_residual_source(self) -> None:
        targets = torch.tensor([[0, 1, 0], [1, 0, 1]])
        expected = torch.tensor([[1, 0], [0, 1]])
        torch.testing.assert_close(intervention_target_pairs(targets), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
