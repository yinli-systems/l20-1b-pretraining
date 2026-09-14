#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from train_highres_evidence_crop_bridge_v1 import candidate_scores, epoch_rows, inject_global_and_crop


class TokenBridge(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))
        self.image_start = torch.nn.Parameter(torch.ones(1, 1, 3) * value)
        self.image_end = torch.nn.Parameter(torch.ones(1, 1, 3) * -value)

    def visual_tokens(self, features):
        return features * self.scale


class CropBridgeTrainingTests(unittest.TestCase):
    def test_epoch_shuffle_keeps_variants_adjacent(self) -> None:
        rows = [
            {"family_id": family, "variant": variant}
            for family in ("a", "b", "c")
            for variant in ("answer_change", "base")
        ]
        ordered = epoch_rows(rows, 17, 3)
        self.assertEqual(len(ordered), 6)
        for index in range(0, len(ordered), 2):
            self.assertEqual(ordered[index]["family_id"], ordered[index + 1]["family_id"])
            self.assertEqual(
                [ordered[index]["variant"], ordered[index + 1]["variant"]],
                ["answer_change", "base"],
            )

    def test_two_view_prefix_uses_isolated_crop_bridge(self) -> None:
        parent, crop = TokenBridge(1.0), TokenBridge(2.0)
        text = torch.zeros(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.long)
        labels = torch.zeros(2, 4, dtype=torch.long)
        features = torch.ones(2, 2, 3)
        inputs, expanded_mask, targets = inject_global_and_crop(
            parent, crop, text, mask, labels, features, features
        )
        self.assertEqual(inputs.shape, (2, 12, 3))
        self.assertTrue(torch.equal(inputs[:, 4:6], torch.full((2, 2, 3), 2.0)))
        self.assertTrue(expanded_mask[:, :8].bool().all())
        self.assertTrue(targets[:, :8].eq(-100).all())

    def test_candidate_score_excludes_eos_and_averages_answer_tokens(self) -> None:
        logits = torch.zeros(2, 4, 6)
        targets = torch.tensor([[-100, -100, 2, 0], [-100, 3, 4, 0]])
        logits[0, 1, 2] = 5.0
        logits[1, 0, 3] = 3.0
        logits[1, 1, 4] = 3.0
        scores = candidate_scores(logits, targets, eos_token_id=0, candidates_per_row=2)
        self.assertEqual(scores.shape, (1, 2))
        self.assertGreater(float(scores[0, 0]), float(scores[0, 1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
