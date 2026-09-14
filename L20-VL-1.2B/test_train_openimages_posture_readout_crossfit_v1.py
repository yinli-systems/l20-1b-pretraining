import unittest

import torch

from train_openimages_posture_readout_crossfit_v1 import (
    folds_from_pairs,
    pair_ranking_objective,
    region_attention_objective,
)


class PostureReadoutCrossfitTests(unittest.TestCase):
    def test_pair_atomic_folds(self):
        pairs = [
            {"pair_id": f"p{index}", "crossfit": {"fold": index % 5}}
            for index in range(10)
        ]
        folds = folds_from_pairs(pairs, 5)
        self.assertEqual([len(folds[index]) for index in range(5)], [2, 2, 2, 2, 2])
        self.assertEqual(len({row["pair_id"] for rows in folds.values() for row in rows}), 10)

    def test_region_attention_mass(self):
        attention = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],
                [[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]],
            ]
        )
        loss, mass, top1, gap = region_attention_objective(attention, [[2], [0, 1]])
        self.assertAlmostEqual(float(mass), 0.8, places=6)
        self.assertAlmostEqual(float(top1), 1.0, places=6)
        self.assertAlmostEqual(float(gap), 0.0, places=6)
        self.assertGreater(float(loss), 0.0)

    def test_pair_ranking(self):
        targets = [
            {"pair_id": "p", "expected": "sitting"},
            {"pair_id": "p", "expected": "standing"},
        ]
        scores = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        loss, separation, accuracy = pair_ranking_objective(scores, targets, 1.0)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(separation), 6.0)
        self.assertEqual(float(accuracy), 1.0)

    def test_pair_ranking_rejects_incomplete_pair(self):
        with self.assertRaisesRegex(ValueError, "incomplete pair"):
            pair_ranking_objective(
                torch.tensor([[1.0, 0.0]]),
                [{"pair_id": "p", "expected": "sitting"}],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
