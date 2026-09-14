import unittest

import torch

from evidence_acquisition import (
    EvidenceAcquisitionRouter,
    build_evidence_targets,
    choose_evidence_action,
    evidence_router_loss,
    patch_index_to_crop_box,
)


class EvidenceAcquisitionTests(unittest.TestCase):
    def test_router_shapes_and_neutral_initial_gain(self):
        torch.manual_seed(1)
        router = EvidenceAcquisitionRouter(vision_dim=16, language_dim=24, rank=8)
        vision = torch.randn(3, 9, 16)
        text = torch.randn(3, 5, 24)
        mask = torch.ones(3, 5, dtype=torch.long)
        labels = torch.full((3, 5), -100, dtype=torch.long)
        logits, gain = router(vision, text, mask, labels)
        self.assertEqual(logits.shape, (3, 10))
        self.assertEqual(gain.shape, (3,))
        self.assertTrue(torch.equal(gain, torch.zeros_like(gain)))

    def test_masked_patch_cannot_be_selected(self):
        torch.manual_seed(2)
        router = EvidenceAcquisitionRouter(vision_dim=8, language_dim=12, rank=4)
        vision = torch.randn(1, 4, 8)
        text = torch.randn(1, 3, 12)
        text_mask = torch.ones(1, 3, dtype=torch.long)
        patch_mask = torch.tensor([[True, False, True, True]])
        logits, _ = router(vision, text, text_mask, patch_mask=patch_mask)
        self.assertTrue(torch.isneginf(logits[0, 1]))

    def test_counterfactual_gain_sets_patch_or_stop(self):
        targets = build_evidence_targets(
            torch.tensor([2.0, 1.0, 1.5]),
            torch.tensor([0.5, 0.9, 1.7]),
            torch.tensor([3, 2, 1], dtype=torch.long),
            patch_count=4,
            minimum_gain=0.25,
        )
        self.assertEqual(targets.pointer.tolist(), [3, 4, 4])
        self.assertEqual(targets.zoom_is_beneficial.tolist(), [True, False, False])

    def test_loss_backpropagates_through_router_outputs(self):
        logits = torch.randn(2, 5, requires_grad=True)
        gain = torch.randn(2, requires_grad=True)
        targets = build_evidence_targets(
            torch.tensor([2.0, 1.0]),
            torch.tensor([0.5, 1.1]),
            torch.tensor([2, 0], dtype=torch.long),
            patch_count=4,
            minimum_gain=0.2,
        )
        total, parts = evidence_router_loss(logits, gain, targets)
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(gain.grad)
        self.assertEqual(set(parts), {"pointer_ce", "gain_huber"})

    def test_action_requires_non_stop_gain_and_probability(self):
        logits = torch.tensor([[0.0, 4.0, -2.0], [0.0, 4.0, -2.0], [0.0, -2.0, 4.0]])
        gain = torch.tensor([0.8, 0.1, 1.0])
        action = choose_evidence_action(logits, gain, 0.5, 0.8)
        self.assertEqual(action.tolist(), [1, 2, 2])

    def test_crop_box_is_normalized_and_edge_clamped(self):
        self.assertEqual(patch_index_to_crop_box(0, 14, 1.0), (0.0, 0.0, 2 / 14, 2 / 14))
        box = patch_index_to_crop_box(195, 14, 1.0)
        for observed, expected in zip(box, (12 / 14, 12 / 14, 1.0, 1.0)):
            self.assertAlmostEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
