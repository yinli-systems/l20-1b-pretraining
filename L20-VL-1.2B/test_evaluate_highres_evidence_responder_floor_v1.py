import unittest

import numpy as np
import torch
from torch import nn

from evaluate_highres_evidence_responder_floor_v1 import (
    condition_metrics,
    inject_two_views,
    mismatched_crop_mapping,
    paired_bootstrap,
)


class DummyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_start = nn.Parameter(torch.ones(1, 1, 3))
        self.image_end = nn.Parameter(torch.full((1, 1, 3), 2.0))

    def visual_tokens(self, features):
        return features


class HighresEvidenceResponderFloorTests(unittest.TestCase):
    def test_two_view_injection_masks_visual_prefix(self):
        bridge = DummyBridge()
        text = torch.zeros(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.long)
        labels = torch.zeros(2, 4, dtype=torch.long)
        global_features = torch.randn(2, 2, 3)
        crop_features = torch.randn(2, 2, 3)
        inputs, output_mask, targets = inject_two_views(
            bridge, text, mask, labels, global_features, crop_features
        )
        self.assertEqual(inputs.shape, (2, 12, 3))
        self.assertEqual(output_mask.shape, (2, 12))
        self.assertTrue(torch.equal(targets[:, :8], torch.full((2, 8), -100)))

    def test_mismatched_crop_never_uses_same_family(self):
        rows = []
        for family in ("a", "b", "c"):
            for variant in ("base", "answer_change"):
                rows.append({"family_id": family, "variant": variant, "task": "read_local_digit"})
        mapping = mismatched_crop_mapping(rows)
        self.assertTrue(all(source[0] != target[0] for source, target in mapping.items()))

    def test_metrics_and_paired_delta(self):
        rows = [
            {"family_id": "a", "answer": "1"},
            {"family_id": "a", "answer": "2"},
            {"family_id": "b", "answer": "3"},
            {"family_id": "b", "answer": "4"},
        ]
        left = ["1", "2", "3", "4"]
        right = ["0", "0", "3", "0"]
        self.assertEqual(condition_metrics(rows, left)["family_joint_accuracy"], 1.0)
        result = paired_bootstrap(rows, left, right, 1000, 1)
        self.assertEqual(result["point_estimate"], 0.75)


if __name__ == "__main__":
    unittest.main()
