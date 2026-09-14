#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from train_stage_c_counterfactual import FamilyBatchBuilder


class TokenizerStub:
    eos_token_id = 0
    pad_token_id = 1

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [2 + index for index, _ in enumerate(text.split())]


class ProcessorStub:
    def __call__(self, images, return_tensors: str):
        self.last_count = len(images)
        if return_tensors != "pt":
            raise AssertionError(return_tensors)
        return {"pixel_values": torch.zeros(len(images), 3, 4, 4)}


class StageDTrainingTests(unittest.TestCase):
    def test_answer_only_batch_does_not_require_teacher_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for variant in ("base", "edited", "invariant"):
                path = root / f"{variant}.png"
                Image.new("RGB", (4, 4), "white").save(path)
                paths[variant] = str(path)
            row = {
                "scene_family_id": "family",
                "task": "count",
                "question": "Are there two circles?",
                "candidate_answers": ["yes", "no"],
                "base_answer": "yes",
                "edited_answer": "no",
                "invariant_answer": "yes",
                **{f"{variant}_image_path": path for variant, path in paths.items()},
            }
            processor = ProcessorStub()
            batch = FamilyBatchBuilder(TokenizerStub(), processor, 32, None)([row])
            self.assertEqual(processor.last_count, 3)
            self.assertEqual(batch["target_indices"].tolist(), [[0, 1, 0]])
            self.assertEqual(batch["teacher_scores"].shape, (1, 3, 2))
            self.assertFalse(batch["teacher_correct"].any())
            self.assertFalse(batch["teacher_pair_correct"].any())

    def test_shared_scene_images_are_deduplicated_without_changing_feature_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for variant in ("base", "edited", "invariant"):
                path = root / f"{variant}.png"
                Image.new("RGB", (4, 4), "white").save(path)
                paths[variant] = str(path)
            rows = []
            for index in range(2):
                rows.append({
                    "scene_family_id": f"family-{index}",
                    "task": "binding_intervention",
                    "question": f"Question {index}?",
                    "candidate_answers": ["yes", "no"],
                    "base_answer": "yes",
                    "edited_answer": "no",
                    "invariant_answer": "yes",
                    **{f"{variant}_image_path": path for variant, path in paths.items()},
                })
            processor = ProcessorStub()
            batch = FamilyBatchBuilder(
                TokenizerStub(), processor, 32, None, deduplicate_images=True
            )(rows)
            self.assertEqual(processor.last_count, 3)
            self.assertEqual(batch["image_feature_indices"].tolist(), [[0, 1, 2], [0, 1, 2]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
