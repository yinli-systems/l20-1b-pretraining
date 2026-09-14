#!/usr/bin/env python3
from __future__ import annotations

import unittest

from train_stage_a_full_token import balanced_rows, cosine_lr, encode_prompt_response


class TokenizerStub:
    eos_token_id = 0

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [2 + index for index, _ in enumerate(text.split())]


class StageATrainingTests(unittest.TestCase):
    def test_prompt_is_masked_and_response_is_supervised(self) -> None:
        ids, labels = encode_prompt_response(
            TokenizerStub(), "What color is the object?", "blue", 32
        )
        self.assertEqual(len(ids), len(labels))
        self.assertEqual(labels[-1], TokenizerStub.eos_token_id)
        self.assertGreater(sum(label == -100 for label in labels), 0)
        self.assertGreater(sum(label != -100 for label in labels), 0)

    def test_source_balance_and_shuffle_are_deterministic(self) -> None:
        left = [{"id": f"a-{i}"} for i in range(4)]
        right = [{"id": f"b-{i}"} for i in range(4)]
        self.assertEqual(balanced_rows(left, right, 7), balanced_rows(left, right, 7))
        with self.assertRaises(ValueError):
            balanced_rows(left, right[:-1], 7)

    def test_cosine_schedule_respects_floor(self) -> None:
        self.assertEqual(cosine_lr(1, 10, 2, 0.1), 0.5)
        self.assertAlmostEqual(cosine_lr(10, 10, 2, 0.1), 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
