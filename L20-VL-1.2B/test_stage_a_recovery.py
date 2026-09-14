#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from types import SimpleNamespace

from counterfactual_losses import candidate_classification_loss
from train_stage_a_recovery import recovery_rows
from modeling import vision_features


class StageARecoveryTests(unittest.TestCase):
    def test_vision_features_selects_declared_layer(self) -> None:
        class DummyVision(torch.nn.Module):
            def forward(self, pixel_values, output_hidden_states=False):
                hidden = (pixel_values + 1, pixel_values + 2, pixel_values + 3)
                return SimpleNamespace(
                    last_hidden_state=hidden[-1],
                    hidden_states=hidden if output_hidden_states else None,
                )

        pixels = torch.zeros(2, 4, 8)
        self.assertTrue(torch.equal(vision_features(DummyVision(), pixels, -1), pixels + 3))
        self.assertTrue(torch.equal(vision_features(DummyVision(), pixels, -2), pixels + 2))

    def test_recovery_rows_use_only_train_and_are_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            rows = []
            for index in range(3500):
                answer = "yes" if index % 2 == 0 else "no"
                other = "no" if answer == "yes" else "yes"
                rows.append({
                    "split": "train", "scene_family_id": f"family-{index}", "task": "count",
                    "question": "Is the fact true?", "base_answer": answer,
                    "edited_answer": other, "invariant_answer": answer,
                    "base_image_path": "base.png", "edited_image_path": "edited.png",
                    "invariant_image_path": "invariant.png"
                })
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            expanded = recovery_rows(path, "train", epochs=2, seed=7)
            self.assertEqual(len(expanded), 21000)
            self.assertEqual(sum(row["response"] == "yes" for row in expanded), 10500)
            self.assertEqual(sum(row["response"] == "no" for row in expanded), 10500)
            self.assertEqual({row["variant"] for row in expanded}, {"base", "edited", "invariant"})
            self.assertTrue(all(row["candidate_answers"] == ["yes", "no"] for row in expanded))

    def test_candidate_classification_loss_prefers_expected_candidate(self) -> None:
        logits = torch.full((4, 3, 8), -4.0)
        targets = torch.tensor([[-100, 2, 0], [-100, 3, 0], [-100, 2, 0], [-100, 3, 0]])
        logits[0, 0, 2] = 4.0
        logits[0, 1, 0] = 4.0
        logits[1, 0, 3] = 0.0
        logits[1, 1, 0] = 0.0
        logits[2, 0, 2] = 0.0
        logits[2, 1, 0] = 0.0
        logits[3, 0, 3] = 4.0
        logits[3, 1, 0] = 4.0
        loss, accuracy = candidate_classification_loss(
            logits, targets, torch.tensor([0, 1])
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(accuracy), 0.99)

    def test_recovery_rejects_non_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text("")
            with self.assertRaisesRegex(RuntimeError, "only the train split"):
                recovery_rows(path, "development", epochs=1, seed=7)

    def test_recovery_accepts_declared_family_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            rows = []
            for index in range(4):
                answer = "yes" if index % 2 == 0 else "no"
                rows.append({
                    "split": "train", "scene_family_id": f"family-{index}", "task": "count",
                    "question": "Is the fact true?", "base_answer": answer,
                    "edited_answer": "no" if answer == "yes" else "yes", "invariant_answer": answer,
                    "base_image_path": "base.png", "edited_image_path": "edited.png",
                    "invariant_image_path": "invariant.png",
                })
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            expanded = recovery_rows(path, "train", epochs=1, seed=7, expected_families=4)
            self.assertEqual(len(expanded), 12)

    def test_recovery_task_repeats_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            rows = []
            tasks = ["color_binding", "count"]
            for index in range(3500):
                task = tasks[index % 2]
                answer = "yes" if (index // 2) % 2 == 0 else "no"
                other = "no" if answer == "yes" else "yes"
                rows.append({
                    "split": "train", "scene_family_id": f"family-{index}", "task": task,
                    "question": "Is the fact true?", "base_answer": answer,
                    "edited_answer": other, "invariant_answer": answer,
                    "base_image_path": "base.png", "edited_image_path": "edited.png",
                    "invariant_image_path": "invariant.png", "candidate_answers": ["yes", "no"]
                })
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            expanded = recovery_rows(
                path, "train", epochs=1, seed=7,
                task_repeats={"color_binding": 3, "count": 1},
            )
            self.assertEqual(len(expanded), (1750 * 3 + 1750) * 3)
            by_source = {source: sum(row["source"] == source for row in expanded) for source in {row["source"] for row in expanded}}
            self.assertEqual(by_source["counterfactual_color_binding"], 15750)
            self.assertEqual(by_source["counterfactual_count"], 5250)


if __name__ == "__main__":
    unittest.main(verbosity=2)
