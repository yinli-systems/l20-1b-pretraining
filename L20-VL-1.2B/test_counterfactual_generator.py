#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math
import tempfile
import unittest

from PIL import Image

from generate_counterfactual_diagnostic import (
    TASKS,
    build_scene,
    build_specs,
    evaluate_scene,
    render_scene,
)


class CounterfactualGeneratorTests(unittest.TestCase):
    def test_split_is_scene_atomic_balanced_and_exact(self) -> None:
        specs = build_specs(1000, 7)
        self.assertEqual(len(specs), 1000)
        self.assertEqual(len({item["scene_family_id"] for item in specs}), 1000)
        self.assertEqual(sum(item["split"] == "train" for item in specs), 700)
        self.assertEqual(sum(item["split"] == "development" for item in specs), 150)
        self.assertEqual(sum(item["split"] == "test" for item in specs), 150)
        for task in TASKS:
            for answer in ("yes", "no"):
                stratum = [item for item in specs if item["task"] == task and item["base_answer"] == answer]
                self.assertEqual([sum(item["split"] == split for item in stratum) for split in ("train", "development", "test")], [70, 15, 15])

    def test_every_task_flips_relevant_answer_and_preserves_invariant_semantics(self) -> None:
        for task in TASKS:
            for answer in ("yes", "no"):
                scene = build_scene(task, answer, 12345)
                self.assertTrue(scene["relevant_edit"]["answer_must_change"])
                self.assertFalse(scene["invariant_edit"]["answer_must_change"])
                self.assertNotEqual(scene["base_state"], scene["edited_state"])
                self.assertNotEqual(scene["base_state"], scene["invariant_state"])
                self.assertEqual(scene["base_state"]["objects"], scene["invariant_state"]["objects"])
                expected_edit = "no" if answer == "yes" else "yes"
                self.assertEqual(evaluate_scene(task, scene["base_state"], scene["oracle_query"]), answer)
                self.assertEqual(evaluate_scene(task, scene["edited_state"], scene["oracle_query"]), expected_edit)
                self.assertEqual(evaluate_scene(task, scene["invariant_state"], scene["oracle_query"]), answer)

    def test_render_is_deterministic_and_variants_are_distinct(self) -> None:
        scene = build_scene("left_right_relation", "yes", 99)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, edited = root / "first.png", root / "second.png", root / "edited.png"
            render_scene(scene["base_state"], first, 3)
            render_scene(scene["base_state"], second, 3)
            render_scene(scene["edited_state"], edited, 3)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotEqual(first.read_bytes(), edited.read_bytes())
            with Image.open(first) as image:
                self.assertEqual(image.size, (256, 256))

    def test_relation_variants_have_no_object_overlap(self) -> None:
        for task in ("left_right_relation", "above_below_relation"):
            for answer in ("yes", "no"):
                for seed in range(100):
                    scene = build_scene(task, answer, seed)
                    for variant in ("base_state", "edited_state", "invariant_state"):
                        objects = scene[variant]["objects"]
                        for index, left in enumerate(objects):
                            for right in objects[index + 1:]:
                                distance = math.hypot(left["x"] - right["x"], left["y"] - right["y"])
                                self.assertGreater(distance, left["size"] + right["size"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
