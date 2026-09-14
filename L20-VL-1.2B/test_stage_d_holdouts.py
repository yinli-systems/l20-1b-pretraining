#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import math

from generate_counterfactual_diagnostic import POSITIONS
from generate_stage_d_holdouts import (
    BINDING_TASKS,
    build_binding_swap_scene,
    build_specs,
    evaluate_binding_scene,
    generate,
)


class StageDHoldoutTests(unittest.TestCase):
    def test_split_sizes_and_strata_are_exact(self) -> None:
        specs = build_specs(100, 200, 80, 11)
        self.assertEqual(len(specs), 380)
        self.assertEqual(len({row["scene_family_id"] for row in specs}), 380)
        self.assertEqual(Counter(row["split"] for row in specs), {
            "development": 100,
            "iid_test": 200,
            "ood_binding": 80,
        })
        strata = Counter((row["split"], row["task"], row["base_answer"]) for row in specs)
        self.assertTrue(all(count == 10 for key, count in strata.items() if key[0] == "development"))
        self.assertTrue(all(count == 20 for key, count in strata.items() if key[0] == "iid_test"))
        self.assertTrue(all(count == 20 for key, count in strata.items() if key[0] == "ood_binding"))
        challenges = Counter(
            (row["challenge"], row["task"], row["base_answer"])
            for row in specs if row["split"] == "ood_binding"
        )
        self.assertEqual(set(key[0] for key in challenges), {
            "binding_swap_only", "binding_swap_plus_boundary_geometry"
        })
        self.assertTrue(all(count == 10 for count in challenges.values()))

    def test_binding_swap_preserves_attribute_multisets_and_flips_answer(self) -> None:
        for task in BINDING_TASKS:
            for answer in ("yes", "no"):
                for seed in range(20):
                    scene = build_binding_swap_scene(task, answer, seed, "iid")
                    base = scene["base_state"]
                    edited = scene["edited_state"]
                    self.assertEqual(
                        Counter(item["color"] for item in base["objects"]),
                        Counter(item["color"] for item in edited["objects"]),
                    )
                    self.assertEqual(
                        Counter(item["shape"] for item in base["objects"]),
                        Counter(item["shape"] for item in edited["objects"]),
                    )
                    self.assertEqual(evaluate_binding_scene(task, base, scene["oracle_query"]), answer)
                    self.assertNotEqual(
                        evaluate_binding_scene(task, edited, scene["oracle_query"]), answer
                    )

    def test_ood_geometry_uses_non_grid_positions(self) -> None:
        standard = set(POSITIONS)
        seen_non_grid = False
        for seed in range(20):
            scene = build_binding_swap_scene("color_binding", "yes", seed, "ood_geometry")
            targets = scene["base_state"]["objects"][:2]
            seen_non_grid |= any((item["x"], item["y"]) not in standard for item in targets)
        self.assertTrue(seen_non_grid)

    def test_ood_objects_are_visible_and_non_overlapping(self) -> None:
        for seed in range(100):
            scene = build_binding_swap_scene("shape_binding", "yes", seed, "ood_geometry")
            objects = scene["base_state"]["objects"]
            for item in objects:
                self.assertGreaterEqual(item["x"] - item["size"], 0)
                self.assertGreaterEqual(item["y"] - item["size"], 0)
                self.assertLessEqual(item["x"] + item["size"], 256)
                self.assertLessEqual(item["y"] + item["size"], 256)
            for index, left in enumerate(objects):
                for right in objects[index + 1:]:
                    distance = math.hypot(left["x"] - right["x"], left["y"] - right["y"])
                    self.assertGreater(distance, left["size"] + right["size"])

    def test_bounded_generation_is_complete_and_collision_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"status": "test"}))
            protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
            exclusion = root / "exclude.jsonl"
            exclusion.write_text("")
            admission = root / "admission.json"
            output = root / "holdouts"
            receipt = root / "receipt.json"
            admission.write_text(json.dumps({
                "required_protocol_sha256": protocol_hash,
                "generator_authorized": True,
                "formal_training_authorized": False,
                "destination": str(output),
                "receipt": str(receipt),
                "exclude_manifest": {
                    "path": str(exclusion),
                    "sha256": hashlib.sha256(exclusion.read_bytes()).hexdigest(),
                },
                "development_families": 10,
                "iid_test_families": 10,
                "ood_binding_families": 8,
                "render_scale": 1,
                "max_total_bytes": 20_000_000,
                "seed": 9,
            }))
            result = generate(admission, protocol)
            self.assertEqual(result["scene_families"], 28)
            self.assertEqual(result["rendered_images"], 84)
            self.assertEqual(result["rendered_images"], result["unique_image_hashes"])
            self.assertEqual(result["exact_stage_c_image_overlap"], 0)
            self.assertEqual(result["exact_stage_c_family_overlap"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
