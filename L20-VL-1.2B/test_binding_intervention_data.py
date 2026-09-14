#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from generate_binding_intervention_data import (
    CHALLENGES,
    QUESTION_ROLES,
    SPLITS,
    build_scene_pair,
    build_specs,
    evaluate_query,
    generate,
)
from audit_binding_intervention_data import audit


class BindingInterventionDataTests(unittest.TestCase):
    def partitions(self, count: int = 4):
        return {
            "train": {"scene_pairs": count, "distribution": "grid"},
            "mechanism_dev": {"scene_pairs": count, "distribution": "grid"},
            "selection_dev": {"scene_pairs": count, "distribution": "jitter"},
            "final_test": {"scene_pairs": count, "distribution": "boundary"},
        }

    def test_specs_are_balanced_and_pair_atomic(self) -> None:
        specs = build_specs(self.partitions(8), 17)
        self.assertEqual(len(specs), 32)
        self.assertEqual(len({row["scene_pair_id"] for row in specs}), 32)
        self.assertEqual(Counter(row["split"] for row in specs), {split: 8 for split in SPLITS})
        self.assertEqual(
            Counter((row["split"], row["challenge"]) for row in specs),
            {(split, challenge): 4 for split in SPLITS for challenge in CHALLENGES},
        )

    def test_every_scene_has_balanced_selective_questions(self) -> None:
        for challenge in CHALLENGES:
            for distribution in ("grid", "jitter", "boundary"):
                for seed in range(20):
                    scene = build_scene_pair(challenge, seed, distribution)
                    self.assertEqual(
                        {row["question_role"] for row in scene["questions"]}, set(QUESTION_ROLES)
                    )
                    base_colors = Counter(item["color"] for item in scene["base_state"]["objects"])
                    edited_colors = Counter(item["color"] for item in scene["edited_state"]["objects"])
                    base_shapes = Counter(item["shape"] for item in scene["base_state"]["objects"])
                    edited_shapes = Counter(item["shape"] for item in scene["edited_state"]["objects"])
                    self.assertEqual(base_colors, edited_colors)
                    self.assertEqual(base_shapes, edited_shapes)
                    answer_pairs = Counter()
                    for row in scene["questions"]:
                        answers = row["answers"]
                        for variant in ("base", "edited", "invariant"):
                            self.assertEqual(
                                answers[variant],
                                evaluate_query(scene[f"{variant}_state"], row["oracle_query"]),
                            )
                        self.assertEqual(answers["base"], answers["invariant"])
                        answer_pairs[(answers["base"], answers["edited"])] += 1
                    self.assertEqual(answer_pairs, {
                        ("yes", "no"): 1,
                        ("no", "yes"): 1,
                        ("yes", "yes"): 2,
                        ("no", "no"): 2,
                    })

    def test_bounded_generation_has_shared_images_and_exact_position_histograms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            output = root / "data"
            receipt_path = root / "receipt.json"
            protocol.write_text(json.dumps({
                "status": "authorized_binding_intervention_data_generation_v1",
                "model_training_authorized_by_this_protocol": False,
                "external_downloads_authorized": False,
                "final_test_unseal_authorized": True,
                "seed": 23,
                "render_scale": 1,
                "max_total_image_bytes": 50_000_000,
                "output": str(output),
                "receipt": str(receipt_path),
                "partitions": self.partitions(4),
            }))
            receipt = generate(protocol)
            self.assertEqual(receipt["scene_pairs"], 16)
            self.assertEqual(receipt["question_families"], 96)
            self.assertEqual(receipt["rendered_images"], 48)
            self.assertEqual(receipt["unique_image_hashes"], 48)
            self.assertIn("deterministic_collision_retries", receipt)
            self.assertTrue(receipt["invariants"]["position_swap_exact_rgb_histogram_preserved"])
            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len({row["scene_family_id"] for row in rows}), 96)
            self.assertEqual(len({row["statistical_cluster_id"] for row in rows}), 16)
            for pair_id in {row["scene_pair_id"] for row in rows}:
                family = [row for row in rows if row["scene_pair_id"] == pair_id]
                self.assertEqual(len(family), 6)
                self.assertEqual(len({row["base_image_sha256"] for row in family}), 1)
                self.assertEqual(len({row["split"] for row in family}), 1)
            audit_path = root / "audit.json"
            audit_result = audit(protocol, receipt_path, audit_path)
            self.assertEqual(audit_result["status"], "passed_static_and_render_audit_pending_human_review")
            self.assertEqual(audit_result["independently_rerendered_images"], 48)
            self.assertTrue(audit_result["gates"]["final_test_remains_absent"] is False)

    def test_invalid_partition_count_fails_closed(self) -> None:
        partitions = self.partitions(4)
        partitions["train"]["scene_pairs"] = 3
        with self.assertRaises(ValueError):
            build_specs(partitions, 3)

    def test_sealed_final_test_cannot_be_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({
                "status": "authorized_binding_intervention_data_generation_v1",
                "model_training_authorized_by_this_protocol": False,
                "external_downloads_authorized": False,
                "final_test_unseal_authorized": False,
                "seed": 7,
                "render_scale": 1,
                "max_total_image_bytes": 50_000_000,
                "output": str(root / "data"),
                "receipt": str(root / "receipt.json"),
                "partitions": self.partitions(4),
            }))
            with self.assertRaisesRegex(RuntimeError, "remains sealed"):
                generate(protocol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
