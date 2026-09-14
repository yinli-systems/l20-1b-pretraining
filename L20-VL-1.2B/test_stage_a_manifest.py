#!/usr/bin/env python3
from __future__ import annotations

import unittest

from prepare_stage_a_manifest import caption_gate, deduplicate_images, hamming_distance


class StageAManifestTests(unittest.TestCase):
    def test_caption_gate_accepts_descriptive_non_person_scene(self) -> None:
        accepted, reason = caption_gate(
            "A bright red ceramic vase is standing beside a blue bowl on a wooden table."
        )
        self.assertTrue(accepted, reason)

    def test_caption_gate_rejects_person_and_contact_information(self) -> None:
        self.assertEqual(caption_gate("A man is standing near several colorful objects on a table.")[1], "word_count")
        person_caption = (
            "A man is standing beside a blue bowl and several bright objects on a wooden table."
        )
        self.assertEqual(caption_gate(person_caption)[1], "person_term")
        possessive_caption = (
            "A person's hand holds a silver watch beside a wooden table and a clear glass bottle."
        )
        self.assertEqual(caption_gate(possessive_caption)[1], "person_term")
        kid_caption = (
            "A small kid sits beside a red chair with toys spread across the wooden floor."
        )
        self.assertEqual(caption_gate(kid_caption)[1], "person_term")
        contact_caption = (
            "A product sign beside a red vase contains contact address test@example.com for more details."
        )
        self.assertEqual(caption_gate(contact_caption)[1], "pii_or_url_pattern")

    def test_near_duplicate_is_rejected_deterministically(self) -> None:
        canonical = {
            "selection_sha256": "0" * 64,
            "image_sha256": "a",
            "dhash64": "0000000000000000",
            "image_id": "a",
        }
        duplicate = {
            "selection_sha256": "1" * 64,
            "image_sha256": "b",
            "dhash64": "0000000000000001",
            "image_id": "b",
        }
        kept, rejected = deduplicate_images([duplicate, canonical])
        self.assertEqual([record["image_id"] for record in kept], ["a"])
        self.assertEqual(rejected[0]["canonical"], "a")
        self.assertEqual(hamming_distance(0, 1), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
