import unittest

import build_openimages_posture_pilot as module


class OpenImagesPosturePilotTests(unittest.TestCase):
    def test_caption_corroboration_requires_class_and_posture(self):
        self.assertTrue(module.caption_supports("A woman is sitting on a bench.", "Woman", "Sit"))
        self.assertFalse(module.caption_supports("A woman is near a bench.", "Woman", "Sit"))
        self.assertFalse(module.caption_supports("Someone is sitting on a bench.", "Woman", "Sit"))

    def test_caption_corroboration_rejects_secondary_media(self):
        self.assertFalse(module.caption_supports("A statue of a standing woman.", "Woman", "Stand"))
        self.assertFalse(module.caption_supports("A boy is sitting in a newspaper photo.", "Boy", "Sit"))
        self.assertFalse(module.caption_supports("Drawings of women standing.", "Woman", "Stand"))

    def test_pairing_is_balanced_and_deterministic(self):
        protocol = {"seed": 9}
        selected = []
        for class_name in module.CLASS_PATTERNS:
            for attribute_name, answer in (("Sit", "sitting"), ("Stand", "standing")):
                for index in range(2):
                    selected.append({
                        "image_id": f"{class_name}-{attribute_name}-{index}",
                        "class_name": class_name,
                        "attribute_name": attribute_name,
                        "answer": answer,
                    })
        first = module.make_pairs(protocol, selected)
        second = module.make_pairs(protocol, list(reversed(selected)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertTrue(all(row["image_a"]["answer"] == "sitting" for row in first))
        self.assertTrue(all(row["image_b"]["answer"] == "standing" for row in first))

    def test_download_selection_adds_deduplication_order_key(self):
        candidates = []
        for class_name in module.CLASS_PATTERNS:
            for attribute_name in module.ATTRIBUTE_PATTERNS:
                candidates.append({
                    "image_id": f"{class_name}-{attribute_name}",
                    "class_name": class_name,
                    "attribute_name": attribute_name,
                    "bbox_key": ("0", "1", "0", "1"),
                })
        protocol = {"seed": 9, "sampling": {"download_candidates_per_stratum": 1}}
        selected = module.select_download_candidates(protocol, candidates)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(len(row["selection_sha256"]) == 64 for row in selected))


if __name__ == "__main__":
    unittest.main()
