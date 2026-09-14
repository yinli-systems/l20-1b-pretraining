import unittest

import build_openimages_attribute_routing_data as module


class OpenImagesAttributeRoutingBuilderTests(unittest.TestCase):
    def test_internal_split_keeps_parent_development_sealed(self):
        policy = {"seed": 7, "bucket_count": 20, "selection_buckets": [0, 1, 2]}
        self.assertEqual(
            module.internal_split({"image_id": "x", "split": "development"}, policy),
            "sealed_confirmation",
        )

    def test_internal_split_is_deterministic_for_parent_train(self):
        policy = {"seed": 7, "bucket_count": 20, "selection_buckets": [0, 1, 2]}
        row = {"image_id": "abc", "split": "train"}
        self.assertEqual(module.internal_split(row, policy), module.internal_split(row, policy))
        self.assertIn(module.internal_split(row, policy), {"train", "selection_dev"})

    def test_display_attribute_is_frozen_and_fail_closed(self):
        self.assertEqual(module.display_attribute("(made of)Leather"), "leather")
        self.assertEqual(module.display_attribute("Stand"), "standing")
        with self.assertRaisesRegex(RuntimeError, "no frozen answer"):
            module.display_attribute("Transparent")

    def test_choose_control_excludes_forbidden_images(self):
        candidates = [
            {"image_id": "a", "bbox_key": ("0",)},
            {"image_id": "b", "bbox_key": ("1",)},
        ]
        chosen = module.choose_control(candidates, {"a"}, (1, "test"))
        self.assertEqual(chosen["image_id"], "b")
        self.assertIsNone(module.choose_control(candidates, {"a", "b"}, (1, "test")))


if __name__ == "__main__":
    unittest.main()
