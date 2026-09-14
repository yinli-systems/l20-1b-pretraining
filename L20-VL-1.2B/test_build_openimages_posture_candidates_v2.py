import unittest

import build_openimages_posture_candidates_v2 as module


def row(caption="A person is sitting on a chair.", attribute="Sit", bbox=None):
    return {
        "caption": caption,
        "attribute_name": attribute,
        "bbox": bbox or [0.1, 0.6, 0.1, 0.8],
    }


class StrictPostureCandidateTests(unittest.TestCase):
    def test_sitting_requires_support_context(self):
        self.assertEqual(module.passes_strict_filters(row(), {"min_edge": 0.3, "min_area": 0.08}), (True, "pass"))
        self.assertEqual(
            module.passes_strict_filters(row("A person is sitting outside."), {"min_edge": 0.3, "min_area": 0.08}),
            (False, "sitting_without_support_term"),
        )

    def test_ambiguous_posture_is_rejected(self):
        keep, reason = module.passes_strict_filters(
            row("A person is crouching beside a chair."), {"min_edge": 0.3, "min_area": 0.08}
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "ambiguous_posture_term")

    def test_geometry_is_stricter_than_first_pilot(self):
        keep, reason = module.passes_strict_filters(
            row(bbox=[0.1, 0.35, 0.1, 0.9]), {"min_edge": 0.3, "min_area": 0.08}
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "small_edge")

    def test_first_pilot_ids_are_excluded(self):
        candidate = {**row(), "image_id": "old"}
        accepted, counts = module.strict_filter([candidate], {"old"}, {"min_edge": 0.3, "min_area": 0.08})
        self.assertEqual(accepted, [])
        self.assertEqual(counts["first_pilot_image_exclusion"], 1)


if __name__ == "__main__":
    unittest.main()
