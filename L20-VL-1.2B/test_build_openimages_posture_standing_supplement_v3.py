import unittest

import build_openimages_posture_standing_supplement_v3 as module


FILTERS = {
    "min_width": 0.15,
    "max_width": 0.45,
    "min_height": 0.35,
    "min_area": 0.04,
    "min_bbox_height_to_width": 1.0,
}


def row(class_name="Man", attribute="Stand", caption="A man is standing outside.", bbox=None, title=""):
    return {
        "image_id": "new",
        "class_name": class_name,
        "attribute_name": attribute,
        "caption": caption,
        "bbox": bbox or [0.2, 0.45, 0.05, 0.9],
        "attribution": {"title": title},
    }


class StandingSupplementV3Tests(unittest.TestCase):
    def test_accepts_tall_narrow_standing_target(self):
        self.assertEqual(module.passes_standing_filters(row(), FILTERS), (True, "pass"))

    def test_excludes_non_target_strata(self):
        self.assertEqual(
            module.passes_standing_filters(row(class_name="Girl"), FILTERS),
            (False, "non_target_class"),
        )
        self.assertEqual(
            module.passes_standing_filters(row(attribute="Sit"), FILTERS),
            (False, "non_standing_attribute"),
        )

    def test_rejects_ambiguous_and_obvious_truncation_context(self):
        self.assertEqual(
            module.passes_standing_filters(row(caption="A man is crouching outside."), FILTERS),
            (False, "ambiguous_posture_term"),
        )
        self.assertEqual(
            module.passes_standing_filters(row(title="selfie portrait"), FILTERS),
            (False, "obvious_truncation_context_term"),
        )

    def test_rejects_closeup_geometry(self):
        self.assertEqual(
            module.passes_standing_filters(row(bbox=[0.1, 0.8, 0.0, 1.0]), FILTERS),
            (False, "excessive_width_closeup_risk"),
        )

    def test_prior_images_are_excluded_before_filtering(self):
        accepted, counts = module.standing_filter([row()], {"new"}, FILTERS)
        self.assertEqual(accepted, [])
        self.assertEqual(counts["prior_pilot_image_exclusion"], 1)

    def test_review_pool_is_ranked_by_frozen_pose_score(self):
        protocol = {"seed": 7, "sampling": {"review_candidates_per_stratum": 1}}
        rows = []
        for class_name in sorted(module.TARGET_CLASSES):
            rows.extend(
                [
                    {
                        "image_id": f"{class_name}-low",
                        "class_name": class_name,
                        "curation_pose_features": {"mean_lower_body_score": 0.1},
                    },
                    {
                        "image_id": f"{class_name}-high",
                        "class_name": class_name,
                        "curation_pose_features": {"mean_lower_body_score": 0.9},
                    },
                ]
            )
        selected = module.rank_review_rows(protocol, rows)
        self.assertEqual({row["image_id"] for row in selected}, {f"{name}-high" for name in module.TARGET_CLASSES})


if __name__ == "__main__":
    unittest.main()
