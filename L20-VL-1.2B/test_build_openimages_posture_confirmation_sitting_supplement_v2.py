import unittest

from build_openimages_posture_confirmation_sitting_supplement_v2 import (
    assign_by_scarcity,
    caption_evidence,
    select_for_download,
)


class ConfirmationSittingSupplementTests(unittest.TestCase):
    def test_caption_evidence_is_class_specific(self):
        self.assertEqual(caption_evidence("Boy", "A child is sitting on a chair.")[:2], (True, 0))
        self.assertEqual(caption_evidence("Man", "A person is sitting on a chair.")[0], False)
        self.assertEqual(caption_evidence("Man", "A man is seated on a bench.")[:2], (True, 0))
        self.assertEqual(caption_evidence("Woman", "A woman is crouching by a chair.")[0], False)

    def test_scarcity_assignment_preserves_smaller_class(self):
        rows = [
            {"image_id": "shared", "class_name": "Boy", "curation_caption_tier": 0, "bbox_area": 0.2, "bbox_key": ("a",)},
            {"image_id": "shared", "class_name": "Man", "curation_caption_tier": 0, "bbox_area": 0.2, "bbox_key": ("b",)},
            {"image_id": "man-only", "class_name": "Man", "curation_caption_tier": 0, "bbox_area": 0.2, "bbox_key": ("c",)},
        ]
        assigned, counts = assign_by_scarcity(rows, 7)
        self.assertEqual({row["class_name"] for row in assigned if row["image_id"] == "shared"}, {"Boy"})
        self.assertEqual(counts["multi_class_images_resolved"], 1)

    def test_download_selection_obeys_per_class_quota(self):
        rows = []
        for class_name in ("Boy", "Girl", "Man", "Woman"):
            for index in range(3):
                rows.append({
                    "image_id": f"{class_name}-{index}",
                    "class_name": class_name,
                    "curation_caption_tier": index,
                    "bbox_key": (str(index),),
                })
        protocol = {
            "seed": 5,
            "sampling": {"download_candidates_per_class": {name: 2 for name in ("Boy", "Girl", "Man", "Woman")}},
        }
        selected = select_for_download(protocol, rows)
        self.assertEqual(len(selected), 8)
        self.assertEqual({name: sum(row["class_name"] == name for row in selected) for name in protocol["sampling"]["download_candidates_per_class"]}, {name: 2 for name in protocol["sampling"]["download_candidates_per_class"]})


if __name__ == "__main__":
    unittest.main()
