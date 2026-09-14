import unittest

from build_openimages_unmarked_grounding_pilot_v1 import bbox_tuple, build_rows, stable_key


class OpenImagesUnmarkedGroundingBuilderTests(unittest.TestCase):
    def test_bbox_tuple_rounds_official_coordinates(self):
        row = {"XMin1": "0.1000004", "XMax1": "0.9", "YMin1": "0", "YMax1": "1"}
        self.assertEqual(bbox_tuple(row), (0.1, 0.9, 0.0, 1.0))

    def test_stable_key_is_deterministic(self):
        self.assertEqual(stable_key(3, "x"), stable_key(3, "x"))
        self.assertNotEqual(stable_key(3, "x"), stable_key(4, "x"))

    def test_only_unique_class_instance_is_admitted(self):
        source = {
            "source_image_id": "image",
            "class_id": "class",
            "class_name": "Bottle",
            "source_bbox_xmin_xmax_ymin_ymax": [0.1, 0.3, 0.2, 0.4],
            "split": "train",
            "source_image_path": __file__,
            "source_image_sha256": "hash",
            "license": "license",
            "attribution": {},
            "expected_action": "POINT",
            "answer": "unused",
            "global_answer": "A bottle is on the table.",
        }
        rows = build_rows([source], {("image", "class"): {(0.1, 0.3, 0.2, 0.4)}}, 0.15, 14)
        self.assertEqual(len(rows), 2)
        point = next(row for row in rows if row["expected_action"] == "POINT")
        stop = next(row for row in rows if row["expected_action"] == "STOP")
        self.assertEqual(point["image_path"], stop["image_path"])
        self.assertIn("only bottle", point["question"])
        self.assertEqual(stop["answer"], "A bottle is on the table.")

    def test_ambiguous_class_instance_is_rejected(self):
        source = {"source_image_id": "image", "class_id": "class"}
        boxes = {("image", "class"): {(0.1, 0.2, 0.1, 0.2), (0.3, 0.4, 0.3, 0.4)}}
        self.assertEqual(build_rows([source], boxes, 0.15, 14), [])


if __name__ == "__main__":
    unittest.main()
