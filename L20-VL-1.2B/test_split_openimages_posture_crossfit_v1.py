import unittest

import split_openimages_posture_crossfit_v1 as module


def make_pair(pair_id: str, class_name: str) -> dict:
    return {
        "pair_id": pair_id,
        "class_name": class_name,
        "candidate_answers": ["sitting", "standing"],
        "image_a": {
            "image_id": f"{pair_id}-sit",
            "image_sha256": f"sha-{pair_id}-sit",
            "answer": "sitting",
            "class_name": class_name,
            "human_audit": {"decision": "accept"},
        },
        "image_b": {
            "image_id": f"{pair_id}-stand",
            "image_sha256": f"sha-{pair_id}-stand",
            "answer": "standing",
            "class_name": class_name,
            "human_audit": {"decision": "accept"},
        },
    }


class PostureCrossfitSplitTests(unittest.TestCase):
    def setUp(self):
        counts = {"Boy": 18, "Girl": 16, "Man": 23, "Woman": 20}
        self.pairs = [
            make_pair(f"{class_name}-{index:02d}", class_name)
            for class_name, count in counts.items()
            for index in range(count)
        ]

    def test_assignment_is_deterministic_and_balanced(self):
        first = module.assign_folds(self.pairs, fold_count=5, seed=20261005)
        second = module.assign_folds(list(reversed(self.pairs)), fold_count=5, seed=20261005)
        first_map = {row["pair_id"]: row["crossfit"]["fold"] for row in first}
        second_map = {row["pair_id"]: row["crossfit"]["fold"] for row in second}
        self.assertEqual(first_map, second_map)
        module.validate_assignment(first, 5)
        fold_sizes = [sum(row["crossfit"]["fold"] == fold for row in first) for fold in range(5)]
        self.assertEqual(sorted(fold_sizes), [15, 15, 15, 16, 16])

    def test_pair_validation_rejects_image_reuse(self):
        rows = [make_pair("a", "Boy"), make_pair("b", "Boy")]
        rows[1]["image_a"]["image_id"] = rows[0]["image_a"]["image_id"]
        with self.assertRaisesRegex(ValueError, "image leakage"):
            module.validate_pairs(rows, expected_pairs=2)

    def test_pair_validation_rejects_nonaccepted_image(self):
        row = make_pair("a", "Boy")
        row["image_b"]["human_audit"]["decision"] = "reject"
        with self.assertRaisesRegex(ValueError, "non-accepted"):
            module.validate_pairs([row], expected_pairs=1)


if __name__ == "__main__":
    unittest.main()
