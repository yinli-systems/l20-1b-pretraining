import unittest

from record_openimages_posture_confirmation_sitting_v2_audit import validate_strata


class SittingV2AuditTests(unittest.TestCase):
    def test_complete_strata_are_summarized(self):
        manifest = [
            {"image_id": "a", "class_name": "Boy", "answer": "sitting"},
            {"image_id": "b", "class_name": "Boy", "answer": "sitting"},
            {"image_id": "c", "class_name": "Girl", "answer": "sitting"},
        ]
        review = {
            "answer": "sitting",
            "strata": [
                {"class_name": "Boy", "decisions": [
                    {"image_id": "a", "accepted": True, "reason": "accept_unambiguous"},
                    {"image_id": "b", "accepted": False, "reason": "posture_occluded"},
                ]},
                {"class_name": "Girl", "decisions": [
                    {"image_id": "c", "accepted": False, "reason": "held_not_sitting"},
                ]},
            ],
        }
        result = validate_strata(review, manifest)
        self.assertEqual([(x["reviewed"], x["accepted"]) for x in result], [(2, 1), (1, 0)])

    def test_incomplete_stratum_fails(self):
        with self.assertRaisesRegex(RuntimeError, "exactly"):
            validate_strata(
                {"answer": "sitting", "strata": [{"class_name": "Boy", "decisions": []}]},
                [{"image_id": "a", "class_name": "Boy", "answer": "sitting"}],
            )

    def test_reason_disagreement_fails(self):
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            validate_strata(
                {"answer": "sitting", "strata": [{"class_name": "Boy", "decisions": [
                    {"image_id": "a", "accepted": True, "reason": "posture_occluded"}
                ]}]},
                [{"image_id": "a", "class_name": "Boy", "answer": "sitting"}],
            )


if __name__ == "__main__":
    unittest.main()
