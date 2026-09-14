import unittest

from record_openimages_posture_confirmation_early_audit_v1 import validate_decisions


class ConfirmationEarlyAuditTests(unittest.TestCase):
    def test_complete_stratum_review(self):
        manifest = [
            {"image_id": "a", "class_name": "Boy", "answer": "sitting"},
            {"image_id": "b", "class_name": "Boy", "answer": "sitting"},
            {"image_id": "c", "class_name": "Girl", "answer": "sitting"},
        ]
        review = {
            "class_name": "Boy",
            "answer": "sitting",
            "decisions": [
                {"image_id": "a", "accepted": True, "reason": "accept_unambiguous"},
                {"image_id": "b", "accepted": False, "reason": "posture_occluded"},
            ],
        }
        reviewed, accepted, reasons = validate_decisions(review, manifest)
        self.assertEqual((reviewed, accepted), (2, 1))
        self.assertEqual(reasons["posture_occluded"], 1)

    def test_missing_target_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exactly"):
            validate_decisions(
                {
                    "class_name": "Boy",
                    "answer": "sitting",
                    "decisions": [
                        {"image_id": "a", "accepted": True, "reason": "accept_unambiguous"}
                    ],
                },
                [
                    {"image_id": "a", "class_name": "Boy", "answer": "sitting"},
                    {"image_id": "b", "class_name": "Boy", "answer": "sitting"},
                ],
            )

    def test_acceptance_reason_disagreement_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            validate_decisions(
                {
                    "class_name": "Boy",
                    "answer": "sitting",
                    "decisions": [
                        {"image_id": "a", "accepted": True, "reason": "posture_occluded"}
                    ],
                },
                [{"image_id": "a", "class_name": "Boy", "answer": "sitting"}],
            )


if __name__ == "__main__":
    unittest.main()
