import unittest

from train_openimages_posture_readout_candidate_v1 import (
    FROZEN_OPTIMIZATION,
    validate_frozen_recipe,
    validate_selection_receipt,
)


class PostureReadoutCandidateTests(unittest.TestCase):
    def test_accepts_frozen_selection_and_recipe(self):
        validate_selection_receipt(
            {
                "status": "complete_audited_method_development_only",
                "decision": {
                    "selected_for_new_sealed_confirmation": "query_ce",
                    "do_not_select": "grounded_query",
                },
            }
        )
        validate_frozen_recipe(
            {
                "selected_arm": "query_ce",
                "loss_weights": {"answer": 1.0, "address": 0.0, "pair_rank": 0.0},
                "optimization": dict(FROZEN_OPTIMIZATION),
            }
        )

    def test_rejects_unselected_grounded_recipe(self):
        with self.assertRaisesRegex(RuntimeError, "query_ce"):
            validate_selection_receipt(
                {
                    "status": "complete_audited_method_development_only",
                    "decision": {
                        "selected_for_new_sealed_confirmation": "grounded_query",
                        "do_not_select": "query_ce",
                    },
                }
            )

    def test_rejects_hyperparameter_drift(self):
        changed = dict(FROZEN_OPTIMIZATION)
        changed["epochs"] = 13
        with self.assertRaisesRegex(RuntimeError, "optimization"):
            validate_frozen_recipe(
                {
                    "selected_arm": "query_ce",
                    "loss_weights": {"answer": 1.0, "address": 0.0, "pair_rank": 0.0},
                    "optimization": changed,
                }
            )

    def test_rejects_auxiliary_loss_drift(self):
        with self.assertRaisesRegex(RuntimeError, "objective"):
            validate_frozen_recipe(
                {
                    "selected_arm": "query_ce",
                    "loss_weights": {"answer": 1.0, "address": 0.1, "pair_rank": 0.0},
                    "optimization": dict(FROZEN_OPTIMIZATION),
                }
            )


if __name__ == "__main__":
    unittest.main()
