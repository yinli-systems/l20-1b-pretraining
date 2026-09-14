import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluate_highres_evidence_router_confirmation_v1 import confirmation_gate, load_confirmation_rows


class EvidenceRouterConfirmationTests(unittest.TestCase):
    def test_loader_returns_only_confirmation_split(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            rows = [
                {"family_id": "b", "variant": "base", "split": "router_confirmation"},
                {"family_id": "a", "variant": "base", "split": "train"},
                {"family_id": "b", "variant": "answer_change", "split": "router_confirmation"},
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            selected = load_confirmation_rows(manifest, "router_confirmation")
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(row["split"] == "router_confirmation" for row in selected))

    def test_gate_requires_router_and_local_policy(self):
        metrics = {
            "action_accuracy": 1.0,
            "point_recall": 1.0,
            "stop_accuracy": 1.0,
            "exact_patch_accuracy": 1.0,
            "within_one_patch_accuracy": 1.0,
            "gain_mae": 0.5,
        }
        policy = {
            "local_evidence": {"row_accuracy": 1.0, "family_joint_accuracy": 1.0},
            "candidate_order_invariance": {"POINT_ACTION": True, "STOP_ACTION": True},
        }
        gate = {
            "minimum_action_accuracy": 0.9,
            "minimum_point_recall": 0.9,
            "minimum_stop_accuracy": 0.9,
            "minimum_exact_patch_accuracy": 0.5,
            "minimum_within_one_patch_accuracy": 0.8,
            "maximum_gain_mae": 1.0,
            "minimum_local_policy_row_accuracy": 0.9,
            "minimum_local_policy_family_accuracy": 0.9,
        }
        self.assertTrue(confirmation_gate(metrics, policy, gate))
        policy["local_evidence"]["family_joint_accuracy"] = 0.8
        self.assertFalse(confirmation_gate(metrics, policy, gate))


if __name__ == "__main__":
    unittest.main()
