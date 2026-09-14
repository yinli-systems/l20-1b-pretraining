import unittest

import finalize_openimages_posture_pairs_v1 as module


def manifest_row(image_id, answer="standing", class_name="Boy"):
    return {"image_id": image_id, "answer": answer, "class_name": class_name}


def decision_row(image_id, decision="accept", posture="standing", class_name="Boy"):
    return {
        "image_id": image_id,
        "decision": decision,
        "reason": "reviewed",
        "posture": posture,
        "class_name": class_name,
    }


class FinalizePosturePairsTests(unittest.TestCase):
    def test_only_human_accepted_targets_survive(self):
        manifest = [manifest_row("a"), manifest_row("b")]
        decisions = [decision_row("a"), decision_row("b", decision="reject")]
        output = module.merge_human_accepted(manifest, decisions, "test")
        self.assertEqual([row["image_id"] for row in output], ["a"])

    def test_decision_metadata_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            module.merge_human_accepted(
                [manifest_row("a")],
                [decision_row("a", posture="sitting")],
                "test",
            )

    def test_balanced_pairing(self):
        protocol = {"seed": 5, "pair_counts_by_class": {"Boy": 1}}
        selected = [manifest_row("sit", answer="sitting"), manifest_row("stand")]
        pairs = module.make_pairs(protocol, selected)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["image_a"]["answer"], "sitting")
        self.assertEqual(pairs[0]["image_b"]["answer"], "standing")


if __name__ == "__main__":
    unittest.main()
