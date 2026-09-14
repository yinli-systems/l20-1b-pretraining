import unittest

import audit_openimages_posture_candidates_v2_with_vitpose as module


def manifest_row(image_id="a", class_name="Man", answer="standing"):
    return {"image_id": image_id, "class_name": class_name, "answer": answer}


def decision_row(image_id="a", class_name="Man", posture="standing"):
    return {
        "image_id": image_id,
        "class_name": class_name,
        "posture": posture,
        "decision": "accept",
        "reason": "posture_visually_decidable",
    }


class V2PoseReplayTests(unittest.TestCase):
    def make_192(self):
        manifest = [manifest_row(str(index)) for index in range(192)]
        decisions = [decision_row(str(index)) for index in range(192)]
        return manifest, decisions

    def test_exact_join(self):
        manifest, decisions = self.make_192()
        merged = module.merge_examples(manifest, decisions)
        self.assertEqual(len(merged), 192)
        self.assertEqual(merged[0]["human_decision"], "accept")

    def test_mismatched_ids_fail_closed(self):
        manifest, decisions = self.make_192()
        decisions[-1]["image_id"] = "different"
        with self.assertRaisesRegex(ValueError, "image ids differ"):
            module.merge_examples(manifest, decisions)

    def test_mismatched_metadata_fails_closed(self):
        manifest, decisions = self.make_192()
        decisions[0]["posture"] = "sitting"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            module.merge_examples(manifest, decisions)


if __name__ == "__main__":
    unittest.main()
