import json
import unittest
from pathlib import Path

import record_openimages_posture_candidates_v2_human_audit as module


class PostureCandidatesV2HumanAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent
        candidates = [
            root / "evidence" / "openimages-posture-candidates-v2.jsonl",
            root / "data" / "openimages-posture-candidates-v2" / "posture-candidates-v2.jsonl",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"candidate manifest not found in known layouts: {candidates}")
        cls.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_audit_is_exhaustive_and_unique(self):
        decisions = module.build_decisions(self.rows)
        self.assertEqual(len(decisions), 192)
        self.assertEqual(len({item["image_id"] for item in decisions}), 192)
        self.assertEqual(sum(item["decision"] == "reject" for item in decisions), 67)

    def test_frozen_stratum_counts(self):
        strata = module.summarize(module.build_decisions(self.rows))
        accepted = {key: value["accepted"] for key, value in strata.items()}
        self.assertEqual(accepted, module.EXPECTED_ACCEPTED_BY_STRATUM)
        self.assertEqual(
            sorted(key for key, value in strata.items() if not value["passes"]),
            ["Boy/standing", "Man/standing", "Woman/standing"],
        )


if __name__ == "__main__":
    unittest.main()
