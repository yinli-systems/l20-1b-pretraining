import json
import unittest
from pathlib import Path

import record_openimages_posture_standing_supplement_v3_human_audit as module


class StandingSupplementV3HumanAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent
        paths = [
            root / "evidence" / "openimages-posture-standing-supplement-v3.jsonl",
            root / "data" / "openimages-posture-standing-supplement-v3" / "posture-standing-supplement-v3.jsonl",
        ]
        path = next((candidate for candidate in paths if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"standing supplement manifest not found: {paths}")
        cls.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_audit_is_exhaustive_and_unique(self):
        decisions = module.build_decisions(self.rows)
        self.assertEqual(len(decisions), 144)
        self.assertEqual(len({row["image_id"] for row in decisions}), 144)
        self.assertEqual(sum(row["decision"] == "reject" for row in decisions), 65)

    def test_every_stratum_passes_frozen_minimum(self):
        strata = module.summarize(module.build_decisions(self.rows))
        self.assertEqual(
            {key: value["accepted"] for key, value in strata.items()},
            module.EXPECTED_ACCEPTED_BY_STRATUM,
        )
        self.assertTrue(all(value["passes"] for value in strata.values()))


if __name__ == "__main__":
    unittest.main()
