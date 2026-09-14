import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build import REASONS, build


def write_json(path, value):
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildTest(unittest.TestCase):
    def test_extends_prior_and_binds_all_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / "prior.json"
            write_json(prior, {"status": "POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK",
                               "exclude_text_sha256_reasons": {"0" * 64: ["prior"]}})
            supp_rows = root / "candidate-exclusions.jsonl"
            supp_rows.write_text(json.dumps({"text_sha256": "1" * 64}) + "\n")
            supp = root / "supp.json"
            write_json(supp, {"status": "SUPPLEMENTAL_EXACT_SPAN_AUDIT_COMPLETE_ADMISSION_PENDING",
                              "candidates_sha256": sha(supp_rows)})
            legacy_rows = root / "legacy.jsonl"
            legacy_rows.write_text(json.dumps({"text_sha256": "2" * 64}) + "\n")
            legacy = root / "legacy-report.json"
            write_json(legacy, {"status": "LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED",
                                "candidate_rows": 1, "files": [{"candidates": str(legacy_rows),
                                "candidates_sha256": sha(legacy_rows)}]})
            old_rows = root / "old.matches.jsonl.gz"
            with gzip.open(old_rows, "wt") as handle:
                handle.write(json.dumps({"new_documents": [{"text_sha256": "3" * 64}]}) + "\n")
            old = root / "old-report.json"
            write_json(old, {"status": "OLD_SOURCE_NORMALIZED_OVERLAP_COMPLETE_NOT_ADMITTED",
                             "files": [{"matches": str(old_rows),
                             "matches_sha256": sha(old_rows)}]})

            result = build(prior, supp, legacy, old)
            self.assertEqual(result["unique_excluded_text_hashes"], 4)
            self.assertEqual(result["exclude_text_sha256_reasons"]["1" * 64],
                             [REASONS["supplemental"]])
            self.assertEqual(result["input_counts"]["legacy_candidate_rows"], 1)

    def test_rejects_changed_candidate_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / "prior.json"
            write_json(prior, {"status": "POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK",
                               "exclude_text_sha256_reasons": {}})
            rows = root / "candidate-exclusions.jsonl"
            rows.write_text("{}\n")
            supp = root / "supp.json"
            write_json(supp, {"status": "SUPPLEMENTAL_EXACT_SPAN_AUDIT_COMPLETE_ADMISSION_PENDING",
                              "candidates_sha256": "0" * 64})
            legacy = root / "legacy.json"
            write_json(legacy, {"status": "LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED",
                                "candidate_rows": 0, "files": []})
            old = root / "old.json"
            write_json(old, {"status": "OLD_SOURCE_NORMALIZED_OVERLAP_COMPLETE_NOT_ADMITTED",
                             "files": {}})
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build(prior, supp, legacy, old)


if __name__ == "__main__":
    unittest.main()
