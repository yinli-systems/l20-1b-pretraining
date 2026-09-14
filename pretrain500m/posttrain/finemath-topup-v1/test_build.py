import unittest

from build import quality_reasons, select_candidates


def row(row_number, family, tranche=3, selected=True, passed=True, reasons=(), partition="development"):
    return {"source_id": "finemath4", "tranche": tranche, "row": row_number,
            "text_sha256": f"{row_number:064x}", "family_id": family,
            "selected_eligible_representative": selected,
            "passes_filters_and_bound_exclusions": passed,
            "family_exclusion_reasons": list(reasons), "partition": partition}


class SelectionTest(unittest.TestCase):
    def test_rejects_family_bridged_to_prior_and_overrides_new_only_partition(self):
        old = row(0, "bridge", tranche=2)
        expanded = [old, row(10, "bridge"), row(11, "new"), row(12, "new", selected=False)]
        selected, rejected = select_candidates({("finemath4", 2, 0): old["text_sha256"]}, expanded)
        self.assertEqual([item["row"] for item in selected], [11])
        self.assertEqual(selected[0]["partition"], "development")
        self.assertEqual(rejected["observed_preexisting_family_bridge_documents"], 1)
        self.assertEqual(rejected["unselected_duplicate_or_ineligible_documents"], 1)

    def test_rejects_extreme_repetition_using_frozen_thresholds(self):
        text = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 200)
        policy = {"extreme_repetition_rule": {"repeated_10gram_coverage_at_least": .75,
                  "compression_ratio_at_most": .25}, "manual_rejects": {}}
        reasons, coverage, compression = quality_reasons(text, "0" * 64, policy)
        self.assertEqual(reasons, ["extreme_repetition_and_compression"])
        self.assertGreaterEqual(coverage, .75)
        self.assertLessEqual(compression, .25)


if __name__ == "__main__":
    unittest.main()
