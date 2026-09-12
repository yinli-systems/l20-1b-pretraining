"""Protect paired evaluation alignment and bootstrap arithmetic."""

import copy
import unittest

import numpy as np

from analyze_comparison import align, boolq_diagnostics, bootstrap_difference, compare_task


def fixture(values):
    return {
        "versions": {"task": 1}, "n-shot": {"task": 0},
        "configs": {"task": {"dataset_path": "fixture", "num_fewshot": 0}},
        "n-samples": {"task": {"effective": len(values)}},
        "results": {"task": {"acc,none": float(np.mean(values))}},
        "samples": {"task": [
            {"doc_id": i, "doc_hash": f"doc{i}", "prompt_hash": f"prompt{i}",
             "target_hash": f"target{i}", "filter": "none", "acc": value}
            for i, value in enumerate(values)
        ]},
    }


class ComparisonTests(unittest.TestCase):
    def test_order_is_not_alignment(self):
        left = fixture([1, 0, 1, 0])
        right = fixture([0, 1, 1, 0])
        right["samples"]["task"].reverse()
        result, draws = compare_task(left, right, "task", "acc", np.random.default_rng(1), 10000)
        self.assertEqual(result["difference_pp"], 0)
        self.assertEqual(result["ours_better_samples"], 1)
        self.assertEqual(result["baseline_better_samples"], 1)
        self.assertEqual(result["tied_samples"], 2)
        self.assertAlmostEqual(float(draws.mean()), 0, delta=0.02)

    def test_identical_predictions_have_zero_paired_uncertainty(self):
        data = fixture([1, 0, 1, 0])
        pairs, _ = align(data, copy.deepcopy(data), "task", "acc")
        draws = bootstrap_difference(pairs, np.random.default_rng(1), 1000)
        self.assertTrue(np.all(draws == 0))

    def test_changed_prompt_target_or_document_fails(self):
        data = fixture([1, 0])
        for field in ("prompt_hash", "target_hash", "doc_hash"):
            with self.subTest(field=field):
                changed = copy.deepcopy(data)
                changed["samples"]["task"][0][field] = "different"
                with self.assertRaises(ValueError):
                    align(data, changed, "task", "acc")

    def test_mismatched_protocol_fails(self):
        data = fixture([1, 0])
        changed = copy.deepcopy(data)
        changed["n-shot"]["task"] = 5
        with self.assertRaises(ValueError):
            align(data, changed, "task", "acc")

    def test_duplicate_missing_and_wrong_mean_fail(self):
        data = fixture([1, 0])
        for change in ("duplicate", "missing", "wrong_mean"):
            with self.subTest(change=change):
                changed = copy.deepcopy(data)
                if change == "duplicate":
                    changed["samples"]["task"].append(changed["samples"]["task"][0])
                elif change == "missing":
                    changed["samples"]["task"].pop()
                else:
                    changed["results"]["task"]["acc,none"] = 0.25
                with self.assertRaises(ValueError):
                    align(data, changed, "task", "acc")

    def test_continuous_metric_bootstrap(self):
        pairs = np.array([[0.2, 0.1], [0.5, 0.8], [0.9, 0.2]])
        draws = bootstrap_difference(pairs, np.random.default_rng(1), 10000)
        self.assertAlmostEqual(float(draws.mean()), float((pairs[:, 0]-pairs[:, 1]).mean()), delta=0.01)

    def test_boolq_majority_baseline_is_not_model_accuracy(self):
        data = fixture([1, 0, 1, 0])
        for field in data:
            data[field]["boolq"] = data[field].pop("task")
        data["configs"]["boolq"]["doc_to_choice"] = ["no", "yes"]
        for sample, target in zip(data["samples"]["boolq"], [1, 1, 0, 1]):
            sample["target"] = target
            prediction = target if sample["acc"] else 1 - target
            scores = [-2.0, -2.0]
            scores[prediction] = -1.0
            sample["filtered_resps"] = [[score, False] for score in scores]
        result = boolq_diagnostics(data)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["majority_class_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
