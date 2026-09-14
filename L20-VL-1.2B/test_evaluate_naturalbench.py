import itertools
import unittest

from evaluate_naturalbench import (
    answer_token_mask,
    balanced_rank_components,
    bootstrap_interval,
    group_components,
    LazyDatasetRows,
    mean_metrics,
    paired_comparison,
    subgroup_metrics,
    validate_dataset,
)


class NaturalBenchMetricTests(unittest.TestCase):
    def test_lazy_rows_only_fetch_requested_records(self):
        class TrackingDataset:
            def __init__(self):
                self.accessed = []

            def __len__(self):
                return 5

            def __getitem__(self, index):
                self.accessed.append(index)
                return {"index": index}

        dataset = TrackingDataset()
        rows = LazyDatasetRows(dataset)
        self.assertEqual(dataset.accessed, [])
        self.assertEqual(rows[1:3], [{"index": 1}, {"index": 2}])
        self.assertEqual(dataset.accessed, [1, 2])

    def test_balanced_ranking_removes_a_shared_answer_offset(self):
        margins = {
            "q0_i0": 3.0,
            "q0_i1": 1.0,
            "q1_i0": 2.0,
            "q1_i1": 4.0,
        }
        expected = {
            "Balanced_Q_Acc": 1.0,
            "Balanced_I_Acc": 1.0,
            "Balanced_G_Acc": 1.0,
        }
        self.assertEqual(balanced_rank_components(margins), expected)
        shifted = {key: value + 100.0 for key, value in margins.items()}
        self.assertEqual(balanced_rank_components(shifted), expected)

    def test_balanced_ranking_counts_ties_as_failure(self):
        tied = {key: 0.0 for _, _, key in (
            (0, 0, "q0_i0"),
            (0, 1, "q0_i1"),
            (1, 0, "q1_i0"),
            (1, 1, "q1_i1"),
        )}
        self.assertEqual(balanced_rank_components(tied)["Balanced_G_Acc"], 0.0)

    def test_candidate_score_excludes_eos_and_padding(self):
        import torch

        labels = torch.tensor([
            [-100, -100, 42, 2],
            [-100, 42, 2, -100],
        ])
        targets = torch.cat((torch.full((2, 2), -100), labels), dim=1)
        attention = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        mask = answer_token_mask(targets, labels, attention)
        expected = torch.zeros((2, 5), dtype=torch.bool)
        expected[0, 3] = True
        expected[1, 2] = True
        self.assertTrue(torch.equal(mask, expected))

    def test_all_binary_patterns_match_official_equations(self):
        expected = (1, 0, 0, 1)
        keys = ("q0_i0", "q0_i1", "q1_i0", "q1_i1")
        for predictions in itertools.product((0, 1), repeat=4):
            official = {
                "Q_Acc": (
                    float(predictions[0] == 1 and predictions[1] == 0)
                    + float(predictions[3] == 1 and predictions[2] == 0)
                ) / 2,
                "I_Acc": (
                    float(predictions[0] == 1 and predictions[2] == 0)
                    + float(predictions[3] == 1 and predictions[1] == 0)
                ) / 2,
                "Acc": sum(float(a == b) for a, b in zip(predictions, expected)) / 4,
                "G_Acc": float(predictions == expected),
            }
            ours = group_components({
                key: prediction == target
                for key, prediction, target in zip(keys, predictions, expected)
            })
            self.assertEqual(ours, official)

    def test_perfect_group(self):
        components = group_components({
            "q0_i0": True,
            "q0_i1": True,
            "q1_i0": True,
            "q1_i1": True,
        })
        self.assertEqual(components, {"Q_Acc": 1.0, "I_Acc": 1.0, "Acc": 1.0, "G_Acc": 1.0})

    def test_constant_positive_pattern_matches_official_failure_modes(self):
        components = group_components({
            "q0_i0": True,
            "q0_i1": False,
            "q1_i0": False,
            "q1_i1": True,
        })
        self.assertEqual(components, {"Q_Acc": 0.0, "I_Acc": 0.0, "Acc": 0.5, "G_Acc": 0.0})

    def test_mean_and_paired_bootstrap(self):
        perfect = {
            "Q_Acc": 1.0,
            "I_Acc": 1.0,
            "Acc": 1.0,
            "G_Acc": 1.0,
            "Balanced_Q_Acc": 1.0,
            "Balanced_I_Acc": 1.0,
            "Balanced_G_Acc": 1.0,
        }
        zero = {key: 0.0 for key in perfect}
        self.assertEqual(mean_metrics([perfect, zero])["G_Acc"], 0.5)
        interval = bootstrap_interval([1.0] * 8, resamples=100, seed=7)
        self.assertEqual(interval["lower_95_ci_percent"], 100.0)
        comparison = paired_comparison([perfect] * 4, [zero] * 4, resamples=100, seed=7)
        self.assertEqual(comparison["G_Acc"]["estimate_pp"], 100.0)
        self.assertEqual(comparison["G_Acc"]["lower_95_ci_pp"], 100.0)

    def test_dataset_pair_invariant(self):
        row = {
            "Question Type": "yes_no",
            "Source": "DOCCI",
            "Image_0_Question_0": "Yes",
            "Image_1_Question_0": "No",
            "Image_0_Question_1": "No",
            "Image_1_Question_1": "Yes",
        }
        audit = validate_dataset([row], 1)
        self.assertTrue(audit["passes"])
        row["Image_1_Question_1"] = "No"
        self.assertFalse(validate_dataset([row], 1)["passes"])

    def test_subgroups_do_not_mix_sources(self):
        perfect = {"Q_Acc": 1.0, "I_Acc": 1.0, "Acc": 1.0, "G_Acc": 1.0}
        perfect_rank = {
            "Balanced_Q_Acc": 1.0,
            "Balanced_I_Acc": 1.0,
            "Balanced_G_Acc": 1.0,
        }
        zero = {key: 0.0 for key in perfect}
        zero_rank = {key: 0.0 for key in perfect_rank}
        predictions = [
            {
                "source": "A",
                "metric_components": perfect,
                "balanced_rank_components": perfect_rank,
            },
            {
                "source": "B",
                "metric_components": zero,
                "balanced_rank_components": zero_rank,
            },
        ]
        result = subgroup_metrics(predictions, "source")
        self.assertEqual(result["A"]["official_metrics"]["G_Acc"], 1.0)
        self.assertEqual(result["B"]["official_metrics"]["G_Acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
