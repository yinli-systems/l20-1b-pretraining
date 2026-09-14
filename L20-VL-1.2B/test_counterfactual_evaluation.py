#!/usr/bin/env python3
from __future__ import annotations

import unittest

from assemble_counterfactual_predictions import correctness
from evaluate_counterfactual_predictions import evaluate_rows
from evaluate_stage_a_visual_floor import (
    binding_selective_metrics,
    select_binding_scene_pairs,
    select_evaluation_rows,
)
from scoring_contract import metric_contract, random_control_assignment


def row(family: str, task: str, candidate: tuple[bool, bool, bool], baseline: tuple[bool, bool, bool]) -> dict:
    def prediction(values: tuple[bool, bool, bool]) -> dict[str, bool]:
        return dict(zip(("base_correct", "edited_correct", "invariant_correct"), values))

    return {
        "scene_family_id": family,
        "task": task,
        "split": "test",
        "methods": {
            "full_196": prediction((True, True, True)),
            "strong_49": prediction(baseline),
            "proposed_49": prediction(candidate),
        },
    }


class CounterfactualEvaluationTests(unittest.TestCase):
    def control_row(self, family: str, task: str, answer: str) -> dict:
        opposite = "no" if answer == "yes" else "yes"
        return {
            "scene_family_id": family,
            "task": task,
            "base_answer": answer,
            "edited_answer": opposite,
            "invariant_answer": answer,
            "base_image_sha256": f"{family}-base",
            "edited_image_sha256": f"{family}-edited",
            "invariant_image_sha256": f"{family}-invariant",
        }

    def test_stage_d_random_control_is_deterministic_and_stratified(self) -> None:
        rows = [
            self.control_row("a", "count", "yes"),
            self.control_row("b", "count", "yes"),
            self.control_row("c", "count", "no"),
            self.control_row("d", "count", "no"),
            self.control_row("e", "shape", "yes"),
            self.control_row("f", "shape", "yes"),
            self.control_row("g", "shape", "no"),
            self.control_row("h", "shape", "no"),
        ]
        first, audit = random_control_assignment(rows, "base", "within_task_answer_matched")
        second, repeated = random_control_assignment(rows, "base", "within_task_answer_matched")
        self.assertEqual(first, second)
        self.assertEqual(audit, repeated)
        self.assertEqual(audit["self_matches"], 0)
        self.assertEqual(audit["same_task_percent"], 100.0)
        self.assertEqual(audit["same_expected_answer_percent"], 100.0)

    def test_binding_random_control_deranges_clusters_and_shared_images(self) -> None:
        rows = []
        for pair_index in range(4):
            for question_index in range(3):
                rows.append({
                    "scene_family_id": f"pair-{pair_index}-q{question_index}",
                    "scene_pair_id": f"pair-{pair_index}",
                    "statistical_cluster_id": f"pair-{pair_index}",
                    "task": "binding_intervention",
                    "base_answer": "yes",
                    "edited_answer": "no",
                    "invariant_answer": "yes",
                    "base_image_sha256": f"pair-{pair_index}-base",
                    "edited_image_sha256": f"pair-{pair_index}-edited",
                    "invariant_image_sha256": f"pair-{pair_index}-invariant",
                })
        donors, audit = random_control_assignment(
            rows, "base", "within_task_answer_cluster_deranged"
        )
        self.assertEqual(audit["self_matches"], 0)
        self.assertEqual(audit["same_task_percent"], 100.0)
        self.assertEqual(audit["same_expected_answer_percent"], 100.0)
        self.assertEqual(audit["same_statistical_cluster_percent"], 0.0)
        self.assertEqual(audit["same_image_hash_percent"], 0.0)
        for recipient, donor in enumerate(donors):
            self.assertNotEqual(
                rows[recipient]["statistical_cluster_id"], rows[donor]["statistical_cluster_id"]
            )

    def test_stage_d_selection_is_exactly_task_and_answer_balanced(self) -> None:
        rows = []
        for task in ("count", "shape"):
            for answer, count in (("no", 7), ("yes", 8)):
                for index in range(count):
                    rows.append({
                        "scene_family_id": f"{index:02d}-{task}-{answer}",
                        "task": task,
                        "base_answer": answer,
                    })
        selected = select_evaluation_rows(rows, 10, balance_base_answer=True)
        for task in ("count", "shape"):
            task_rows = [row for row in selected if row["task"] == task]
            self.assertEqual(len(task_rows), 10)
            self.assertEqual(sum(row["base_answer"] == "no" for row in task_rows), 5)
            self.assertEqual(sum(row["base_answer"] == "yes" for row in task_rows), 5)

    def test_unbalanced_selection_does_not_duplicate_rows(self) -> None:
        rows = [
            {"scene_family_id": f"f-{index}", "task": "binding", "base_answer": "yes"}
            for index in range(8)
        ]
        selected = select_evaluation_rows(rows, 4, balance_base_answer=False)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["scene_family_id"] for row in selected}), 4)

    def test_binding_selection_preserves_complete_scene_pairs(self) -> None:
        rows = []
        for pair_index in range(4):
            for question_index in range(6):
                rows.append({
                    "scene_pair_id": f"pair-{pair_index}",
                    "scene_family_id": f"pair-{pair_index}-q{question_index}",
                    "question_index": question_index,
                })
        selected = select_binding_scene_pairs(list(reversed(rows)), 2)
        self.assertEqual(len(selected), 12)
        self.assertEqual({row["scene_pair_id"] for row in selected}, {"pair-0", "pair-1"})
        for pair_id in {row["scene_pair_id"] for row in selected}:
            self.assertEqual(sum(row["scene_pair_id"] == pair_id for row in selected), 6)

    def test_binding_metrics_keep_affected_rows_primary(self) -> None:
        rows = []
        for pair_index in range(2):
            pair_id = f"pair-{pair_index}"
            for question_index in range(6):
                affected = question_index < 2
                expected = {
                    "base": "yes",
                    "edited": "no" if affected else "yes",
                    "invariant": "yes",
                }
                true_prediction = dict(expected)
                if affected and pair_index == 1:
                    true_prediction["edited"] = "yes"
                rows.append({
                    "scene_family_id": f"{pair_id}-q{question_index}",
                    "scene_pair_id": pair_id,
                    "statistical_cluster_id": pair_id,
                    "question_role": (
                        "affected_positive_to_negative" if affected else "invariant_control_positive"
                    ),
                    "expected": expected,
                    "predictions": {
                        "true_image": true_prediction,
                        "random_image": {"base": "yes", "edited": "yes", "invariant": "yes"},
                        "no_image": {"base": "yes", "edited": "yes", "invariant": "yes"},
                    },
                })
        metrics = binding_selective_metrics(rows)
        assert metrics is not None
        self.assertEqual(metrics["scene_pairs"], 2)
        self.assertEqual(metrics["affected_questions"], 4)
        self.assertEqual(metrics["invariant_control_questions"], 8)
        self.assertEqual(
            metrics["per_condition"]["true_image"]["affected_question_joint_accuracy_percent"],
            50.0,
        )
        self.assertEqual(
            metrics["per_condition"]["true_image"]["invariant_question_joint_accuracy_percent"],
            100.0,
        )
        self.assertEqual(
            metrics["per_condition"]["true_image"]["scene_pair_selective_all_six_correct_percent"],
            50.0,
        )
        self.assertEqual(metrics["affected_true_minus_no_image"]["estimate_pp"], 50.0)

    def test_metric_contract_explains_zero_no_image_family_joint(self) -> None:
        rows = []
        for family, base_answer, prediction in (("a", "yes", "yes"), ("b", "no", "yes")):
            edited_answer = "no" if base_answer == "yes" else "yes"
            rows.append({
                "scene_family_id": family,
                "task": "count",
                "candidate_answers": ["yes", "no"],
                "expected": {
                    "base": base_answer,
                    "edited": edited_answer,
                    "invariant": base_answer,
                },
                "predictions": {
                    "no_image": {
                        "base": prediction,
                        "edited": prediction,
                        "invariant": prediction,
                    }
                },
            })
        metrics = metric_contract(rows)["overall"]["no_image"]
        self.assertEqual(metrics["family_joint_accuracy_percent"], 0.0)
        self.assertEqual(metrics["ordinary_primary_query_accuracy_percent"], 50.0)
        self.assertEqual(metrics["valid_answer_rate_primary_percent"], 100.0)
        self.assertEqual(metrics["base_edited_same_prediction_percent"], 100.0)

    def test_assembly_uses_true_image_predictions_only(self) -> None:
        assembled = correctness({
            "expected": {"base": "yes", "edited": "no", "invariant": "yes"},
            "predictions": {
                "true_image": {"base": "yes", "edited": "yes", "invariant": "yes"},
                "random_image": {"base": "no", "edited": "no", "invariant": "no"},
            },
        })
        self.assertEqual(assembled, {
            "base_correct": True,
            "edited_correct": False,
            "invariant_correct": True,
        })

    def test_primary_pair_metric_and_failure_reduction(self) -> None:
        rows = [
            row("a", "count", (True, True, True), (True, False, True)),
            row("b", "count", (True, True, True), (True, True, True)),
            row("c", "shape", (True, False, True), (False, False, True)),
            row("d", "shape", (True, True, False), (True, False, True)),
        ]
        result = evaluate_rows(
            rows,
            candidate="proposed_49",
            baseline="strong_49",
            full="full_196",
            resamples=200,
        )
        overall = result["overall"]
        self.assertEqual(overall["paired_joint_accuracy_percent"]["proposed_49"], 75.0)
        self.assertEqual(overall["paired_joint_accuracy_percent"]["strong_49"], 25.0)
        self.assertEqual(
            overall["compression_induced_failure"]["candidate_failure_percent"], 25.0
        )
        self.assertEqual(set(result["by_task"]), {"count", "shape"})

    def test_duplicate_family_is_rejected(self) -> None:
        duplicate = row("a", "count", (True, True, True), (True, True, True))
        with self.assertRaises(ValueError):
            evaluate_rows(
                [duplicate, duplicate],
                candidate="proposed_49",
                baseline="strong_49",
                full="full_196",
                resamples=10,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
