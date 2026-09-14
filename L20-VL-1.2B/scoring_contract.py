"""Pure scoring and random-control utilities for controlled VLM evaluation.

The functions in this module deliberately avoid model dependencies so the
metric contract and control assignment can be unit-tested on CPU.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Iterable


VARIANTS = ("base", "edited", "invariant")
PRIMARY_VARIANTS = ("base", "edited")
RANDOM_CONTROL_POLICIES = (
    "global_shift",
    "within_task_answer_matched",
    "within_task_answer_cluster_deranged",
)


def _percent(numerator: int | float, denominator: int) -> float:
    if denominator < 1:
        raise ValueError("metric denominator must be positive")
    return 100.0 * float(numerator) / denominator


def random_control_assignment(
    rows: list[dict[str, Any]], variant: str, policy: str
) -> tuple[list[int], dict[str, Any]]:
    """Return one deterministic donor index per recipient and an audit summary.

    ``global_shift`` exactly preserves the legacy evaluator. The Stage-D policy
    rotates donors inside task/expected-answer strata. It removes exact-family
    correspondence without introducing cross-task rendering shifts or a forced
    answer-label anticorrelation.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if policy not in RANDOM_CONTROL_POLICIES:
        raise ValueError(f"unsupported random-control policy: {policy}")
    if len(rows) < 2:
        raise ValueError("at least two rows are required for a random control")

    if policy == "global_shift":
        donors = list(range(1, len(rows))) + [0]
    elif policy == "within_task_answer_matched":
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        answer_key = f"{variant}_answer"
        for index, row in enumerate(rows):
            groups[(row["task"], row[answer_key])].append(index)
        undersized = {stratum: len(indices) for stratum, indices in groups.items() if len(indices) < 2}
        if undersized:
            raise ValueError(f"random-control strata require at least two rows: {undersized}")
        donors = [-1] * len(rows)
        for indices in groups.values():
            ordered = sorted(indices, key=lambda index: rows[index]["scene_family_id"])
            rotated = ordered[1:] + ordered[:1]
            for recipient, donor in zip(ordered, rotated):
                donors[recipient] = donor
    else:
        groups = defaultdict(list)
        answer_key = f"{variant}_answer"
        for index, row in enumerate(rows):
            groups[(row["task"], row[answer_key])].append(index)
        undersized = {stratum: len(indices) for stratum, indices in groups.items() if len(indices) < 2}
        if undersized:
            raise ValueError(f"random-control strata require at least two rows: {undersized}")
        donors = [-1] * len(rows)

        def cluster_id(row: dict[str, Any]) -> str:
            return row.get("statistical_cluster_id", row.get("scene_pair_id", row["scene_family_id"]))

        for stratum, indices in groups.items():
            ordered = sorted(indices, key=lambda index: rows[index]["scene_family_id"])
            selected_rotation = None
            for shift in range(1, len(ordered)):
                rotated = ordered[shift:] + ordered[:shift]
                if all(
                    cluster_id(rows[recipient]) != cluster_id(rows[donor])
                    and rows[recipient][f"{variant}_image_sha256"]
                    != rows[donor][f"{variant}_image_sha256"]
                    for recipient, donor in zip(ordered, rotated)
                ):
                    selected_rotation = rotated
                    break
            if selected_rotation is None:
                raise ValueError(
                    f"cannot derange random control across clusters and images: {stratum}"
                )
            for recipient, donor in zip(ordered, selected_rotation):
                donors[recipient] = donor

    answer_key = f"{variant}_answer"
    records = []
    for recipient, donor in enumerate(donors):
        source = rows[recipient]
        replacement = rows[donor]
        records.append({
            "recipient_family": source["scene_family_id"],
            "donor_family": replacement["scene_family_id"],
            "recipient_task": source["task"],
            "donor_task": replacement["task"],
            "recipient_expected_answer": source[answer_key],
            "donor_expected_answer": replacement[answer_key],
            "donor_image_sha256": replacement[f"{variant}_image_sha256"],
        })
    encoded = "\n".join(json.dumps(record, sort_keys=True) for record in records).encode()
    audit = {
        "policy": policy,
        "variant": variant,
        "families": len(rows),
        "self_matches": sum(recipient == donor for recipient, donor in enumerate(donors)),
        "same_task_percent": _percent(
            sum(rows[index]["task"] == rows[donor]["task"] for index, donor in enumerate(donors)),
            len(rows),
        ),
        "same_expected_answer_percent": _percent(
            sum(rows[index][answer_key] == rows[donor][answer_key] for index, donor in enumerate(donors)),
            len(rows),
        ),
        "same_statistical_cluster_percent": _percent(
            sum(
                rows[index].get("statistical_cluster_id", rows[index].get("scene_pair_id", rows[index]["scene_family_id"]))
                == rows[donor].get("statistical_cluster_id", rows[donor].get("scene_pair_id", rows[donor]["scene_family_id"]))
                for index, donor in enumerate(donors)
            ),
            len(rows),
        ),
        "same_image_hash_percent": _percent(
            sum(
                rows[index][f"{variant}_image_sha256"]
                == rows[donor][f"{variant}_image_sha256"]
                for index, donor in enumerate(donors)
            ),
            len(rows),
        ),
        "mapping_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return donors, audit


def metric_contract(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute every Stage-D metric from per-family predictions.

    Primary family-joint accuracy requires both base and edited answers to be
    correct. Ordinary accuracy is also exposed over the two primary queries and
    over all three variants. Validity is candidate-membership, not a free-form
    parser success rate, because this evaluator uses candidate scoring.
    """
    if not prediction_rows:
        raise ValueError("prediction rows cannot be empty")
    family_ids = [row["scene_family_id"] for row in prediction_rows]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("scene families must be unique")
    conditions = set(prediction_rows[0]["predictions"])
    if not conditions:
        raise ValueError("at least one prediction condition is required")
    if any(set(row["predictions"]) != conditions for row in prediction_rows):
        raise ValueError("all rows must expose the same conditions")

    def summarize(rows: Iterable[dict[str, Any]], condition: str) -> dict[str, Any]:
        selected = list(rows)
        correct = {
            variant: [
                row["predictions"][condition][variant] == row["expected"][variant]
                for row in selected
            ]
            for variant in VARIANTS
        }
        candidates = [set(row.get("candidate_answers", ("yes", "no"))) for row in selected]
        valid = {
            variant: [
                row["predictions"][condition][variant] in candidates[index]
                for index, row in enumerate(selected)
            ]
            for variant in VARIANTS
        }
        family_joint = [
            base_ok and edited_ok for base_ok, edited_ok in zip(correct["base"], correct["edited"])
        ]
        base_invariant_joint = [
            base_ok and invariant_ok
            for base_ok, invariant_ok in zip(correct["base"], correct["invariant"])
        ]
        primary_correct = sum(sum(correct[variant]) for variant in PRIMARY_VARIANTS)
        all_correct = sum(sum(correct[variant]) for variant in VARIANTS)
        primary_valid = sum(sum(valid[variant]) for variant in PRIMARY_VARIANTS)
        all_valid = sum(sum(valid[variant]) for variant in VARIANTS)
        distributions = {}
        for variant in VARIANTS:
            counts = Counter(row["predictions"][condition][variant] for row in selected)
            distributions[variant] = {
                answer: {"count": count, "percent": _percent(count, len(selected))}
                for answer, count in sorted(counts.items())
            }
        same_counterfactual_prediction = sum(
            row["predictions"][condition]["base"]
            == row["predictions"][condition]["edited"]
            for row in selected
        )
        return {
            "families": len(selected),
            "family_joint_accuracy_percent": _percent(sum(family_joint), len(selected)),
            "base_plus_invariant_joint_accuracy_percent": _percent(
                sum(base_invariant_joint), len(selected)
            ),
            "ordinary_primary_query_accuracy_percent": _percent(
                primary_correct, len(selected) * len(PRIMARY_VARIANTS)
            ),
            "ordinary_all_variant_accuracy_percent": _percent(
                all_correct, len(selected) * len(VARIANTS)
            ),
            "per_variant_accuracy_percent": {
                variant: _percent(sum(correct[variant]), len(selected)) for variant in VARIANTS
            },
            "valid_answer_rate_primary_percent": _percent(
                primary_valid, len(selected) * len(PRIMARY_VARIANTS)
            ),
            "valid_answer_rate_all_variants_percent": _percent(
                all_valid, len(selected) * len(VARIANTS)
            ),
            "base_edited_same_prediction_percent": _percent(
                same_counterfactual_prediction, len(selected)
            ),
            "prediction_distribution": distributions,
        }

    overall = {condition: summarize(prediction_rows, condition) for condition in sorted(conditions)}
    tasks = sorted({row["task"] for row in prediction_rows})
    by_task = {
        task: {
            condition: summarize(
                [row for row in prediction_rows if row["task"] == task], condition
            )
            for condition in sorted(conditions)
        }
        for task in tasks
    }
    expected_distribution = {
        variant: dict(sorted(Counter(row["expected"][variant] for row in prediction_rows).items()))
        for variant in VARIANTS
    }
    return {
        "schema_version": "2026-09-13-v1",
        "primary_unit": "scene_family",
        "primary_metric": "base_and_edited_family_joint_accuracy",
        "ordinary_query_denominator": "base_and_edited_queries_counted_individually",
        "candidate_scoring_validity_definition": "prediction_is_one_of_manifest_candidate_answers",
        "expected_answer_distribution": expected_distribution,
        "overall": overall,
        "by_task": by_task,
        "no_image_interpretation": (
            "When base and edited prompts are identical but their labels are opposites, a deterministic "
            "question-only scorer that emits the same answer for both has zero family-joint accuracy even "
            "when its ordinary per-query accuracy is nonzero."
        ),
    }
