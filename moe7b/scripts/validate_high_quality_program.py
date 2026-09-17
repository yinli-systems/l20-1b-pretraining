#!/usr/bin/env python3
"""Validate the frozen arithmetic and fail-closed gates of the 5.1T programme."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MIX_KEYS = {
    "general_filtered_web",
    "permissive_educational_code",
    "mathematics_reasoning",
    "multilingual_filtered_web",
    "science_reference",
}


def validate(plan: dict) -> list[str]:
    errors: list[str] = []
    accounting = plan["token_accounting"]
    base = int(accounting["formal_base_pretraining_tokens"])
    anneal = int(accounting["formal_high_quality_anneal_tokens"])
    total = int(accounting["formal_total_prediction_tokens"])
    if base + anneal != total:
        errors.append("base plus anneal tokens do not equal the formal total")
    if total != 5_100_000_000_000:
        errors.append("the formal programme must remain exactly 5.1T tokens")
    checkpoints = [int(value) for value in plan["checkpoint_ladder_prediction_tokens"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != total:
        errors.append("checkpoint ladder must be unique, increasing, and end at 5.1T")
    for name, mix in plan["data_mix_selection"]["candidate_percentages"].items():
        if set(mix) != EXPECTED_MIX_KEYS:
            errors.append(f"{name} data categories drifted")
        if sum(int(value) for value in mix.values()) != 100:
            errors.append(f"{name} percentages do not sum to 100")
    anneal_mix = plan["high_quality_anneal"]["candidate_percentages"]
    if sum(int(value) for value in anneal_mix.values()) != 100:
        errors.append("anneal percentages do not sum to 100")
    systems = plan["systems_gates"]
    if float(systems["bounded_screen_minimum_median_causal_useful_mfu"]) < 0.50:
        errors.append("MFU gate was weakened below 50%")
    if int(systems["constant_prediction_tokens_per_optimizer_step"]) != 131_072:
        errors.append("tokens per optimizer step drifted")
    if float(plan["quality_gates"]["minimum_primary_seven_task_macro"]) < 0.50:
        errors.append("primary seven-task gate was weakened below 50%")
    if plan["status"] == "TRAINING_ADMITTED":
        errors.append("research plan cannot self-assert data admission")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    arguments = parser.parse_args()
    plan = json.loads(arguments.plan.read_text())
    errors = validate(plan)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "formal_total_prediction_tokens": plan["token_accounting"][
            "formal_total_prediction_tokens"
        ],
        "checkpoint_count": len(plan["checkpoint_ladder_prediction_tokens"]),
    }
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
