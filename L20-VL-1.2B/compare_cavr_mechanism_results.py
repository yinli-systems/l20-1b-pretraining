#!/usr/bin/env python3
"""Paired scene-cluster comparison of CAVR and query-CE mechanism evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from counterfactual_losses import paired_cluster_bootstrap
from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


MECHANISM_KEYS = (
    "address_all_variants",
    "normal_base_edit_joint",
    "swapped_base_edit_joint",
    "invariant_swapped_correct",
    "causal_six_way",
)


def paired_mechanism_difference(
    cavr: dict[str, Any], query: dict[str, Any], key: str
) -> dict[str, Any]:
    cavr_records = {
        row["scene_family_id"]: row for row in cavr["records"] if row["affected"]
    }
    query_records = {
        row["scene_family_id"]: row for row in query["records"] if row["affected"]
    }
    if set(cavr_records) != set(query_records) or not cavr_records:
        raise ValueError("mechanism result families do not match")
    family_ids = sorted(cavr_records)
    if any(
        cavr_records[family_id]["scene_pair_id"]
        != query_records[family_id]["scene_pair_id"]
        for family_id in family_ids
    ):
        raise ValueError("mechanism result clusters do not match")
    differences = [
        100.0 * (
            float(cavr_records[family_id][key])
            - float(query_records[family_id][key])
        )
        for family_id in family_ids
    ]
    clusters = [cavr_records[family_id]["scene_pair_id"] for family_id in family_ids]
    interval = paired_cluster_bootstrap(
        differences, clusters, resamples=10_000, seed=20260914
    )
    return {
        "estimate_pp": interval.estimate,
        "lower_95_ci_pp": interval.lower,
        "upper_95_ci_pp": interval.upper,
        "scene_pair_clusters": interval.clusters,
        "question_families": interval.samples,
        "bootstrap_resamples": interval.resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cavr", type=Path, required=True)
    parser.add_argument("--query-ce", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cavr = json.loads(args.cavr.read_text())
    query = json.loads(args.query_ce.read_text())
    for name, result in (("cavr", cavr), ("query_ce", query)):
        if result.get("status") != "complete_development_only":
            raise SystemExit(f"{name} mechanism result is incomplete")
        if result.get("final_test_used") is not False:
            raise SystemExit(f"{name} mechanism result used final test")
    if cavr["split"] != query["split"]:
        raise SystemExit("mechanism result splits do not match")
    comparisons = {
        key: paired_mechanism_difference(cavr, query, key)
        for key in MECHANISM_KEYS
    }
    result = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_development_only",
        "split": cavr["split"],
        "cavr_step": cavr["step"],
        "query_ce_step": query["step"],
        "cavr_result_sha256": sha256_file(args.cavr),
        "query_ce_result_sha256": sha256_file(args.query_ce),
        "comparator_sha256": sha256_file(Path(__file__).resolve()),
        "cavr_minus_query_ce": comparisons,
        "completed_at": utc_now(),
        "final_test_used": False,
        "claim_boundary": "Paired synthetic development comparison clustered by scene pair; not final-test, multi-seed, or real-image evidence.",
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
