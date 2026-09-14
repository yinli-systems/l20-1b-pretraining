#!/usr/bin/env python3
"""Validate and summarize all one-time Stage-D IID/OOD confirmation results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Callable

import numpy as np

from stage_d_confirmation_contract import (
    ARMS,
    CONDITIONS,
    SPLITS,
    crossed_bootstrap_interval,
    family_joint,
    ordinary_primary,
    paired_t_interval,
    sha256_file,
    stable_id_sha256,
)
from train_stage_a_full_token import utc_now, write_json_atomic
from validate_stage_d_confirmation_protocol import validate_protocol


ROOT = Path(__file__).resolve().parent


def evaluation_key(item: dict) -> str:
    return f"{item['arm']}-seed{item['seed']}-{item['split']}"


def evaluation_path(item: dict) -> Path:
    return ROOT / "evidence" / f"stage-d-confirmation-{evaluation_key(item)}-v1.json"


def load_manifest(protocol: dict) -> tuple[dict[str, dict], dict[str, list[str]]]:
    rows = [
        json.loads(line)
        for line in Path(protocol["manifest"]["path"]).read_text().splitlines()
        if line
    ]
    by_id = {row["scene_family_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("confirmation manifest contains duplicate family IDs")
    split_ids = {
        split: sorted(row["scene_family_id"] for row in rows if row["split"] == split)
        for split in SPLITS
    }
    return by_id, split_ids


def load_confirmation_result(
    protocol: dict,
    item: dict,
    manifest_by_id: dict[str, dict],
    expected_ids: list[str],
) -> dict:
    path = evaluation_path(item)
    result = json.loads(path.read_text())
    arm, seed, split = item["arm"], int(item["seed"]), item["split"]
    checkpoint = protocol["checkpoints"][arm][str(seed)]
    expected_protocol_sha = protocol["source_files"]["replication_protocol"]["sha256"]
    checks = {
        "split": result.get("split") == split,
        "arm": result.get("protocol_arm") == arm,
        "families": result.get("scene_families") == protocol["splits"][split]["scene_families"],
        "manifest": result.get("manifest_sha256") == protocol["manifest"]["sha256"],
        "checkpoint": result.get("checkpoint_sha256") == checkpoint["bridge_sha256"],
        "replication_protocol": result.get("protocol_sha256") == expected_protocol_sha,
        "conditions": result.get("conditions") == list(CONDITIONS),
        "candidate_order": result.get("candidate_order") == "manifest",
        "precision": result.get("scoring_precision") == "float32",
        "tf32": result.get("tf32_enabled") is False,
        "image_hashes": result.get("image_integrity_audit", {}).get("all_match_manifest") is True,
        "image_hash_count": result.get("image_integrity_audit", {}).get("checked_images")
        == 3 * protocol["splits"][split]["scene_families"],
        "candidate_spans": result.get("candidate_answer_span_audit", {}).get("all_nonempty") is True,
        "full_split": result.get("selection", {}).get("families_per_task") is None
        and result.get("selection", {}).get("purpose") == "full_split_gate",
        "random_policy": result.get("random_control", {}).get("policy")
        == protocol["random_control"],
        "training_tokens": result.get("training_prediction_tokens") == 0,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"{evaluation_key(item)} result validation failed: {failed}")
    adapter_files = result.get("language_adapter", {}).get("files", {})
    if adapter_files.get("adapter_model.safetensors", {}).get("sha256") != checkpoint["adapter_model_sha256"]:
        raise RuntimeError(f"{evaluation_key(item)} adapter model hash mismatch")
    if adapter_files.get("adapter_config.json", {}).get("sha256") != checkpoint["adapter_config_sha256"]:
        raise RuntimeError(f"{evaluation_key(item)} adapter config hash mismatch")
    for variant, audit in result.get("random_control", {}).get("audit", {}).items():
        if variant not in ("base", "edited", "invariant"):
            raise RuntimeError(f"{evaluation_key(item)} unexpected random-control variant")
        if audit.get("self_matches") != 0 or audit.get("same_task_percent") != 100.0 or audit.get("same_expected_answer_percent") != 100.0:
            raise RuntimeError(f"{evaluation_key(item)} random-control audit failed")
    predictions = result.get("predictions", [])
    prediction_ids = [row["scene_family_id"] for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)) or sorted(prediction_ids) != expected_ids:
        raise RuntimeError(f"{evaluation_key(item)} prediction family set mismatch")
    compact = {}
    for row in predictions:
        family_id = row["scene_family_id"]
        manifest = manifest_by_id[family_id]
        expected = {
            "base": manifest["base_answer"],
            "edited": manifest["edited_answer"],
            "invariant": manifest["invariant_answer"],
        }
        if row["expected"] != expected or row["task"] != manifest["task"] or row["split"] != split:
            raise RuntimeError(f"{evaluation_key(item)} expected-label or stratum mismatch")
        compact[family_id] = {
            "expected": row["expected"],
            "predictions": row["predictions"],
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "arm": arm,
        "seed": seed,
        "split": split,
        "wall_seconds": float(result["total_wall_seconds"]),
        "metrics": result["metric_contract"],
        "visual_floor": result.get("visual_floor"),
        "predictions": compact,
    }


def matrix_for(
    results: dict[tuple[str, int, str], dict],
    *,
    arm: str,
    seeds: list[int],
    split: str,
    family_ids: list[str],
    metric: Callable[[dict, str], float],
    condition: str,
) -> np.ndarray:
    return np.asarray([
        [metric(results[(arm, seed, split)]["predictions"][family_id], condition) for family_id in family_ids]
        for seed in seeds
    ], dtype=np.float64)


def contrast_summary(
    results: dict[tuple[str, int, str], dict],
    *,
    seeds: list[int],
    split: str,
    family_ids: list[str],
    metric: Callable[[dict, str], float],
    condition: str,
    resamples: int,
    bootstrap_seed: int,
) -> dict:
    matrices = {
        arm: matrix_for(
            results, arm=arm, seeds=seeds, split=split, family_ids=family_ids,
            metric=metric, condition=condition,
        )
        for arm in ARMS
    }
    differences = matrices["A_spatial_49"] - matrices["F_matched_196"]
    per_seed = [100.0 * float(row.mean()) for row in differences]
    return {
        "scene_families": len(family_ids),
        "condition": condition,
        "metric": metric.__name__,
        "arm_mean_percent": {arm: 100.0 * float(matrix.mean()) for arm, matrix in matrices.items()},
        "per_seed_A_minus_F_pp": [
            {"seed": seed, "delta_pp": delta}
            for seed, delta in zip(seeds, per_seed)
        ],
        "paired_t_interval": paired_t_interval(per_seed),
        "crossed_bootstrap_interval": crossed_bootstrap_interval(
            differences, resamples=resamples, seed=bootstrap_seed
        ),
    }


def visual_control_summary(
    results: dict[tuple[str, int, str], dict],
    *,
    arm: str,
    seeds: list[int],
    split: str,
    family_ids: list[str],
    control: str,
    resamples: int,
    bootstrap_seed: int,
) -> dict:
    true_matrix = matrix_for(
        results, arm=arm, seeds=seeds, split=split, family_ids=family_ids,
        metric=family_joint, condition="true_image",
    )
    control_matrix = matrix_for(
        results, arm=arm, seeds=seeds, split=split, family_ids=family_ids,
        metric=family_joint, condition=control,
    )
    differences = true_matrix - control_matrix
    per_seed = [100.0 * float(row.mean()) for row in differences]
    return {
        "arm": arm,
        "contrast": f"true_image_minus_{control}",
        "per_seed_pp": [{"seed": seed, "delta_pp": value} for seed, value in zip(seeds, per_seed)],
        "paired_t_interval": paired_t_interval(per_seed),
        "crossed_bootstrap_interval": crossed_bootstrap_interval(
            differences, resamples=resamples, seed=bootstrap_seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    validation = validate_protocol(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    manifest_by_id, split_ids = load_manifest(protocol)
    results = {}
    result_records = []
    for item in protocol["execution_order"]:
        compact = load_confirmation_result(
            protocol, item, manifest_by_id, split_ids[item["split"]]
        )
        key = (item["arm"], int(item["seed"]), item["split"])
        if key in results:
            raise RuntimeError(f"duplicate confirmation result: {key}")
        results[key] = compact
        result_records.append({key: value for key, value in compact.items() if key != "predictions"})

    seeds = protocol["seeds"]
    statistics = protocol["statistics"]
    resamples = int(statistics["crossed_bootstrap_resamples"])
    base_seed = int(statistics["crossed_bootstrap_seed"])
    split_summaries = {}
    for split_index, split in enumerate(SPLITS):
        ids = split_ids[split]
        primary = contrast_summary(
            results, seeds=seeds, split=split, family_ids=ids,
            metric=family_joint, condition="true_image", resamples=resamples,
            bootstrap_seed=base_seed + split_index * 100,
        )
        ordinary = contrast_summary(
            results, seeds=seeds, split=split, family_ids=ids,
            metric=ordinary_primary, condition="true_image", resamples=resamples,
            bootstrap_seed=base_seed + split_index * 100 + 1,
        )
        tasks = {}
        for task_index, task in enumerate(sorted({manifest_by_id[x]["task"] for x in ids})):
            task_ids = [x for x in ids if manifest_by_id[x]["task"] == task]
            tasks[task] = contrast_summary(
                results, seeds=seeds, split=split, family_ids=task_ids,
                metric=family_joint, condition="true_image", resamples=resamples,
                bootstrap_seed=base_seed + split_index * 100 + 10 + task_index,
            )
        challenges = {}
        if split == "ood_binding":
            for challenge_index, challenge in enumerate(sorted({manifest_by_id[x]["challenge"] for x in ids})):
                challenge_ids = [x for x in ids if manifest_by_id[x]["challenge"] == challenge]
                challenges[challenge] = contrast_summary(
                    results, seeds=seeds, split=split, family_ids=challenge_ids,
                    metric=family_joint, condition="true_image", resamples=resamples,
                    bootstrap_seed=base_seed + 200 + challenge_index,
                )
        controls = {
            arm: {
                control: visual_control_summary(
                    results, arm=arm, seeds=seeds, split=split, family_ids=ids,
                    control=control, resamples=resamples,
                    bootstrap_seed=base_seed + split_index * 100 + 30 + arm_index * 2 + control_index,
                )
                for control_index, control in enumerate(("random_image", "no_image"))
            }
            for arm_index, arm in enumerate(ARMS)
        }
        split_summaries[split] = {
            "primary_family_joint": primary,
            "ordinary_primary_query": ordinary,
            "by_task_family_joint": tasks,
            "by_challenge_family_joint": challenges,
            "visual_controls_family_joint": controls,
        }

    primary = split_summaries[statistics["primary_split"]]["primary_family_joint"]
    crossed = primary["crossed_bootstrap_interval"]
    paired_t = primary["paired_t_interval"]
    margin = float(statistics["noninferiority_margin_pp"])
    superiority = (
        crossed["estimate_pp"] >= float(statistics["superiority_point_threshold_pp"])
        and crossed["lower_95_ci_pp"] > 0.0
        and paired_t["lower_95_ci_pp"] > 0.0
    )
    noninferiority = crossed["lower_95_ci_pp"] > margin and paired_t["lower_95_ci_pp"] > margin
    summary = {
        "schema_version": "2026-09-13-v1",
        "status": "complete_confirmatory_iid_and_secondary_ood_results_frozen",
        "completed_at": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_validation": validation,
        "results": result_records,
        "split_summaries": split_summaries,
        "primary_decision": {
            "comparison": statistics["primary_comparison"],
            "split": statistics["primary_split"],
            "superiority_pass": superiority,
            "noninferiority_margin_pp": margin,
            "noninferiority_pass": noninferiority,
            "decision_rule": statistics["primary_inference_rule"],
        },
        "evaluation_gpu_hours": sum(record["wall_seconds"] for record in result_records) / 3600.0,
        "expected_evaluations": protocol["expected_evaluations"],
        "completed_evaluations": len(result_records),
        "all_visual_floor_gates_pass": all(
            record["visual_floor"] is not None and record["visual_floor"].get("passes_all") is True
            for record in result_records
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    if summary["evaluation_gpu_hours"] > protocol["budget"]["confirmation_l20_gpu_hours_cap"]:
        raise RuntimeError("completed confirmation exceeds the frozen GPU budget")
    write_json_atomic(output, summary)
    print(json.dumps({
        "status": summary["status"],
        "output": str(output),
        "sha256": sha256_file(output),
        "primary": primary,
        "primary_decision": summary["primary_decision"],
        "evaluation_gpu_hours": summary["evaluation_gpu_hours"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
