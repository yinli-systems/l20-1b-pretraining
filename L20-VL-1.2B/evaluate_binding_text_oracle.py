#!/usr/bin/env python3
"""Measure whether the frozen language path can solve binding from exact scene text."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_stage_a_visual_floor import (
    binding_selective_metrics,
    directory_records,
    score_variant,
    sha256_file,
)
from modeling import freeze
from scoring_contract import VARIANTS
from train_stage_a_full_token import utc_now, write_json_atomic


def variant_states(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base = deepcopy(row["scene_state"])
    edited = deepcopy(base)
    objects = {item["id"]: item for item in edited["objects"]}
    objects["target"]["color"], objects["partner"]["color"] = (
        objects["partner"]["color"],
        objects["target"]["color"],
    )
    invariant = deepcopy(base)
    invariant["background"] = row["invariant_edit"]["after"]
    return {"base": base, "edited": edited, "invariant": invariant}


def describe_scene(state: dict[str, Any]) -> str:
    ordered = sorted(state["objects"], key=lambda item: (item["x"], item["y"]))
    records = "; ".join(
        f"object {index}: color={item['color']}, shape={item['shape']}, "
        f"x={item['x']}, y={item['y']}"
        for index, item in enumerate(ordered, 1)
    )
    return (
        "Exact scene records (smaller x means farther left): "
        f"{records}."
    )


def oracle_prompt(row: dict[str, Any], variant: str) -> str:
    state = variant_states(row)[variant]
    return f"{describe_scene(state)}\nQuestion: {row['question']} Answer yes or no."


def verify_protocol(protocol: dict[str, Any], source_path: Path) -> None:
    if protocol.get("status") != "authorized_binding_text_oracle_diagnostic_v1":
        raise SystemExit("text-oracle protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    source_code = protocol["source_code"]
    if sha256_file(source_path) != source_code["evaluator_sha256"]:
        raise SystemExit("text-oracle evaluator hash mismatch")
    root = source_path.parent
    for filename, key in (
        ("evaluate_stage_a_visual_floor.py", "visual_evaluator_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("scoring_contract.py", "scoring_contract_sha256"),
        ("train_stage_a_full_token.py", "stage_a_utility_sha256"),
    ):
        if sha256_file(root / filename) != source_code[key]:
            raise SystemExit(f"text-oracle dependency hash mismatch: {filename}")
    prerequisite = protocol["prerequisite"]
    summary_path = Path(prerequisite["representation_probe"])
    if sha256_file(summary_path) != prerequisite["representation_probe_sha256"]:
        raise SystemExit("representation-probe prerequisite hash mismatch")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != prerequisite["required_status"]:
        raise SystemExit("representation-probe prerequisite status mismatch")
    if summary.get("diagnosis", {}).get("localization") != prerequisite["required_localization"]:
        raise SystemExit("representation-probe localization mismatch")
    if summary.get("final_test_used") is not False:
        raise SystemExit("representation probe used final test")


def evaluate_arm(
    arm_name: str,
    arm: dict[str, Any],
    rows: list[dict[str, Any]],
    language_path: Path,
    tokenizer,
    batch_families: int,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.monotonic()
    language = AutoModelForCausalLM.from_pretrained(
        language_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).cuda()
    adapter_records = None
    if arm.get("language_adapter"):
        from peft import PeftModel

        adapter_path = Path(arm["language_adapter"])
        adapter_records = directory_records(adapter_path)
        if {
            name: record["sha256"] for name, record in adapter_records.items()
        } != arm["language_adapter_files"]:
            raise SystemExit(f"language adapter hash mismatch: {arm_name}")
        language = PeftModel.from_pretrained(
            language,
            adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
    freeze(language)
    predictions: dict[str, list[str]] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        text_rows = [
            {**row, "question": oracle_prompt(row, variant)} for row in rows
        ]
        predictions[variant], diagnostics[variant] = score_variant(
            text_rows,
            None,
            language,
            None,
            None,
            tokenizer,
            None,
            batch_families,
            max_tokens,
            -1,
            "manifest",
            torch.bfloat16,
        )
        print(
            f"TEXT_ORACLE_PROGRESS arm={arm_name} variant={variant} rows={len(rows)} "
            f"elapsed_seconds={time.monotonic() - started:.1f}",
            flush=True,
        )
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append({
            "scene_family_id": row["scene_family_id"],
            "scene_pair_id": row["scene_pair_id"],
            "statistical_cluster_id": row["statistical_cluster_id"],
            "task": row["task"],
            "challenge": row["challenge"],
            "question_role": row["question_role"],
            "split": row["split"],
            "candidate_answers": row["candidate_answers"],
            "expected": {
                variant: row[f"{variant}_answer"] for variant in VARIANTS
            },
            "predictions": {
                "text_oracle": {
                    variant: predictions[variant][index] for variant in VARIANTS
                }
            },
            "candidate_score_diagnostics": {
                "text_oracle": {
                    variant: diagnostics[variant][index] for variant in VARIANTS
                }
            },
        })
    metrics = binding_selective_metrics(prediction_rows)
    del language
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "prediction_rows": prediction_rows,
        "language_adapter_records": adapter_records,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    verify_protocol(protocol, Path(__file__))
    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("text-oracle manifest hash mismatch")
    for evidence in protocol["data"]["evidence"].values():
        if sha256_file(Path(evidence["path"])) != evidence["sha256"]:
            raise SystemExit(f"text-oracle data evidence hash mismatch: {evidence['path']}")
    all_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    rows = sorted(
        [row for row in all_rows if row["split"] == protocol["data"]["split"]],
        key=lambda row: (row["scene_pair_id"], row["question_index"]),
    )
    if {row["split"] for row in all_rows} != {"train", "mechanism_dev", "selection_dev"}:
        raise SystemExit("text-oracle partition set mismatch")
    if len(rows) != protocol["data"]["rows"]:
        raise SystemExit("text-oracle row count mismatch")
    language_path = Path(protocol["parents"]["language"])
    tokenizer = AutoTokenizer.from_pretrained(language_path, local_files_only=True)
    started = time.monotonic()
    arms = {
        name: evaluate_arm(
            name,
            arm,
            rows,
            language_path,
            tokenizer,
            protocol["evaluation"]["batch_families"],
            protocol["evaluation"]["max_tokens"],
        )
        for name, arm in protocol["arms"].items()
    }
    write_json_atomic(args.output, {
        "schema_version": "2026-09-14-v1",
        "status": "complete_diagnostic_only",
        "completed_at": utc_now(),
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": sha256_file(manifest),
        "scene_pairs": len({row["scene_pair_id"] for row in rows}),
        "rows": len(rows),
        "arms": arms,
        "total_elapsed_seconds": time.monotonic() - started,
        "final_test_used": False,
        "training_prediction_tokens": 0,
        "claim_boundary": (
            "Exact symbolic scene records provide privileged development-only input. "
            "Success is a language/readout upper bound, not multimodal model performance; "
            "failure can also reflect the chosen serialization or prompting format."
        ),
    })


if __name__ == "__main__":
    main()
