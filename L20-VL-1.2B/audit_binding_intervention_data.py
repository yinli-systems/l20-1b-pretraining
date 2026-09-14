#!/usr/bin/env python3
"""Independently verify binding-intervention manifests, labels, and renders."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image

from generate_binding_intervention_data import (
    CHALLENGES,
    QUESTION_ROLES,
    VARIANTS,
    _histogram_l1,
    _rgb_histogram,
    evaluate_query,
)
from generate_counterfactual_diagnostic import render_scene, sha256_file


ROOT = Path(__file__).resolve().parent


def _reconstruct_states(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base = deepcopy(row["scene_state"])
    edited = deepcopy(base)
    by_id = {item["id"]: item for item in edited["objects"]}
    left, right = row["relevant_edit"]["target_ids"]
    by_id[left]["color"], by_id[right]["color"] = by_id[right]["color"], by_id[left]["color"]
    invariant = deepcopy(base)
    invariant["background"] = row["invariant_edit"]["after"]
    return {"base": base, "edited": edited, "invariant": invariant}


def audit(protocol_path: Path, generation_receipt_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    protocol = json.loads(protocol_path.read_text())
    receipt = json.loads(generation_receipt_path.read_text())
    if receipt.get("status") != "complete_pending_human_audit":
        raise RuntimeError("generation receipt is not complete and pending audit")
    if receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("protocol hash mismatch")
    if receipt.get("generator_sha256") != sha256_file(ROOT / "generate_binding_intervention_data.py"):
        raise RuntimeError("generator hash mismatch")
    manifest = Path(receipt["manifest_path"])
    if sha256_file(manifest) != receipt.get("manifest_sha256"):
        raise RuntimeError("manifest hash mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if len(rows) != receipt.get("question_families"):
        raise RuntimeError("question-family count mismatch")
    if len({row["scene_family_id"] for row in rows}) != len(rows):
        raise RuntimeError("scene_family_id values must be unique per question")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["scene_pair_id"]].append(row)
    if len(by_pair) != receipt.get("scene_pairs"):
        raise RuntimeError("scene-pair count mismatch")

    image_hash_to_split = {}
    role_answer_counts = Counter()
    challenge_histogram_counts = Counter()
    verified_images = 0
    rerendered_images = 0
    with tempfile.TemporaryDirectory(prefix="binding-audit-") as temporary:
        temporary_root = Path(temporary)
        for pair_index, (pair_id, family) in enumerate(sorted(by_pair.items()), 1):
            if len(family) != len(QUESTION_ROLES):
                raise RuntimeError(f"{pair_id}: expected {len(QUESTION_ROLES)} questions")
            family.sort(key=lambda row: row["question_index"])
            if [row["question_index"] for row in family] != list(range(len(QUESTION_ROLES))):
                raise RuntimeError(f"{pair_id}: question indices are not exact")
            if {row["question_role"] for row in family} != set(QUESTION_ROLES):
                raise RuntimeError(f"{pair_id}: question roles are incomplete")
            constant_fields = (
                "split",
                "challenge",
                "distribution",
                "base_image_path",
                "edited_image_path",
                "invariant_image_path",
                "base_image_sha256",
                "edited_image_sha256",
                "invariant_image_sha256",
                "render_seed",
            )
            for field in constant_fields:
                if len({json.dumps(row[field], sort_keys=True) for row in family}) != 1:
                    raise RuntimeError(f"{pair_id}: {field} differs within a scene pair")
            first = family[0]
            if first["challenge"] not in CHALLENGES:
                raise RuntimeError(f"{pair_id}: unknown challenge")
            if any(row["statistical_cluster_id"] != pair_id for row in family):
                raise RuntimeError(f"{pair_id}: incorrect statistical cluster")

            states = _reconstruct_states(first)
            for variant in VARIANTS:
                declared_path = Path(first[f"{variant}_image_path"])
                if not declared_path.is_file():
                    raise RuntimeError(f"{pair_id}: missing {variant} image")
                with Image.open(declared_path) as image:
                    if image.size != (256, 256) or image.mode != "RGB":
                        raise RuntimeError(f"{pair_id}: invalid {variant} image format")
                digest = sha256_file(declared_path)
                if digest != first[f"{variant}_image_sha256"]:
                    raise RuntimeError(f"{pair_id}: {variant} image hash mismatch")
                prior_split = image_hash_to_split.setdefault(digest, first["split"])
                if prior_split != first["split"]:
                    raise RuntimeError(f"{pair_id}: image hash crosses splits")
                verified_images += 1

                regenerated = temporary_root / f"{pair_index:05d}-{variant}.png"
                render_scene(states[variant], regenerated, int(protocol["render_scale"]))
                if sha256_file(regenerated) != digest:
                    raise RuntimeError(f"{pair_id}: {variant} image does not match symbolic state")
                rerendered_images += 1

            base_path = Path(first["base_image_path"])
            edited_path = Path(first["edited_image_path"])
            hist_equal = _rgb_histogram(base_path) == _rgb_histogram(edited_path)
            hist_l1 = _histogram_l1(_rgb_histogram(base_path), _rgb_histogram(edited_path))
            if hist_equal != bool(first["base_edited_rgb_histogram_equal"]):
                raise RuntimeError(f"{pair_id}: RGB histogram equality mismatch")
            if hist_l1 != int(first["base_edited_rgb_histogram_l1_pixels"]):
                raise RuntimeError(f"{pair_id}: RGB histogram L1 mismatch")
            if first["challenge"] == "position_color_swap" and not hist_equal:
                raise RuntimeError(f"{pair_id}: exact-histogram control failed")
            challenge_histogram_counts[(first["challenge"], hist_equal)] += 1

            for row in family:
                expected = {
                    variant: evaluate_query(states[variant], row["oracle_query"])
                    for variant in VARIANTS
                }
                actual = {variant: row[f"{variant}_answer"] for variant in VARIANTS}
                if expected != actual:
                    raise RuntimeError(f"{pair_id}: oracle mismatch for {row['question_role']}")
                affected = row["question_role"].startswith("affected_")
                if affected != bool(row["answer_change_required"]):
                    raise RuntimeError(f"{pair_id}: answer-change declaration mismatch")
                if affected != (actual["base"] != actual["edited"]):
                    raise RuntimeError(f"{pair_id}: selective answer contract failed")
                if actual["base"] != actual["invariant"]:
                    raise RuntimeError(f"{pair_id}: invariant answer contract failed")
                role_answer_counts[(row["split"], row["question_role"], actual["base"], actual["edited"])] += 1

    generated_splits = set(receipt["generated_partitions"])
    observed_splits = {row["split"] for row in rows}
    if observed_splits != generated_splits:
        raise RuntimeError("observed splits differ from generation receipt")
    if not receipt.get("final_test_generated") and "final_test" in observed_splits:
        raise RuntimeError("sealed final test unexpectedly appears in manifest")

    result = {
        "schema_version": "2026-09-14-v1",
        "status": "passed_static_and_render_audit_pending_human_review",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol_path),
        "generation_receipt_sha256": sha256_file(generation_receipt_path),
        "manifest_sha256": sha256_file(manifest),
        "auditor_sha256": sha256_file(Path(__file__)),
        "scene_pairs": len(by_pair),
        "question_families": len(rows),
        "verified_source_images": verified_images,
        "independently_rerendered_images": rerendered_images,
        "unique_image_hashes": len(image_hash_to_split),
        "generated_partitions": sorted(observed_splits),
        "final_test_present": "final_test" in observed_splits,
        "role_answer_counts": {"|".join(key): value for key, value in sorted(role_answer_counts.items())},
        "challenge_histogram_counts": {"|".join(map(str, key)): value for key, value in sorted(challenge_histogram_counts.items())},
        "gates": {
            "manifest_and_source_hashes_match": True,
            "scene_pair_split_atomicity": True,
            "six_question_selectivity_contract": True,
            "symbolic_states_match_rendered_images": True,
            "position_swap_exact_rgb_histogram_preserved": True,
            "final_test_remains_absent": "final_test" not in observed_splits,
        },
        "formal_training_authorized": False,
        "claim_boundary": "Static and deterministic rerender checks passed. Human visual audit is still required before any training admission; no model result is implied.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "binding_intervention_data_protocol.json")
    parser.add_argument(
        "--generation-receipt",
        type=Path,
        default=ROOT / "evidence" / "binding-intervention-data-generation-v1-development.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "binding-intervention-data-static-audit-v1-development.json",
    )
    args = parser.parse_args()
    audit(args.protocol, args.generation_receipt, args.output)


if __name__ == "__main__":
    main()
