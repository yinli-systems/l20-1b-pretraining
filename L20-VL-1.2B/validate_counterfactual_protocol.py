#!/usr/bin/env python3
"""Fail-closed validation for the evidence-preserving compression protocol."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent
    protocol = json.loads((root / "counterfactual_protocol.json").read_text())
    admission = json.loads((root / "counterfactual_data_admission.json").read_text())
    errors: list[str] = []
    if protocol.get("formal_training_authorized") is not False:
        errors.append("counterfactual protocol cannot authorize formal training")
    if protocol.get("automatic_stage_advance") is not False:
        errors.append("automatic stage advance must be disabled")
    vision = protocol["parents"]["vision"]
    if (vision["resolution"], vision["patch_size"], vision["input_visual_tokens"]) != (224, 16, 196):
        errors.append("v1 must keep the verified 224px, 196-token vision parent")
    decision = protocol["resolution_decision"]
    if decision.get("primary_compressed_tokens") != 49 or decision.get("compression_ratio") != 4.0:
        errors.append("primary compression condition must be 196 to 49 tokens")
    data = protocol["diagnostic_data"]
    if data.get("synthetic_pair_target") != 5000:
        errors.append("diagnostic target must remain 5000 scene families")
    if sum(data["split"][key] for key in ("train_percent", "development_percent", "test_percent")) != 100:
        errors.append("diagnostic split percentages must sum to 100")
    if data["split"].get("atomic_unit") != "scene_family_id":
        errors.append("counterfactual variants must split by scene family")
    if data["split"].get("cross_split_variant_overlap_allowed") is not False:
        errors.append("paired variants cannot cross splits")
    matrix = protocol["experiment_matrix"]
    if len(matrix.get("primary_49_token_sequence", [])) != 4:
        errors.append("exactly four primary 49-token experiments are required")
    if "parameter_free_2d_adaptive_pool_49_deco_style" not in matrix.get("reference_and_architecture_controls", []):
        errors.append("DeCo-style parameter-free pooling control is required")
    required_ablations = {
        "shuffle_pair_links_with_data_labels_and_compute_unchanged",
        "apply_delta_objective_to_full_token_branch",
    }
    if set(matrix.get("mandatory_ablations", [])) != required_ablations:
        errors.append("both mechanism ablations are mandatory")
    if protocol["metrics"].get("primary") != "paired_joint_accuracy_base_and_answer_changing_edit_both_correct":
        errors.append("primary metric must require both counterfactual answers to be correct")
    stats = protocol["statistics"]
    if stats.get("resamples", 0) < 10000 or stats.get("cluster") != "scene_family_id_or_original_real_image_family":
        errors.append("clustered bootstrap protocol is incomplete")
    if stats.get("confirmatory_training_seeds", 0) < 3:
        errors.append("confirmatory result requires at least three training seeds")
    if "requires_a_separately_predeclared_equivalence_margin" not in stats.get("equivalence_claim_rule", ""):
        errors.append("CI crossing zero cannot be called equivalence")
    if protocol["required_foundation"].get("full_token_vlm_checkpoint_status") != "not_yet_trained":
        errors.append("foundation checkpoint status changed without a protocol revision")
    if protocol["real_image_transfer"].get("status", "").startswith("blocked_") is False:
        errors.append("real-image benchmarks must remain blocked until frozen")
    protocol_hash = hashlib.sha256((root / "counterfactual_protocol.json").read_bytes()).hexdigest()
    if admission.get("required_protocol_sha256") != protocol_hash:
        errors.append("diagnostic data admission does not pin the current protocol")
    if admission.get("generator_authorized") is not True:
        errors.append("diagnostic generator needs explicit authorization")
    if admission.get("external_downloads_authorized") is not False:
        errors.append("diagnostic generator cannot authorize external downloads")
    if admission.get("formal_training_authorized") is not False:
        errors.append("diagnostic data admission cannot authorize training")
    if admission.get("pair_count") != data.get("synthetic_pair_target"):
        errors.append("diagnostic pair count differs between protocol and admission")
    if admission.get("variants_per_pair") != 3:
        errors.append("every diagnostic family must contain exactly three image variants")
    if errors:
        raise SystemExit("FAIL\n" + "\n".join(errors))
    print("PASS: counterfactual implementation only; 5000-pair generator allowed, formal training blocked")


if __name__ == "__main__":
    main()
