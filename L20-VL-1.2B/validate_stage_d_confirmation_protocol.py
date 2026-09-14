#!/usr/bin/env python3
"""Fail closed unless the Stage-D confirmation protocol is fully frozen."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from stage_d_confirmation_contract import (
    ARMS,
    CONDITIONS,
    SPLITS,
    confirmation_execution_order,
    sha256_file,
    stable_id_sha256,
)


EXPECTED_STATUS = "authorized_one_time_stage_d_iid_ood_confirmation_after_all_checkpoints_frozen"


def validate_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != EXPECTED_STATUS or protocol.get("authorized_by_user") is not True:
        raise RuntimeError("Stage-D confirmation is not authorized")
    if protocol.get("arms") != list(ARMS) or len(protocol.get("seeds", [])) != 5:
        raise RuntimeError("confirmation must contain exactly two arms and five seeds")
    if protocol.get("conditions") != list(CONDITIONS):
        raise RuntimeError("confirmation conditions changed")
    if protocol.get("execution_order") != confirmation_execution_order(protocol["seeds"]):
        raise RuntimeError("confirmation execution order changed")
    cells = [(item["arm"], item["seed"], item["split"]) for item in protocol["execution_order"]]
    expected = {(arm, seed, split) for arm in ARMS for seed in protocol["seeds"] for split in SPLITS}
    if len(cells) != len(set(cells)) or set(cells) != expected:
        raise RuntimeError("confirmation matrix is incomplete or duplicated")
    if protocol.get("expected_evaluations") != 20:
        raise RuntimeError("confirmation requires exactly twenty arm/seed/split evaluations")
    for record in list(protocol["source_files"].values()) + list(protocol["scorer_files"].values()):
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"frozen source hash mismatch: {record['path']}")
    manifest_path = Path(protocol["manifest"]["path"])
    if sha256_file(manifest_path) != protocol["manifest"]["sha256"]:
        raise RuntimeError("confirmation manifest hash mismatch")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        frozen = protocol["splits"][split]
        if len(selected) != frozen["scene_families"]:
            raise RuntimeError(f"{split} family count mismatch")
        if stable_id_sha256(row["scene_family_id"] for row in selected) != frozen["family_id_sha256"]:
            raise RuntimeError(f"{split} family identity mismatch")
        if dict(sorted(Counter(row["task"] for row in selected).items())) != frozen["tasks"]:
            raise RuntimeError(f"{split} task balance mismatch")
        if dict(sorted(Counter(row["base_answer"] for row in selected).items())) != frozen["base_answers"]:
            raise RuntimeError(f"{split} answer balance mismatch")
    d1_summary = json.loads(Path(protocol["source_files"]["replication_summary"]["path"]).read_text())
    if d1_summary.get("status") != "complete_checkpoints_frozen_tests_still_sealed":
        raise RuntimeError("D1 summary no longer reports frozen checkpoints and sealed tests")
    for key in ("prior_test_predictions_produced", "new_iid_test_predictions_produced", "ood_test_predictions_produced"):
        if d1_summary.get(key) != 0:
            raise RuntimeError(f"D1 summary reports prior test predictions: {key}")
    for arm in ARMS:
        if set(protocol["checkpoints"][arm]) != {str(seed) for seed in protocol["seeds"]}:
            raise RuntimeError(f"{arm} checkpoint seeds changed")
        for seed in protocol["seeds"]:
            record = protocol["checkpoints"][arm][str(seed)]
            checks = {
                "bridge": (Path(record["bridge"]), record["bridge_sha256"]),
                "adapter_model": (Path(record["adapter"]) / "adapter_model.safetensors", record["adapter_model_sha256"]),
                "adapter_config": (Path(record["adapter"]) / "adapter_config.json", record["adapter_config_sha256"]),
                "selection": (Path(record["selection_evidence"]), record["selection_evidence_sha256"]),
                "development": (Path(record["development_evidence"]), record["development_evidence_sha256"]),
                "prune": (Path(record["prune_receipt"]), record["prune_receipt_sha256"]),
            }
            for kind, (candidate, expected_hash) in checks.items():
                if sha256_file(candidate) != expected_hash:
                    raise RuntimeError(f"{arm}/{seed} {kind} hash mismatch")
            selection = json.loads(Path(record["selection_evidence"]).read_text())
            development = json.loads(Path(record["development_evidence"]).read_text())
            prune = json.loads(Path(record["prune_receipt"]).read_text())
            if selection["selected_step"] != record["step"]:
                raise RuntimeError(f"{arm}/{seed} selected step mismatch")
            if Path(selection["selected_checkpoint"]).resolve() != Path(record["checkpoint"]).resolve():
                raise RuntimeError(f"{arm}/{seed} selected checkpoint mismatch")
            if development["split"] != "development" or prune.get("status") != "complete":
                raise RuntimeError(f"{arm}/{seed} D1 evidence is incomplete")
    if protocol["scoring_precision"] != "float32" or protocol["tf32_enabled"] is not False:
        raise RuntimeError("confirmation scorer must remain float32 with TF32 disabled")
    if protocol["budget"]["projected_with_margin_gpu_hours"] > protocol["budget"]["confirmation_l20_gpu_hours_cap"]:
        raise RuntimeError("confirmation projection exceeds GPU budget")
    return {
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "evaluations": len(cells),
        "arms": list(ARMS),
        "seeds": protocol["seeds"],
        "splits": protocol["splits"],
        "projected_with_margin_gpu_hours": protocol["budget"]["projected_with_margin_gpu_hours"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_protocol(args.protocol.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
