#!/usr/bin/env python3
"""Fail closed unless every Stage-C checkpoint was frozen before test use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_stage_a_full_token import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "authorized_one_time_stage_c_test_after_all_primary_arms_frozen":
        raise SystemExit("Stage-C test is not unsealed")
    matrix = protocol["matrix_protocol"]
    if sha256_file(Path(matrix["path"])) != matrix["sha256"]:
        raise SystemExit("matrix protocol changed after test unseal")
    expected_arms = {"answer_49", "kd_49", "strong_49", "proposed_49"}
    if set(protocol["selected_arms"]) != expected_arms:
        raise SystemExit("all four primary arms must be frozen")
    for arm, record in protocol["selected_arms"].items():
        bridge = Path(record["bridge"])
        adapter = bridge.parent / "language_adapter"
        checks = {
            "bridge": (bridge, record["bridge_sha256"]),
            "adapter_model": (adapter / "adapter_model.safetensors", record["adapter_model_sha256"]),
            "adapter_config": (adapter / "adapter_config.json", record["adapter_config_sha256"]),
            "selection": (Path(record["selection_evidence"]), record["selection_evidence_sha256"]),
            "development": (Path(record["development_evidence"]), record["development_evidence_sha256"]),
        }
        for kind, (path, expected) in checks.items():
            if sha256_file(path) != expected:
                raise SystemExit(f"{arm} {kind} hash mismatch")
        selection = json.loads(Path(record["selection_evidence"]).read_text())
        development = json.loads(Path(record["development_evidence"]).read_text())
        if selection["selected_step"] != record["step"]:
            raise SystemExit(f"{arm} selected step mismatch")
        selected_path = Path(selection["selected_checkpoint"])
        if not selected_path.is_absolute():
            selected_path = Path.cwd() / selected_path
        if selected_path.resolve() != bridge.resolve():
            raise SystemExit(f"{arm} selected checkpoint path mismatch")
        if development["split"] != "development":
            raise SystemExit(f"{arm} development evidence used wrong split")
    print(json.dumps({
        "status": "pass",
        "unseal_protocol_sha256": sha256_file(args.protocol),
        "arms": sorted(expected_arms),
        "selected_steps": {arm: protocol["selected_arms"][arm]["step"] for arm in sorted(expected_arms)},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
