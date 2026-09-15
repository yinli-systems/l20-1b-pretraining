import hashlib
import json
from pathlib import Path
import subprocess
import sys


TASKS = ("hellaswag", "piqa", "winogrande", "openbookqa", "arc_easy", "arc_challenge", "boolq")


def write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(score: float, protocol: str) -> dict:
    return {
        "status": "PASS_FROZEN_EVALUATION_AGGREGATE",
        "protocol_sha256": protocol,
        "aggregate": {"score": score},
        "tasks": {task: {"score": score} for task in TASKS},
    }


def test_summarizes_hash_bound_candidate_against_base(tmp_path):
    protocol = "a" * 64
    base = tmp_path / "base.json"
    base_sha = write(base, aggregate(0.4, protocol))
    candidate_id = "r4-lr6e5-t1b-seed20260916"
    checkpoint_sha = hashlib.sha256(candidate_id.encode()).hexdigest()
    candidates = [
        {"id": candidate_id, "checkpoint_sha256": checkpoint_sha, "expected_step": 512},
        {"id": "r4-lr6e5-t1b-seed20260917", "checkpoint_sha256": "b" * 64, "expected_step": 512},
    ]
    plan = tmp_path / "plan.json"
    write(plan, {
        "schema": "p529m-r4-extension-capability-screen-v1",
        "status": "FROZEN_BEFORE_EXTENSION_CAPABILITY_SCREEN",
        "protocol_id": "test",
        "protocol_sha256": protocol,
        "base": {"aggregate_sha256": base_sha},
        "candidates": candidates,
        "claim_boundary": "test only",
    })
    candidate_aggregate = tmp_path / "candidate.json"
    write(candidate_aggregate, aggregate(0.5, protocol))
    receipt = tmp_path / "export-receipt.json"
    write(receipt, {
        "status": "PASS_HF_EXPORT_BITWISE_STATE_AND_BOUNDED_BF16_LOGIT_DRIFT",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 512,
    })
    output = tmp_path / "result.json"
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("summarize_candidate.py")),
        "--plan", str(plan),
        "--candidate-id", candidate_id,
        "--aggregate", str(candidate_aggregate),
        "--export-receipt", str(receipt),
        "--base", str(base),
        "--output", str(output),
    ], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["candidate_aggregate_score"] == 0.5
    assert result["aggregate_delta_vs_base"] == 0.09999999999999998
    assert result["formal_promotion"] is False
