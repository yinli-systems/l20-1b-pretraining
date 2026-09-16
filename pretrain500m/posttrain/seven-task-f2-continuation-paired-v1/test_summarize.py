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


def test_summary_binds_matched_checkpoints_and_progression_gate(tmp_path):
    protocol = "a" * 64
    base_path = tmp_path / "base.json"
    write(base_path, aggregate(0.4, protocol))
    reference_path = tmp_path / "reference.json"
    reference_sha = write(reference_path, aggregate(0.39, protocol))
    base_checkpoint_sha = hashlib.sha256(b"base").hexdigest()
    base_receipt = {
        "status": "PASS_HF_EXPORT_BITWISE_STATE_AND_BOUNDED_BF16_LOGIT_DRIFT",
        "checkpoint_sha256": base_checkpoint_sha,
        "checkpoint_step": 7629,
    }
    base_receipt_path = tmp_path / "results" / "base" / "export-receipt.json"
    base_receipt_sha = write(base_receipt_path, base_receipt)
    candidates = []
    for role, values in (("parent", (0.48, 0.49)), ("continuation", (0.50, 0.51))):
        for offset, score in enumerate(values):
            seed = 20260914 + offset
            candidate_id = f"F2_{role}_seed{seed}"
            checkpoint_sha = hashlib.sha256(candidate_id.encode()).hexdigest()
            candidates.append({"id": candidate_id, "recipe": f"F2_{role}", "seed": seed,
                               "role": role,
                               "checkpoint": f"/{candidate_id}.pt",
                               "checkpoint_sha256": checkpoint_sha,
                               "expected_step": 1024 if role == "parent" else 256})
            root = tmp_path / "results" / candidate_id
            write(root / "export-receipt.json", {
                "status": "PASS_HF_EXPORT_BITWISE_STATE_AND_BOUNDED_BF16_LOGIT_DRIFT",
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_step": 1024 if role == "parent" else 256,
            })
            write(root / "seven-task-aggregate.json", aggregate(score, protocol))
    plan_path = tmp_path / "plan.json"
    write(plan_path, {
        "schema": "p529m-seven-task-f2-continuation-paired-2gpu-plan-v1",
        "status": "FROZEN_BEFORE_THIS_COMPARISON",
        "protocol_id": "test", "protocol_sha256": protocol,
        "evaluation_class": "adaptive test fixture",
        "execution": {"world_size": 2},
        "base": {"checkpoint_sha256": base_checkpoint_sha, "expected_step": 7629,
                 "hf_export_receipt_sha256": base_receipt_sha},
        "reference_base_aggregate_sha256": reference_sha,
        "candidates": candidates,
    })
    output = tmp_path / "summary.json"
    subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")),
                    "--plan", str(plan_path), "--base", str(base_path),
                    "--reference-base", str(reference_path),
                    "--results-root", str(tmp_path / "results"), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["selected_recipe_by_seven_task_accuracy"] == "F2_continuation"
    assert result["recipes"][0]["mean_two_seed_aggregate"] == 0.505
    assert result["recipes"][1]["mean_two_seed_aggregate"] == 0.485
    assert all(pair["aggregate_delta_new_vs_parent"] > 0 for pair in result["pairs"])
    assert all(result["progression_gate"].values())
    assert result["two_gpu_base_delta_vs_reference"]["aggregate"] == 0.010000000000000009
    assert result["formal_promotion"] is False
