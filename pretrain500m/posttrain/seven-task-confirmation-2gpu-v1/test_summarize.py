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


def test_summary_binds_receipts_and_averages_two_seeds(tmp_path):
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
    for recipe, values in (("F2", (0.5, 0.6)), ("F3", (0.7, 0.8))):
        for offset, score in enumerate(values):
            seed = 10 + offset
            candidate_id = f"{recipe}_{seed}"
            checkpoint_sha = hashlib.sha256(candidate_id.encode()).hexdigest()
            candidates.append({"id": candidate_id, "recipe": recipe, "seed": seed,
                               "checkpoint": f"/{candidate_id}.pt",
                               "checkpoint_sha256": checkpoint_sha, "expected_step": 1024})
            root = tmp_path / "results" / candidate_id
            write(root / "export-receipt.json", {
                "status": "PASS_HF_EXPORT_BITWISE_STATE_AND_BOUNDED_BF16_LOGIT_DRIFT",
                "checkpoint_sha256": checkpoint_sha, "checkpoint_step": 1024,
            })
            write(root / "seven-task-aggregate.json", aggregate(score, protocol))
    plan_path = tmp_path / "plan.json"
    write(plan_path, {
        "schema": "p529m-seven-task-confirmation-2gpu-plan-v1",
        "status": "FROZEN_BEFORE_MATCHED_TWO_GPU_EVALUATION",
        "protocol_id": "test", "protocol_sha256": protocol,
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
    assert result["selected_recipe_by_seven_task_accuracy"] == "F3"
    assert result["recipes"][0]["mean_two_seed_aggregate"] == 0.75
    assert result["recipes"][1]["mean_two_seed_aggregate"] == 0.55
    assert result["two_gpu_base_delta_vs_reference"]["aggregate"] == 0.010000000000000009
    assert result["formal_promotion"] is False
