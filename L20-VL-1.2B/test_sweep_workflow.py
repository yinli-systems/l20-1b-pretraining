#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SweepWorkflowTests(unittest.TestCase):
    def make_fixture(self, root: Path, complete: bool = True) -> tuple[Path, Path]:
        run = root / "run"
        run.mkdir()
        (run / "run.json").write_text(json.dumps({
            "status": "complete" if complete else "running",
            "optimizer_step": 875 if complete else 700,
            "arm": "spatial_49",
        }))
        for step in (100, 875):
            checkpoint = run / f"step-{step:06d}"
            adapter = checkpoint / "language_adapter"
            adapter.mkdir(parents=True)
            (checkpoint / "bridge.safetensors").write_bytes(f"bridge-{step}".encode())
            (checkpoint / "state.json").write_text(json.dumps({"step": step}))
            (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
            (adapter / "adapter_model.safetensors").write_bytes(f"adapter-{step}".encode())
            (adapter / "adapter_config.json").write_text("{}")
        selected = run / "step-000875"
        selection = root / "selection.json"
        selection.write_text(json.dumps({
            "protocol_arm": "spatial_49",
            "selected_step": 875,
            "selected_checkpoint": str(selected),
            "selected_checkpoint_sha256": digest(selected / "bridge.safetensors"),
            "selected_language_adapter_model_sha256": digest(selected / "language_adapter" / "adapter_model.safetensors"),
        }))
        return run, selection

    def test_prune_retains_only_selected_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, selection = self.make_fixture(root)
            receipt = root / "receipt.json"
            subprocess.run([
                sys.executable,
                str(ROOT / "prune_completed_sweep_run.py"),
                "--run", str(run),
                "--selection", str(selection),
                "--output", str(receipt),
            ], check=True, capture_output=True, text=True)
            self.assertFalse((run / "step-000100").exists())
            self.assertTrue((run / "step-000875" / "bridge.safetensors").exists())
            self.assertFalse((run / "step-000875" / "optimizer.pt").exists())
            result = json.loads(receipt.read_text())
            self.assertEqual(result["status"], "complete")
            self.assertGreater(result["bytes_reclaimed"], 0)

    def test_prune_rejects_incomplete_run_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, selection = self.make_fixture(root, complete=False)
            result = subprocess.run([
                sys.executable,
                str(ROOT / "prune_completed_sweep_run.py"),
                "--run", str(run),
                "--selection", str(selection),
                "--output", str(root / "receipt.json"),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((run / "step-000100").exists())
            self.assertTrue((run / "step-000875" / "optimizer.pt").exists())

    def test_prune_accepts_declared_microsearch_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            checkpoint = run / "step-000150"
            adapter = checkpoint / "language_adapter"
            adapter.mkdir(parents=True)
            (checkpoint / "bridge.safetensors").write_bytes(b"bridge")
            (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            (run / "run.json").write_text(json.dumps({
                "status": "complete",
                "optimizer_step": 150,
                "arm": "F_matched_196",
            }))
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "protocol_arm": "F_matched_196",
                "selected_checkpoint": str(checkpoint),
                "selected_step": 150,
                "selected_checkpoint_sha256": digest(checkpoint / "bridge.safetensors"),
                "selected_language_adapter_model_sha256": digest(
                    adapter / "adapter_model.safetensors"
                ),
            }))
            receipt = root / "receipt.json"
            subprocess.run([
                sys.executable,
                str(ROOT / "prune_completed_sweep_run.py"),
                "--run", str(run),
                "--selection", str(selection),
                "--output", str(receipt),
                "--expected-optimizer-steps", "150",
            ], check=True)
            result = json.loads(receipt.read_text())
            self.assertEqual(result["expected_optimizer_steps"], 150)
            self.assertFalse((checkpoint / "optimizer.pt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
