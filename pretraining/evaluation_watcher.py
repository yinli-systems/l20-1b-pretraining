#!/usr/bin/env python3
"""Wait for a complete final checkpoint, then run the external evaluation suite once."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/hhai/pretrain")
STATUS = ROOT / "manifests/evaluation-status.json"


def write_status(stage: str, **values) -> None:
    payload = {"stage": stage, "updated_unix": time.time(), **values}
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, STATUS)


def main() -> None:
    final_checkpoint = ROOT / "checkpoints/full/final/lit_model.pth"
    evaluation_receipt = ROOT / "manifests/final-evaluation-receipt.json"
    while not final_checkpoint.is_file():
        write_status("waiting_for_final_checkpoint")
        time.sleep(300)
    if evaluation_receipt.is_file():
        write_status("complete", receipt=str(evaluation_receipt))
        return
    while importlib.util.find_spec("lm_eval") is None:
        write_status("waiting_for_lm_eval_dependency", checkpoint=str(final_checkpoint))
        time.sleep(300)
    write_status("evaluating", checkpoint=str(final_checkpoint))
    with (ROOT / "logs/evaluate-final.log").open("a") as log:
        result = subprocess.run(
            [sys.executable, str(ROOT / "src/hq-pretrain/evaluate_final.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        write_status("failed", returncode=result.returncode, checkpoint=str(final_checkpoint))
        raise SystemExit(result.returncode)
    write_status("complete", receipt=str(evaluation_receipt))


if __name__ == "__main__":
    main()
