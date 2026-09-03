#!/usr/bin/env python3
"""Keep the fail-closed production supervisor alive across transient exits."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/hhai/pretrain")
PROJECT = ROOT / "src/hq-pretrain"
STATUS = ROOT / "manifests/watchdog-status.json"
PRODUCTION_STATUS = ROOT / "manifests/production-status.json"
SESSION = "production"
POLL_SECONDS = 30


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_status(stage: str, **values) -> None:
    payload = {"stage": stage, "updated_unix": time.time(), **values}
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, STATUS)
    print(json.dumps(payload, sort_keys=True), flush=True)


def production_is_running() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def progress_signature() -> dict:
    production = read_json(PRODUCTION_STATUS)
    sources = {}
    for source in ("math", "code", "web", "dclm"):
        progress = read_json(ROOT / "data/full-npy" / source / "progress.json")
        sources[source] = {
            "train_tokens": progress.get("train_tokens"),
            "val_tokens": progress.get("val_tokens"),
        }
    checkpoints = ROOT / "checkpoints/full"
    latest_checkpoint_mtime_ns = max(
        (path.stat().st_mtime_ns for path in checkpoints.glob("**/lit_model.pth")),
        default=None,
    )
    return {
        "production_stage": production.get("stage"),
        "sources": sources,
        "latest_checkpoint_mtime_ns": latest_checkpoint_mtime_ns,
    }


def start_production() -> None:
    command = (
        f"cd {PROJECT} && exec {ROOT / '.venv/bin/python'} production_supervisor.py "
        f">> {ROOT / 'logs/production-supervisor.log'} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, command], check=True)


def main() -> None:
    restart_count = 0
    repeated_signature_count = 0
    last_restart_signature = None
    while True:
        production = read_json(PRODUCTION_STATUS)
        if production.get("stage") == "complete":
            atomic_status("complete", restart_count=restart_count)
            return
        if production_is_running():
            atomic_status("monitoring", restart_count=restart_count)
            time.sleep(POLL_SECONDS)
            continue

        signature = progress_signature()
        if signature == last_restart_signature:
            repeated_signature_count += 1
        else:
            repeated_signature_count = 0
        last_restart_signature = signature
        delay = min(30 * 2 ** min(repeated_signature_count, 5), 900)
        atomic_status(
            "waiting_to_restart",
            restart_count=restart_count,
            repeated_signature_count=repeated_signature_count,
            delay_seconds=delay,
            progress_signature=signature,
        )
        time.sleep(delay)
        if production_is_running():
            continue
        start_production()
        restart_count += 1
        atomic_status("restarted", restart_count=restart_count, progress_signature=signature)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
