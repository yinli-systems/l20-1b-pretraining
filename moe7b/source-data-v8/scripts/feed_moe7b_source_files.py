#!/usr/bin/env python3
"""Login-node download feeder for compute nodes without outbound internet access."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_moe7b_full_data.py"
SPEC = importlib.util.spec_from_file_location("moe7b_builder", BUILD_SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(BUILDER.canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def progress_by_component(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {item["component_id"]: item for item in payload.get("components", [])}


def cached(raw_root: Path, component: dict, record: dict) -> bool:
    path = BUILDER.safe_cache_path(raw_root, component["source_id"], record["path"])
    marker = path.with_suffix(path.suffix + ".verified.json")
    if not path.is_file() or not marker.is_file() or path.stat().st_size != record["size"]:
        return False
    receipt = json.loads(marker.read_text())
    return receipt.get("bytes") == record["size"] and receipt.get("sha256") == record["lfs_sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--prefetch-files", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    if args.prefetch_files < 1:
        raise SystemExit("--prefetch-files must be positive")
    config = json.loads(args.build.read_text())
    lock_root = ROOT / config["source_lock_directory"]
    locks = {
        component["source_id"]: json.loads((lock_root / f"{component['source_id']}.json").read_text())
        for component in config["components"]
    }
    stop = [False]

    def request_stop(_signum, _frame) -> None:
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    downloaded = 0
    while not stop[0]:
        observed = progress_by_component(args.build_root / "progress.json")
        active = None
        for component in config["components"]:
            state = observed.get(component["component_id"], {})
            if state.get("target_reached"):
                continue
            active = component
            break
        if active is None:
            atomic_json(
                args.status,
                {"status": "COMPLETE", "downloaded_files_this_run": downloaded, "updated_unix": time.time()},
            )
            return 0
        lock = locks[active["source_id"]]
        files = BUILDER.input_files(active, lock, int(config["selection_seed"]))
        completed = int(observed.get(active["component_id"], {}).get("completed_input_files", 0))
        selected = files[completed : completed + args.prefetch_files]
        if not selected:
            raise RuntimeError(f"source exhausted before target: {active['component_id']}")
        for record in selected:
            if stop[0]:
                break
            if cached(args.raw_root, active, record):
                continue
            path = BUILDER.download_verified(
                args.endpoint.rstrip("/"),
                args.raw_root,
                active["source_id"],
                lock["dataset_id"],
                lock["requested_revision"],
                record,
            )
            downloaded += 1
            atomic_json(
                args.status,
                {
                    "status": "RUNNING",
                    "active_component": active["component_id"],
                    "latest_repository_path": record["path"],
                    "latest_local_path": str(path),
                    "downloaded_files_this_run": downloaded,
                    "updated_unix": time.time(),
                },
            )
        time.sleep(args.poll_seconds)
    atomic_json(
        args.status,
        {"status": "CHECKPOINTED_STOP", "downloaded_files_this_run": downloaded, "updated_unix": time.time()},
    )
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
