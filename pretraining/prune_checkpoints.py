#!/usr/bin/env python3
"""Keep only the newest complete LitGPT step checkpoints in one scoped run directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    allowed_root = Path("/home/hhai/pretrain/checkpoints").resolve()
    if allowed_root not in run_dir.parents:
        raise SystemExit(f"refusing path outside {allowed_root}: {run_dir}")
    complete = sorted(path for path in run_dir.glob("step-*" ) if (path / "lit_model.pth").is_file())
    victims = complete[:-args.keep] if args.keep else complete
    for path in victims:
        print(f"{'DELETE' if args.apply else 'WOULD_DELETE'} {path}")
        if args.apply:
            shutil.rmtree(path)


if __name__ == "__main__":
    main()
