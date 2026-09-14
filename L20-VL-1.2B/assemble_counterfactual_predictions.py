#!/usr/bin/env python3
"""Assemble true-image evaluator outputs into paired method predictions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


VARIANTS = ("base", "edited", "invariant")


def load_evaluation(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    result = json.loads(path.read_text())
    rows = {row["scene_family_id"]: row for row in result["predictions"]}
    if len(rows) != len(result["predictions"]):
        raise ValueError(f"duplicate scene families in {path}")
    return result, rows


def correctness(row: dict[str, Any]) -> dict[str, bool]:
    return {
        f"{variant}_correct": row["predictions"]["true_image"][variant] == row["expected"][variant]
        for variant in VARIANTS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--full-name", default="full_196")
    parser.add_argument("--baseline-name", default="strong_49")
    parser.add_argument("--candidate-name", default="proposed_49")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loaded = [load_evaluation(path) for path in (args.full, args.baseline, args.candidate)]
    metadata = [item[0] for item in loaded]
    methods = [item[1] for item in loaded]
    if len({item["split"] for item in metadata}) != 1:
        raise RuntimeError("evaluation splits differ")
    family_sets = [set(item) for item in methods]
    if family_sets[0] != family_sets[1] or family_sets[0] != family_sets[2]:
        raise RuntimeError("evaluation family sets differ")
    names = (args.full_name, args.baseline_name, args.candidate_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with partial.open("w") as handle:
        for family in sorted(family_sets[0]):
            source_rows = [method[family] for method in methods]
            if len({row["task"] for row in source_rows}) != 1:
                raise RuntimeError(f"task mismatch for {family}")
            if any(row["expected"] != source_rows[0]["expected"] for row in source_rows[1:]):
                raise RuntimeError(f"expected-answer mismatch for {family}")
            handle.write(json.dumps({
                "scene_family_id": family,
                "task": source_rows[0]["task"],
                "split": source_rows[0]["split"],
                "methods": {
                    name: correctness(row) for name, row in zip(names, source_rows)
                },
            }, sort_keys=True) + "\n")
    os.replace(partial, args.output)
    print(json.dumps({
        "split": metadata[0]["split"],
        "scene_families": len(family_sets[0]),
        "methods": names,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
