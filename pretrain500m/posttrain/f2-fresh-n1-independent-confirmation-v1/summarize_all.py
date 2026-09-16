#!/usr/bin/env python3
"""Combine four completed domain summaries under the frozen cross-domain gate."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate_plan = json.loads(args.gate_plan.read_text())
    summaries = [json.loads(path.read_text()) for path in args.summary]
    by_domain = {row["domain"]: row for row in summaries}
    expected = list(gate_plan["domains"])
    if set(by_domain) != set(expected) or len(summaries) != len(expected):
        raise ValueError("exactly one summary per frozen domain is required")
    domain_pass = {name: bool(by_domain[name]["gate"]["passed"]) for name in expected}
    primary_improvements = {
        "reading": by_domain["reading"]["metrics"]["acc_norm"]["improvement_direction_delta"],
        "math": by_domain["math"]["metrics"]["exact_numeric_match"]["improvement_direction_delta"],
        "code": by_domain["code"]["metrics"]["execution_pass_at_1"]["improvement_direction_delta"],
        "knowledge": by_domain["knowledge"]["metrics"]["acc_norm"]["improvement_direction_delta"],
    }
    improving = [name for name, value in primary_improvements.items() if value > 0]
    advance = all(domain_pass.values()) and len(improving) >= 2
    document = {
        "schema": "p529m-f2-fresh-n1-independent-confirmation-summary-v1",
        "status": "ADVANCE_TO_SECOND_SEED_REPLICATION" if advance else "HOLD_CANDIDATE_NO_AUTOMATIC_CONTINUATION",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate_id": "f2n1-lr3e5",
        "parent_id": "F2_parent_seed20260915",
        "domain_gate_pass": domain_pass,
        "primary_improvement_direction_delta": primary_improvements,
        "strictly_improving_primary_domains": improving,
        "cross_domain_gate": {
            "rule": "all four frozen retention gates pass and at least two primary metrics strictly improve",
            "passed": advance,
        },
        "domain_summaries": {name: by_domain[name] for name in expected},
        "gate_plan_sha256": digest(args.gate_plan),
        "input_sha256": {path.name + ":" + json.loads(path.read_text())["domain"]: digest(path) for path in args.summary},
        "formal_promotion": False,
        "claim_boundary": gate_plan["claim_boundary"],
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
