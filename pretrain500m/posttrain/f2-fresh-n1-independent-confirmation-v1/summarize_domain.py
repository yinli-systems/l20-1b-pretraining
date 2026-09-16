#!/usr/bin/env python3
"""Build a paired, pre-gated summary for one confirmation domain."""

import argparse
import datetime
import hashlib
import json
import random
from pathlib import Path
from typing import Optional


PARENT_ID = "F2_parent_seed20260915"
CANDIDATE_ID = "f2n1-lr3e5"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bootstrap_delta(values: list[float], seed: int, resamples: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(resamples):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return [samples[int(0.025 * (resamples - 1))], samples[int(0.975 * (resamples - 1))]]


def paired_rows(parent: dict, candidate: dict) -> list[tuple[dict, dict]]:
    left = sorted(parent["results"], key=lambda row: row["ordinal"])
    right = sorted(candidate["results"], key=lambda row: row["ordinal"])
    if len(left) != len(right):
        raise ValueError("paired result lengths differ")
    for a, b in zip(left, right):
        if (a["ordinal"], a["row_sha256"]) != (b["ordinal"], b["row_sha256"]):
            raise ValueError("paired result identity mismatch")
    return list(zip(left, right))


def metric(pairs, parent_key: str, candidate_key: Optional[str] = None, lower_better=False):
    candidate_key = candidate_key or parent_key
    parent_values = [float(a[parent_key]) for a, _ in pairs]
    candidate_values = [float(b[candidate_key]) for _, b in pairs]
    deltas = [b - a for a, b in zip(parent_values, candidate_values)]
    if lower_better:
        improvement = [-value for value in deltas]
    else:
        improvement = deltas
    return {
        "parent": sum(parent_values) / len(parent_values),
        "candidate": sum(candidate_values) / len(candidate_values),
        "candidate_minus_parent": sum(deltas) / len(deltas),
        "improvement_direction_delta": sum(improvement) / len(improvement),
        "paired_bootstrap_improvement_ci95": bootstrap_delta(improvement, 20260916, 10000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["reading", "math", "code", "knowledge"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    parent = json.loads(args.parent.read_text())
    candidate = json.loads(args.candidate.read_text())
    if parent["candidate"]["id"] != PARENT_ID or candidate["candidate"]["id"] != CANDIDATE_ID:
        raise ValueError("result candidate identity mismatch")
    if parent["role"] != "confirmation" or candidate["role"] != "confirmation":
        raise ValueError("results are not confirmation results")
    plan_sha = digest(args.plan)
    if parent["plan_sha256"] != plan_sha or candidate["plan_sha256"] != plan_sha:
        raise ValueError("result plan mismatch")
    pairs = paired_rows(parent, candidate)

    if args.domain in ("reading", "knowledge"):
        primary = metric(pairs, "correct_acc_norm")
        secondary = metric(pairs, "correct_acc")
        metrics = {"acc_norm": primary, "acc": secondary}
        gate = {
            "rule": "paired bootstrap 95% lower bound for acc_norm improvement is at least -0.01",
            "passed": primary["paired_bootstrap_improvement_ci95"][0] >= -0.01,
        }
    elif args.domain == "math":
        primary = metric(pairs, "exact_numeric_match")
        nll = metric(pairs, "gold_answer_token_nll", lower_better=True)
        metrics = {"exact_numeric_match": primary, "gold_answer_token_nll": nll}
        parent_correct = sum(bool(a["exact_numeric_match"]) for a, _ in pairs)
        candidate_correct = sum(bool(b["exact_numeric_match"]) for _, b in pairs)
        gate = {
            "rule": "candidate exact-match correct count is no lower than parent and mean gold-answer NLL is no worse",
            "parent_correct": parent_correct,
            "candidate_correct": candidate_correct,
            "passed": candidate_correct >= parent_correct and nll["improvement_direction_delta"] >= 0,
        }
    else:
        primary = metric(pairs, "execution_pass")
        syntax = metric(pairs, "syntax_and_policy_pass")
        nll = metric(pairs, "gold_code_token_nll", lower_better=True)
        metrics = {"execution_pass_at_1": primary, "syntax_and_policy_rate": syntax,
                   "gold_code_token_nll": nll}
        parent_correct = sum(bool(a["execution_pass"]) for a, _ in pairs)
        candidate_correct = sum(bool(b["execution_pass"]) for _, b in pairs)
        gate = {
            "rule": "candidate must exceed parent execution passes, break the zero-pass floor, and retain syntax/policy within a -0.01 paired lower bound",
            "parent_correct": parent_correct,
            "candidate_correct": candidate_correct,
            "passed": (candidate_correct > parent_correct and candidate_correct > 0 and
                       syntax["paired_bootstrap_improvement_ci95"][0] >= -0.01),
        }

    report = {
        "schema": "p529m-f2-fresh-n1-independent-confirmation-domain-summary-v1",
        "status": "INDEPENDENT_CONFIRMATION_DOMAIN_COMPLETE",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "domain": args.domain,
        "role": "confirmation",
        "parent_id": PARENT_ID,
        "candidate_id": CANDIDATE_ID,
        "samples": len(pairs),
        "metrics": metrics,
        "gate": gate,
        "plan_sha256": plan_sha,
        "input_sha256": {"parent": digest(args.parent), "candidate": digest(args.candidate)},
        "formal_promotion": False,
        "claim_boundary": plan["claim_boundary"],
    }
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
