"""Verify nested quality-feature provenance before appending the F2 tranche."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
PRIOR_SOURCE = ROOT / "source/quality-family-expansion-v1/scan.py"
EXPECTED_SCANNER_SHA256 = "3674f92a51e8ad819fda601d647ce070e27f66ab10e2da2e265484ad81a2338c"
ALLOWED_FEATURE_ROOTS = {
    ROOT / "data/quality-family-expansion-v1",
    ROOT / "data/quality-family-dclm-topup-v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_reusable_results(scanner, work, reuse_output, expected_report_sha256,
                            report, exclusions):
    prior_path = Path(reuse_output) / "report.json"
    if not expected_report_sha256 or sha256(prior_path) != expected_report_sha256:
        raise ValueError("nested reused quality report identity mismatch")
    prior = json.loads(prior_path.read_text())
    if prior.get("status") != "QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED" or prior.get("policy") != scanner.POLICY:
        raise ValueError("nested reused quality report status/policy mismatch")
    if report.get("incremental_from_report_sha256") != prior.get("raw_report_sha256"):
        raise ValueError("nested raw audit does not append the quality prefix")
    if exclusions.get("prior_union_sha256") != prior.get("exclusion_plan_sha256"):
        raise ValueError("nested exclusions do not append the quality prefix")

    prior_files = {(x["source_id"], x["tranche"]): x for x in prior["files"]}
    current_keys = {(x[0], x[1]) for x in work}
    if not set(prior_files) <= current_keys or sum(x["rows"] for x in prior_files.values()) != prior["rows"]:
        raise ValueError("nested reused quality file set is not a complete prefix")

    checked = {}
    for source_id, tranche, path, rows, _output in work:
        key = (source_id, tranche)
        if key not in prior_files:
            continue
        record = prior_files[key]
        if prior["input_bindings"].get(path) != report["input_bindings"].get(path):
            raise ValueError("nested reused raw file binding changed")
        feature = Path(record["features"])
        if feature.name != f"{tranche}-{source_id}.features.jsonl.gz" or feature.parent not in ALLOWED_FEATURE_ROOTS:
            raise ValueError("nested reused feature path is outside the frozen prefix")
        item = (source_id, tranche, path, rows, str(feature.parent))
        actual = scanner.completed_result(item)
        if actual is None or actual != record:
            raise ValueError("nested reused quality feature/progress/report mismatch")
        checked[key] = actual
    if set(checked) != set(prior_files):
        raise ValueError("nested reused quality prefix is incomplete")
    return checked


def load_scanner():
    if sha256(PRIOR_SOURCE) != EXPECTED_SCANNER_SHA256:
        raise ValueError("frozen quality scanner source changed")
    sys.path.insert(0, str(PRIOR_SOURCE.parent))
    spec = importlib.util.spec_from_file_location("p529m_f2_quality", PRIOR_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen quality scanner")
    scanner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scanner
    spec.loader.exec_module(scanner)
    scanner.reusable_results = lambda work, reuse_output, expected, report, exclusions: (
        nested_reusable_results(scanner, work, reuse_output, expected, report, exclusions)
    )
    return scanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--expected-raw-report-sha256", required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--expected-exclusions-sha256", required=True)
    parser.add_argument("--audit-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-output", type=Path, required=True)
    parser.add_argument("--expected-reuse-report-sha256", required=True)
    args = parser.parse_args()
    args.resume = False
    load_scanner().run(args)


if __name__ == "__main__":
    main()
