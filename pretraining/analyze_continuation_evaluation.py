#!/usr/bin/env python3
"""Analyze a continuation receipt on CPU, without loading models or contacting hosts.

Example: python analyze_continuation_evaluation.py --branch A --phase core
Writes core-summary.json for core, or summary.json for all completed suites.
Input directories can be overridden for local copies
(receipt result paths must then refer to those copies, or be relative paths).
The analysis protocol is fixed: 10,000 paired draws, seed 20260911.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import numpy as np

from analyze_comparison import PRIMARY, boolq_diagnostics, compare_task, read_verified
from evaluate_comparison import BASELINES, CORE, ORIGINALS, OUTPUT, PROTOCOL, ROOT, save_json, sha256

RUN = ROOT / "continuation/20260911-v1"
REPETITIONS = 10000
SEED = 20260911
SUITES = {"core": (list(CORE), 0), "mmlu_5shot": (["mmlu"], 5),
          "gsm8k_5shot": (["gsm8k"], 5)}
BOOLQ_REASON = "BoolQ was not explicitly covered by original 20B benchmark decontamination."


def resolve_path(value: str, directory: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else directory / path


class Evidence:
    """Bind every consumed file and detect concurrent changes before publication."""

    def __init__(self):
        self.files = {}
        self.stamps = {}

    @staticmethod
    def stamp(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def read(self, path: Path, digest: str | None = None, result: bool = False):
        path = path.resolve()
        before = self.stamp(path)
        if result:
            value = read_verified(path, digest)
        else:
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if digest is not None and actual != digest:
                raise ValueError(f"Hash mismatch: {path}")
            digest = actual
            value = json.loads(raw)
        self.record(path, digest, before)
        return value

    def record(self, path, digest, before):
        if self.stamp(path) != before or (path in self.stamps and self.stamps[path] != before):
            raise ValueError(f"Input changed during analysis: {path}")
        self.stamps[path] = before
        self.files[str(path)] = digest

    def artifact(self, path, digest):
        path = path.resolve()
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 for {path}")
        if str(path) in self.files:
            if self.files[str(path)] != digest:
                raise ValueError(f"Conflicting hashes for {path}")
            return
        before = self.stamp(path)
        if sha256(path) != digest:
            raise ValueError(f"Hash mismatch: {path}")
        self.record(path, digest, before)

    def unchanged(self):
        for path, before in self.stamps.items():
            if self.stamp(path) != before:
                raise ValueError(f"Input changed during analysis: {path}")


def verify_identity(receipt: dict, directory: Path, branch: str,
                    model: str, evidence: Evidence) -> dict:
    """Check runner identity and export metadata; expose unavailable evidence.

    The export manifest is mandatory when its hash is declared. Native checkpoint
    files may be absent in a copied evidence bundle. Unknown identity fields are
    retained, and shared manifest fields must agree without freezing key names.
    """
    checks, gaps = [], []
    identity = receipt.get("identity", {})
    if "branch" in receipt and receipt["branch"] != branch:
        raise ValueError("Branch identity mismatch")
    if "branch" not in receipt:
        gaps.append("Receipt has no branch identity.")

    def optional_artifact(path, digest, label):
        if not path.is_file():
            gaps.append(f"{label}: artifact unavailable: {path}")
            return
        evidence.artifact(path, digest)
        checks.append({"field": label, "path": str(path.resolve()), "sha256": digest})
    paths = {"checkpoint_sha256": directory.parent / f"pilot-{branch}/resume.pth",
             "parent_sha256": ROOT / "checkpoints/full/final/lit_model.pth",
             "data_plan_sha256": directory.parent / f"pilot-{branch}/data-plan.json"}
    for key, path in paths.items():
        if key in identity:
            optional_artifact(path, identity[key], key)
        else:
            gaps.append(f"Identity has no {key}.")
    manifest_path = directory / "hf/manifest.json"
    if "export_manifest_sha256" in identity:
        manifest = evidence.read(manifest_path, identity["export_manifest_sha256"])
        declared = manifest.get("identity", {})
        for key in set(identity) | set(declared):
            if key == "export_manifest_sha256":
                continue
            if key not in identity or key not in declared:
                gaps.append(f"Identity field {key} is not present in both receipt and export.")
            elif identity[key] != declared[key]:
                raise ValueError(f"Checkpoint/export identity mismatch: {key}")
        if not manifest.get("parity", {}).get("passed"):
            raise ValueError("Export parity record did not pass")
        if not manifest.get("files"):
            raise ValueError("Export manifest has no artifact hashes")
        for name, digest in manifest["files"].items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Invalid export artifact basename: {name}")
            path = manifest_path.parent / name
            evidence.artifact(path, digest)
            checks.append({"field": "export_file", "path": str(path.resolve()), "sha256": digest})
        if resolve_path(model, directory).resolve() != manifest_path.parent.resolve():
            raise ValueError("Export/model identity mismatch")
        checks.append({"field": "export_identity", "model_path_alignment": "PASS",
                       "parity": manifest["parity"]})
    else:
        gaps.append("Identity has no export_manifest_sha256; export is not verified.")
    checkpoint_receipt = directory.parent / f"pilot-{branch}/resume.json"
    if checkpoint_receipt.is_file():
        native = evidence.read(checkpoint_receipt)
        for key, native_key in (("checkpoint_sha256", "sha256"), ("parent_sha256", "parent_sha256"), ("step", "step")):
            if key in identity and identity[key] != native.get(native_key):
                raise ValueError(f"Native checkpoint identity mismatch: {key}")
    if "input_result_sha256" in receipt:
        for path, digest in receipt["input_result_sha256"].items():
            evidence.artifact(resolve_path(path, directory), digest)
    else:
        gaps.append("Receipt has no input_result_sha256; inputs are bound by frozen comparison hashes only.")
    return {"status": "partial" if gaps else "declared_artifacts_verified",
            "evaluated_model": model, "checks": checks, "unverified": gaps,
            "receipt_identity": identity, "runner_sha256": receipt.get("runner_sha256"),
            "scope": "Available declared files and identity links; parity is a hash-bound runner record, not rerun here."}


def validate_suite(result: dict, suite: str, tasks: list[str] | None = None):
    expected_tasks, shot = SUITES[suite]
    observed = set(result["samples"])
    if tasks is not None:
        expected = set(tasks)
    elif suite == "mmlu_5shot":
        expected = observed
        if not expected or any(not name.startswith("mmlu_") for name in expected):
            raise ValueError("Invalid MMLU subject set")
    else:
        expected = set(expected_tasks)
    if observed != expected:
        raise ValueError(f"Wrong task set: {suite}")
    for task in expected:
        if result["n-shot"][task] != shot or result["configs"][task].get("num_fewshot") != shot:
            raise ValueError(f"Wrong few-shot config: {task}")
        if result["n-samples"][task]["effective"] <= 0:
            raise ValueError(f"Empty sample set: {task}")
    if result["config"].get("device") != PROTOCOL["device"]:
        raise ValueError("Wrong evaluation device config")


def read_job(receipt, name, suite, directory, evidence, *, tasks=None, model=None, revision=None):
    row = receipt["jobs"].get(name, {})
    if row.get("status") != "complete":
        raise ValueError(f"Required job is not complete: {name}")
    expected_tasks, shot = SUITES[suite]
    expected_tasks = tasks if tasks is not None else expected_tasks
    job = row["job"]
    if (job.get("name"), job.get("tasks"), job.get("num_fewshot")) != (name, expected_tasks, shot):
        raise ValueError(f"Job protocol mismatch: {name}")
    result = evidence.read(resolve_path(row["result"], directory), row["sha256"], result=True)
    validate_suite(result, suite, tasks)
    actual = result["config"].get("model")
    if not isinstance(actual, str) or not actual:
        raise ValueError(f"Missing model identity: {name}")
    expected_model = model if model is not None else job.get("model")
    if expected_model is not None and actual != expected_model:
        raise ValueError(f"Model identity mismatch: {name}")
    if model is not None and job.get("model") != model:
        raise ValueError(f"Job model identity mismatch: {name}")
    if revision is not None:
        if job.get("revision") != revision or result["config"].get("model_revision") != revision:
            raise ValueError(f"Pinned revision mismatch: {name}")
    return result


def compare(ours, baseline, task, metric, rng, filter_name="none"):
    # Model names, tokenizer IDs, parameter counts and automatic batch choices
    # legitimately differ. Evaluation behavior must agree, including generation.
    for key in ("gen_kwargs", "use_cache", "apply_chat_template", "fewshot_as_multiturn", "chat_template"):
        if ours["config"].get(key) != baseline["config"].get(key):
            raise ValueError(f"Protocol mismatch: {task}/config/{key}")
    return compare_task(ours, baseline, task, metric, rng, REPETITIONS, filter_name)


def aggregate(rows, draws, weights=None):
    weights = np.ones(len(rows)) if weights is None else np.asarray(weights)
    bootstrap = np.average(draws, axis=0, weights=weights)
    return {"ours": float(np.average([r["ours"] for r in rows], weights=weights)),
            "baseline": float(np.average([r["baseline"] for r in rows], weights=weights)),
            "difference_pp": float(np.average([r["difference_pp"] for r in rows], weights=weights)),
            "paired_95_ci_pp": (np.quantile(bootstrap, [0.025, 0.975]) * 100).tolist()}


def analyze(branch: str, evaluation_dir: Path, comparison_dir: Path = OUTPUT,
            final_dir: Path = ROOT / "evaluations/final", phase: str = "all") -> tuple[dict, dict | None]:
    if branch not in ("A", "B"):
        raise ValueError("Branch must be A or B")
    if phase not in ("core", "all"):
        raise ValueError("Phase must be core or all")
    evidence = Evidence()
    receipt = evidence.read(evaluation_dir / "receipt.json")
    reference = evidence.read(comparison_dir / "receipt.json")
    for label, document in (("continuation", receipt), ("comparison", reference)):
        if document.get("protocol") != PROTOCOL:
            raise ValueError(f"Receipt protocol mismatch: {label}")
        if "original_result_sha256" in document and document["original_result_sha256"] != ORIGINALS:
            raise ValueError(f"Original result hashes mismatch: {label}")
    if reference.get("original_result_sha256") != ORIGINALS:
        raise ValueError("Comparison receipt does not bind frozen original results")
    statuses = {name: receipt["jobs"].get(name, {}).get("status", "missing") for name in SUITES}
    required = ("core",) if phase == "core" else tuple(SUITES)
    if any(statuses[name] != "complete" for name in required):
        raise ValueError(f"Required suites are not complete for phase {phase}: {statuses}")
    candidate = {name: read_job(receipt, name, name, evaluation_dir, evidence)
                 for name in required}
    if "core" not in candidate:
        raise ValueError("Continuation core is not complete")
    model = candidate["core"]["config"]["model"]
    if any(result["config"]["model"] != model for result in candidate.values()):
        raise ValueError("Continuation jobs used different models")
    identity = verify_identity(receipt, evaluation_dir, branch, model, evidence)

    def original(suite):
        name = "zero_shot_core" if suite == "core" else suite
        data = evidence.read(final_dir / f"{name}.json", ORIGINALS[name], result=True)
        validate_suite(data, suite, [task for task in CORE if task != "boolq"] if suite == "core" else None)
        return data

    def baseline(name, suite):
        model_id, revision = BASELINES[name]
        return read_job(reference, f"{name}_{suite}", suite, comparison_dir, evidence,
                        model=model_id, revision=revision)

    old_core = original("core")
    old_boolq = read_job(reference, "ours_boolq", "core", comparison_dir, evidence,
                         tasks=["boolq"], model=str(ROOT / "evaluations/final/hf"))
    comparisons = {}
    diagnostics = {"continuation": boolq_diagnostics(candidate["core"]),
                   "original20B": boolq_diagnostics(old_boolq)}
    reported = {"continuation": {"core": candidate["core"]["results"]},
                "original20B": {"core": old_core["results"], "boolq": old_boolq["results"]}}
    rng = np.random.default_rng(SEED)
    for name in ("original20B", *BASELINES):
        source = old_core if name == "original20B" else baseline(name, "core")
        if name != "original20B":
            diagnostics[name] = boolq_diagnostics(source)
            reported[name] = {"core": source["results"]}
        rows, draws, extended = {}, {}, {}
        for task, metric in PRIMARY.items():
            right = old_boolq if name == "original20B" and task == "boolq" else source
            rows[task], draws[task] = compare(candidate["core"], right, task, metric, rng)
        primary_macro = aggregate(list(rows.values()), list(draws.values()))
        sensitivity = aggregate([r for t, r in rows.items() if t != "boolq"],
                                [d for t, d in draws.items() if t != "boolq"])
        sensitivity["reason"] = BOOLQ_REASON
        for task in ("lambada_openai", "truthfulqa_mc2"):
            extended[task], _ = compare(candidate["core"], source, task, "acc", rng)
        comparisons[name] = {"primary": rows, "primary_macro": primary_macro,
                             "six_task_sensitivity_excluding_boolq": sensitivity, "extended": extended}
    core_evidence = dict(evidence.files)
    core = {
        "created_unix": time.time(), "branch": branch, "status": "core_complete",
        "suite_status_at_analysis": receipt.get("status"), "job_statuses": statuses,
        "extended_ready": all(status == "complete" for status in statuses.values()),
        "protocol": PROTOCOL, "baseline_revisions": BASELINES,
        "evaluation_receipt_sha256": evidence.files[str((evaluation_dir / "receipt.json").resolve())],
        "comparison_receipt_sha256": evidence.files[str((comparison_dir / "receipt.json").resolve())],
        "original_result_sha256": ORIGINALS, "input_file_sha256": core_evidence,
        "analysis_sha256": sha256(Path(__file__)),
        "runner_sha256": receipt.get("runner_sha256"),
        "analysis_dependencies_sha256": {name: sha256(Path(__file__).with_name(name))
                                         for name in ("analyze_comparison.py", "evaluate_comparison.py")},
        "analysis_environment": {"python": platform.python_version(), "numpy": np.__version__},
        "identity_verification": identity,
        "uncertainty": {"method": "paired percentile bootstrap; macro stratified by task",
                        "repetitions": REPETITIONS, "seed": SEED,
                        "scope": "test-sample uncertainty, not training-seed uncertainty; no multiple-testing correction"},
        "primary_metrics": PRIMARY, "primary_aggregation": "equal weight across seven tasks",
        "ours_primary_scores": {t: candidate["core"]["results"][t][f"{m},none"] for t, m in PRIMARY.items()},
        "ours_primary_macro": comparisons["original20B"]["primary_macro"]["ours"],
        "boolq_diagnostics": diagnostics, "comparisons": comparisons,
        "claims": {"sota_asserted": False, "automatic_training_promotion": False},
        "limitations": [BOOLQ_REASON, "Public benchmarks are not an unseen selection holdout.",
                        "Same 2048-token cap; tokenizer-dependent truncation can differ.",
                        "Cross-tokenizer perplexities must not be ranked directly.",
                        "No Chat template or code-generation benchmark is included."],
    }
    full = None
    if phase == "all":
        full = copy.deepcopy(core)
        full["status"] = "complete"
        for suite in ("mmlu_5shot", "gsm8k_5shot"):
            left = candidate[suite]
            old = original(suite)
            reported["continuation"][suite] = left["results"]
            reported["original20B"][suite] = old["results"]
            for name in ("original20B", *BASELINES):
                right = old if name == "original20B" else baseline(name, suite)
                reported[name][suite] = right["results"]
                extended = full["comparisons"][name]["extended"]
                if set(left["samples"]) != set(right["samples"]):
                    raise ValueError(f"Different subject/task sets: {name}/{suite}")
                if suite == "mmlu_5shot":
                    subjects, draws = {}, []
                    for task in sorted(left["samples"]):
                        subjects[task], bootstrap = compare(left, right, task, "acc", rng)
                        draws.append(bootstrap)
                    rows = list(subjects.values())
                    weights = [r["n"] for r in rows]
                    result = aggregate(rows, draws, weights)
                    for data, side in ((left, "ours"), (right, "baseline")):
                        if not np.isclose(data["results"]["mmlu"]["acc,none"], result[side], rtol=0, atol=1e-10):
                            raise ValueError("MMLU group aggregation mismatch")
                    result.update(n=sum(weights), subjects=subjects,
                                  aggregation="sample-weighted; bootstrap stratified by subject")
                    extended[suite] = result
                else:
                    for filter_name in ("strict-match", "flexible-extract"):
                        extended[f"gsm8k_{filter_name}"], _ = compare(left, right, "gsm8k", "exact_match", rng, filter_name)
        full["all_reported_metrics"] = reported
        full["input_file_sha256"] = dict(evidence.files)
    evidence.unchanged()
    return core, full


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("A", "B"))
    parser.add_argument("--phase", choices=("core", "all"), default="all")
    parser.add_argument("--output", type=Path, help="Override the summary destination")
    parser.add_argument("--evaluation-dir", type=Path, help="Default: continuation/20260911-v1/evaluation-{branch}-v1")
    parser.add_argument("--comparison-dir", type=Path, default=OUTPUT)
    parser.add_argument("--final-dir", type=Path, default=ROOT / "evaluations/final")
    args = parser.parse_args(argv)
    directory = args.evaluation_dir or RUN / f"evaluation-{args.branch}-v1"
    if directory.resolve() in (args.comparison_dir.resolve(), args.final_dir.resolve()):
        raise ValueError("Summary output must be separate from frozen comparison/final inputs")
    core, full = analyze(args.branch, directory, args.comparison_dir, args.final_dir, args.phase)
    destination = args.output or directory / ("core-summary.json" if args.phase == "core" else "summary.json")
    outputs = {destination: full or core}
    for path in outputs:
        if any(root.resolve() == path.resolve() or root.resolve() in path.resolve().parents
               for root in (args.comparison_dir, args.final_dir)):
            raise ValueError("Summary output must be separate from frozen comparison/final inputs")
        temporary = path.with_suffix(".json.tmp")
        if path.is_symlink() or temporary.exists() or temporary.is_symlink():
            raise ValueError(f"Unsafe or unfinished summary output: {path}")
        if str(path.resolve()) in (full or core)["input_file_sha256"]:
            raise ValueError(f"Summary would overwrite an input: {path}")
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("branch") != args.branch or previous.get("primary_metrics") != PRIMARY:
                raise ValueError(f"Refusing to overwrite an unrelated file: {path}")
    for path, summary in outputs.items():
        save_json(path, summary)
    print(json.dumps({"branch": args.branch, "status": (full or core)["status"],
                      "outputs": [str(p) for p in outputs], "ours_primary_macro": core["ours_primary_macro"],
                      "identity_status": core["identity_verification"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
