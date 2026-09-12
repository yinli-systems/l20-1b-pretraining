"""Synthetic CPU-only continuation analysis, integrity and CLI regression tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import analyze_continuation_evaluation as analysis
from analyze_comparison import SEEDS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return analysis.sha256(path)


def result(tasks, model, shot=0, revision="main", candidate=False):
    data = {"config": {**SEEDS, "limit": None, "model_dtype": "torch.bfloat16",
                       "batch_size": "auto:4", "model": model, "model_revision": revision,
                       "device": "cuda", "gen_kwargs": None, "use_cache": None},
            "lm_eval_version": "0.4.9", "max_length": 2048,
            **{key: {} for key in ("versions", "n-shot", "configs", "n-samples", "results", "samples")}}
    for task in tasks:
        values = [1, 0, 1, 0] if candidate else [0, 1, 1, 0]
        if task == "boolq" and candidate:
            values = [1, 1, 1, 1]
        if task == "mmlu_small":
            values = [int(candidate)] * 2
        if task == "mmlu_large":
            values = [0] * 6 if candidate else [1, 1, 1, 0, 0, 0]
        data["versions"][task] = 1
        data["n-shot"][task] = shot
        data["configs"][task] = {"dataset_path": "synthetic", "num_fewshot": shot}
        if task == "boolq":
            data["configs"][task]["doc_to_choice"] = ["no", "yes"]
        data["n-samples"][task] = {"original": len(values), "effective": len(values)}
        data["results"][task] = {}
        data["samples"][task] = []
        filters = ("strict-match", "flexible-extract") if task == "gsm8k" else ("none",)
        for filter_name in filters:
            for metric in (("exact_match",) if task == "gsm8k" else ("acc", "acc_norm")):
                data["results"][task][f"{metric},{filter_name}"] = float(np.mean(values))
            for i, value in enumerate(values):
                target = i % 2
                prediction = target if value else 1 - target
                data["samples"][task].append({
                    "doc_id": i, "doc_hash": f"{task}-doc-{i}", "prompt_hash": f"prompt-{i}",
                    "target_hash": f"target-{i}", "filter": filter_name, "target": target,
                    "acc": value, "acc_norm": value, "exact_match": value,
                    "filtered_resps": [[-1.0 if j == prediction else -2.0, False] for j in range(2)],
                })
    if tasks and all(t.startswith("mmlu_") for t in tasks):
        total = sum(data["n-samples"][t]["effective"] for t in tasks)
        data["results"]["mmlu"] = {"acc,none": sum(
            data["results"][t]["acc,none"] * data["n-samples"][t]["effective"] for t in tasks) / total}
    return data


class Bundle:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.final = root / "evaluations/final"
        self.comparison = root / "evaluations/comparison-20260911"
        self.directory = root / "continuation/20260911-v1/evaluation-A-v1"
        self.model = str(self.directory / "hf")
        self.reference = {"status": "complete", "protocol": analysis.PROTOCOL, "jobs": {}}
        self.receipt = {"status": "running", "branch": "A", "protocol": analysis.PROTOCOL, "jobs": {}}
        originals = {}
        for suite, (task_names, shot) in analysis.SUITES.items():
            tasks = ["mmlu_small", "mmlu_large"] if suite == "mmlu_5shot" else task_names
            name = "zero_shot_core" if suite == "core" else suite
            originals[name] = write_json(self.final / f"{name}.json", result(
                [t for t in tasks if t != "boolq"], str(self.final / "hf"), shot))
            self.add_job(self.receipt, self.directory, suite, suite,
                         result(tasks, self.model, shot, candidate=True))
            for baseline, (model, revision) in analysis.BASELINES.items():
                self.add_job(self.reference, self.comparison, f"{baseline}_{suite}", suite,
                             result(tasks, model, shot, revision), model=model, revision=revision)
        self.add_job(self.reference, self.comparison, "ours_boolq", "core",
                     result(["boolq"], str(self.final / "hf")), tasks=["boolq"],
                     model=str(self.final / "hf"))
        self.reference["original_result_sha256"] = originals
        monkeypatch.setattr(analysis, "ORIGINALS", originals)
        monkeypatch.setattr(analysis, "ROOT", root)
        monkeypatch.setattr(analysis, "RUN", self.directory.parent)
        pilot = self.directory.parent / "pilot-A"
        pilot.mkdir(parents=True)
        checkpoint = pilot / "resume.pth"
        checkpoint.write_bytes(b"synthetic checkpoint, never deserialized")
        parent = root / "checkpoints/full/final/lit_model.pth"
        parent.parent.mkdir(parents=True)
        parent.write_bytes(b"synthetic parent")
        identity = {"checkpoint_sha256": analysis.sha256(checkpoint), "parent_sha256": analysis.sha256(parent),
                    "step": 190, "branch_prediction_tokens": 198451200,
                    "data_plan_sha256": write_json(pilot / "data-plan.json", {"branch": "A"})}
        write_json(pilot / "resume.json", {"sha256": identity["checkpoint_sha256"],
                                          "parent_sha256": identity["parent_sha256"], "step": 190})
        hf = self.directory / "hf"
        hf.mkdir()
        (hf / "pytorch_model.bin").write_bytes(b"synthetic export")
        manifest = {"identity": identity, "files": {"pytorch_model.bin": analysis.sha256(hf / "pytorch_model.bin")},
                    "parity": {"passed": True, "rmse": 0.01, "argmax_agreement": 0.99}}
        self.receipt["identity"] = {**identity, "export_manifest_sha256": write_json(hf / "manifest.json", manifest)}
        self.sync()

    def add_job(self, receipt, directory, name, suite, data, tasks=None, model=None, revision=None):
        task_names, shot = analysis.SUITES[suite]
        path = directory / f"{name}.json"
        job = {"name": name, "tasks": tasks if tasks is not None else task_names, "num_fewshot": shot}
        if model is not None:
            job.update(model=model, revision=revision)
        receipt["jobs"][name] = {"status": "complete", "job": job, "result": str(path),
                                 "sha256": write_json(path, data)}

    def sync(self):
        comparison_hash = write_json(self.comparison / "receipt.json", self.reference)
        self.receipt["input_result_sha256"] = {
            str(self.final / f"{name}.json"): digest for name, digest in analysis.ORIGINALS.items()}
        self.receipt["input_result_sha256"].update({r["result"]: r["sha256"] for r in self.reference["jobs"].values()})
        self.receipt["input_result_sha256"][str(self.comparison / "receipt.json")] = comparison_hash
        write_json(self.directory / "receipt.json", self.receipt)

    def change_result(self, name, mutate, reference=False):
        row = (self.reference if reference else self.receipt)["jobs"][name]
        path = Path(row["result"])
        data = json.loads(path.read_text())
        mutate(data)
        row["sha256"] = write_json(path, data)
        self.sync()

    def analyze(self, phase="all"):
        return analysis.analyze("A", self.directory, self.comparison, self.final, phase)

    def args(self, phase):
        return ["--branch", "A", "--phase", phase, "--evaluation-dir", str(self.directory),
                "--comparison-dir", str(self.comparison), "--final-dir", str(self.final)]


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    return Bundle(tmp_path, monkeypatch)


def test_core_running_receipt_is_sufficient_and_preserves_raw_files(bundle):
    bundle.receipt["jobs"]["mmlu_5shot"] = {"status": "running"}
    bundle.receipt["jobs"].pop("gsm8k_5shot")
    bundle.sync()
    before = {p: p.read_bytes() for p in bundle.root.rglob("*") if p.is_file()}
    core, full = bundle.analyze("core")
    assert full is None and core["status"] == "core_complete"
    assert core["suite_status_at_analysis"] == "running"
    assert not core["extended_ready"]
    assert core["uncertainty"]["repetitions"] == 10000
    assert core["uncertainty"]["seed"] == 20260911
    assert "not training-seed" in core["uncertainty"]["scope"]
    assert core["identity_verification"]["status"] == "declared_artifacts_verified"
    assert set(core["comparisons"]) == {"original20B", *analysis.BASELINES}
    for comparison in core["comparisons"].values():
        assert len(comparison["primary"]) == 7
        assert comparison["primary_macro"]["difference_pp"] == pytest.approx(50 / 7)
        assert comparison["six_task_sensitivity_excluding_boolq"]["difference_pp"] == 0
        assert len(comparison["six_task_sensitivity_excluding_boolq"]["paired_95_ci_pp"]) == 2
        assert set(comparison["extended"]) == {"lambada_openai", "truthfulqa_mc2"}
    analysis.main(bundle.args("core"))
    assert (bundle.directory / "core-summary.json").is_file()
    assert not (bundle.directory / "summary.json").exists()
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_full_subject_weighting_filters_and_determinism(bundle):
    first, full = bundle.analyze()
    core, _ = bundle.analyze("core")
    assert first["comparisons"] == core["comparisons"]
    _, repeated = bundle.analyze()
    assert repeated["comparisons"] == full["comparisons"]
    assert full["status"] == "complete"
    assert full["claims"] == {"sota_asserted": False, "automatic_training_promotion": False}
    for comparison in full["comparisons"].values():
        extended = comparison["extended"]
        mmlu = extended["mmlu_5shot"]
        assert mmlu["n"] == 8
        assert mmlu["ours"] == 0.25  # Subject macro would incorrectly give 0.5.
        assert mmlu["baseline"] == 0.375
        assert mmlu["difference_pp"] == -12.5
        assert {"gsm8k_strict-match", "gsm8k_flexible-extract"} <= extended.keys()
        assert len(extended) == 5
    assert "samples" not in full
    analysis.main(bundle.args("all"))
    assert (bundle.directory / "summary.json").is_file()


@pytest.mark.parametrize("field", ["doc_hash", "prompt_hash", "target_hash"])
def test_hash_alignment_rejects_changed_sample(bundle, field):
    bundle.change_result("core", lambda d: d["samples"]["hellaswag"][0].update({field: "changed"}))
    with pytest.raises(ValueError, match="Different"):
        bundle.analyze("core")


@pytest.mark.parametrize("change", ["version", "config", "shot", "global", "seed", "harness", "context", "duplicate", "missing", "mean"])
def test_strict_protocol_and_sample_validation(bundle, change):
    def mutate(d):
        task = "hellaswag"
        if change == "version":
            d["versions"][task] = 999
        elif change == "config":
            d["configs"][task]["dataset_path"] = "different"
        elif change == "shot":
            d["n-shot"][task] = 1
        elif change == "global":
            d["config"]["gen_kwargs"] = {"temperature": 0.5}
        elif change == "seed":
            d["config"]["fewshot_seed"] = 7
        elif change == "harness":
            d["lm_eval_version"] = "0.4.8"
        elif change == "context":
            d["max_length"] = 4096
        elif change == "duplicate":
            d["samples"][task].append(copy.deepcopy(d["samples"][task][0]))
        elif change == "missing":
            d["samples"][task].pop()
        else:
            d["results"][task]["acc_norm,none"] = 0.1
    bundle.change_result("core", mutate)
    with pytest.raises(ValueError):
        bundle.analyze("core")


@pytest.mark.parametrize("target", ["candidate", "input", "manifest", "export", "checkpoint", "plan"])
def test_all_declared_hashes_fail_closed(bundle, target):
    paths = {"candidate": bundle.directory / "core.json", "input": bundle.comparison / "tinyllama_3t_gsm8k_5shot.json",
             "manifest": bundle.directory / "hf/manifest.json", "export": bundle.directory / "hf/pytorch_model.bin",
             "checkpoint": bundle.directory.parent / "pilot-A/resume.pth", "plan": bundle.directory.parent / "pilot-A/data-plan.json"}
    paths[target].write_bytes(paths[target].read_bytes() + b" ")
    with pytest.raises(ValueError, match="Hash mismatch"):
        bundle.analyze("core")


@pytest.mark.parametrize("change", ["tasks", "fewshot", "name", "protocol", "branch", "revision", "model"])
def test_receipt_job_and_pinned_model_contract(bundle, change):
    if change == "tasks":
        bundle.receipt["jobs"]["core"]["job"]["tasks"] = list(analysis.PRIMARY)
    elif change == "fewshot":
        bundle.receipt["jobs"]["core"]["job"]["num_fewshot"] = 5
    elif change == "name":
        bundle.receipt["jobs"]["core"]["job"]["name"] = "other"
    elif change == "protocol":
        bundle.receipt["protocol"] = {**analysis.PROTOCOL, "limit": 10}
    elif change == "branch":
        bundle.receipt["branch"] = "B"
    elif change == "revision":
        bundle.change_result("tinyllama_3t_core", lambda d: d["config"].update(model_revision="main"), reference=True)
    else:
        bundle.change_result("core", lambda d: d["config"].update(model="other-export"))
    bundle.sync()
    with pytest.raises(ValueError):
        bundle.analyze("core")


@pytest.mark.parametrize("change", ["identity", "parity", "basename"])
def test_manifest_contract_beyond_hash(bundle, change):
    path = bundle.directory / "hf/manifest.json"
    manifest = json.loads(path.read_text())
    if change == "identity":
        manifest["identity"]["step"] = 191
    elif change == "parity":
        manifest["parity"]["passed"] = False
    else:
        manifest["files"] = {"../escape": "a" * 64}
    bundle.receipt["identity"]["export_manifest_sha256"] = write_json(path, manifest)
    bundle.sync()
    with pytest.raises(ValueError):
        bundle.analyze("core")


def test_evolving_identity_exposes_gaps_without_inventing_verification(bundle):
    bundle.receipt["identity"] = {"future_identity": {"version": 2}}
    bundle.sync()
    core, _ = bundle.analyze("core")
    identity = core["identity_verification"]
    assert identity["status"] == "partial"
    assert identity["unverified"]
    assert identity["receipt_identity"]["future_identity"] == {"version": 2}


@pytest.mark.parametrize("change", ["missing_subject", "aggregate", "gsm_filter", "mixed_model"])
def test_extended_validation(bundle, change):
    if change == "missing_subject":
        bundle.change_result("mmlu_5shot", lambda d: d["samples"].pop("mmlu_large"))
    elif change == "aggregate":
        bundle.change_result("mmlu_5shot", lambda d: d["results"]["mmlu"].update({"acc,none": 0.5}))
    elif change == "gsm_filter":
        bundle.change_result("gsm8k_5shot", lambda d: d["samples"].update(
            gsm8k=[s for s in d["samples"]["gsm8k"] if s["filter"] != "strict-match"]))
    else:
        bundle.change_result("gsm8k_5shot", lambda d: d["config"].update(model="different"))
    with pytest.raises(ValueError):
        bundle.analyze()


def test_all_requires_extended_and_core_requires_all_nine_tasks(bundle):
    bundle.receipt["jobs"]["gsm8k_5shot"]["status"] = "running"
    bundle.sync()
    with pytest.raises(ValueError, match="Required suites"):
        bundle.analyze()
    bundle.change_result("core", lambda d: d["samples"].pop("truthfulqa_mc2"))
    with pytest.raises(ValueError, match="Wrong task set"):
        bundle.analyze("core")


def test_cli_output_override_and_frozen_file_protection(bundle, tmp_path):
    destination = tmp_path / "custom-summary.json"
    analysis.main(bundle.args("core") + ["--output", str(destination)])
    assert json.loads(destination.read_text())["status"] == "core_complete"
    for output in (bundle.final / "zero_shot_core.json", bundle.directory / "core.json"):
        before = output.read_bytes()
        with pytest.raises(ValueError, match="frozen|overwrite"):
            analysis.main(bundle.args("core") + ["--output", str(output)])
        assert output.read_bytes() == before


def test_cli_branch_default_directory_and_phase(monkeypatch, tmp_path):
    captured = []
    def fake_analyze(branch, directory, comparison, final, phase):
        captured.append((branch, directory, phase))
        raise RuntimeError("captured")
    monkeypatch.setattr(analysis, "RUN", tmp_path)
    monkeypatch.setattr(analysis, "analyze", fake_analyze)
    with pytest.raises(RuntimeError, match="captured"):
        analysis.main(["--branch", "B"])
    assert captured == [("B", tmp_path / "evaluation-B-v1", "all")]


def test_input_race_is_detected(tmp_path):
    path = tmp_path / "input.json"
    write_json(path, {"value": 1})
    evidence = analysis.Evidence()
    evidence.read(path)
    write_json(path, {"value": 2})
    with pytest.raises(ValueError, match="Input changed"):
        evidence.unchanged()
