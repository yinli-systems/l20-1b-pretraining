# Reproducibility

## 1. Fast, no-GPU checks

```bash
python3 tools/verify_repository.py
python3 pretraining/reproducibility/recompute.py
```

The repository verifier checks the relocation manifest for all previously
tracked files, published result/source hashes, paired per-image score counts,
question totals, before/after means, retention-stop arithmetic and active
documentation links. It does not rerun model inference or certify dataset
independence. The confidence intervals are the frozen runner's published
results; checking their consistency is not a new independent multi-seed study.

The second command invokes the original, byte-preserved language evidence
verifier against its self-contained directory. Its manifest and source hashes
have not been rewritten to accommodate the new layout.

## 2. Regression suite and original interval replay

Use Python 3.12 or later in an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r pretraining/requirements-test.txt
python -m pytest -q
python pretraining/reproducibility/recompute_efficiency_ci.py
```

The original interval replay requires the pinned NumPy version. It replays
published language-model paired sufficient statistics; it is not a fresh
multimodal prediction run. CI runs these checks and the repository verifier.

## 3. Latest multimodal package

Start with [metrics.json](../results/2026-09-28/metrics.json), then inspect
[source-manifest.json](../results/2026-09-28/source-manifest.json),
[aggregate receipts](../results/2026-09-28/evidence/) and
[paired image-cluster counts](../results/2026-09-28/scores/).

Before/after row alignment was checked during the cluster export. The public
CSV files retain per-image question counts and summed scores; these reproduce
headline means and paired mean changes without releasing source text. They do
not independently establish that each underlying model answer was scored
correctly. Raw inference outputs and model weights remain outside Git.

## 4. Training reproduction is a different task

Language training scripts and their operating assumptions are preserved under
[pretraining/](../pretraining/README.md). They require the original pinned data,
runtime dependencies, GPU resources and stage admission checks.

The newest VLM campaigns also depend on machine-local code/data/checkpoints
not fully packaged here. This cleanup does not turn an evidence archive into a
turnkey VLM training release. No GPU experiment was rerun merely to reorganize
the repository.

## Language-model README and figure

`python3 tools/build_language_showcase.py --check` verifies the peer table and
figure hashes against the unchanged frozen language evidence. To regenerate:

```bash
python3 tools/build_language_showcase.py
python3 -m pip install matplotlib
python3 tools/plot_language_comparison.py
```

This regenerates a presentation of stored results, not model predictions.
