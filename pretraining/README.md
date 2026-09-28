# Language pretraining

**L20-1B-20B-Base** is a 1.100B-parameter English language model with a 32K
byte-level BPE tokenizer, both trained from random initialization on one L20.

[Model card](MODEL_CARD.md) · [Weights](https://huggingface.co/AliceYin/L20-1B-20B-Base) · [Reports](reports/README.md) · [Independent verification](reproducibility/README.md) · [Historical overview](HISTORY.md)

| Measurement | Result |
|---|---:|
| Prediction tokens | 19,999,703,040 |
| Optimizer updates | 19,148 |
| Seven-task zero-shot macro | 51.2601% |
| Six-task macro excluding BoolQ | 49.9411% |
| Final held-out loss | 2.4247146 |
| Final perplexity | 11.2990036 |

These values belong to the **language model**, not the multimodal extension.
Full metric definitions, contamination caveats and both comparison wins and
losses are in the model card and immutable reports.

## Verify the original evidence

From the repository root:

```bash
python3 pretraining/reproducibility/recompute.py
python3 -m pip install -r pretraining/requirements-test.txt
python3 pretraining/reproducibility/recompute_efficiency_ci.py
python3 -m pytest -q
```

The numerical interval replay requires the pinned NumPy version. Replaying
stored evidence does not regenerate training or raw model predictions.

## Work with the training code

```bash
cd pretraining
python run_pretrain.py --help
python build_data.py --help
```

Actual training requires the historical runtime dependencies and an explicitly
prepared GPU/data environment; test dependencies alone are not a training
installation. Operational scripts retain original machine defaults and stage
admission checks. Read [HISTORY.md](HISTORY.md) and the execution protocols before
running them. Do not launch supervisors or cleanup scripts as smoke tests.

## Why this directory is self-contained

The original scripts, tests, reports, configs, artifacts and reproduction
bundle moved together so their relative paths and evidence hashes remain
unchanged. The repository root now separates released language work from
current multimodal results. See the [layout guide](../docs/repository-layout.md).
