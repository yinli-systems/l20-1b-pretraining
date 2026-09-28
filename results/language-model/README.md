# Language-model comparison

**1.100B parameters · 19.9997B prediction tokens · one NVIDIA L20**

This presentation is generated from the unchanged September 12, 2026 frozen
study. It adds no new model inference or leaderboard claims.

## At a glance

| Model / checkpoint | Parameters | Training tokens | 7-task ↑ | 6-task ↑ |
|---|---:|---:|---:|---:|
| **L20-1B · ours** | **1.10B** | **20B** | **51.26** | **49.94** |
| DataDecide · QC7/FW3 · 18B | 1.28B | 18.02B | 52.49 | 51.25 |
| TinyLlama · 21B | 1.10B | 21B | 43.64 | 41.18 |
| WebOrganizer · domain mix | 1.44B | 28.8B | 55.16 | 54.67 |
| Phi-1.5 (synthetic-data reference) | 1.42B | 150B | 65.03 | 63.41 |
| OPT-1.3B | 1.32B | 180B | 51.00 | 49.90 |
| Pythia-1B | 1.01B | 300B | 48.10 | 46.01 |
| Falcon-RW-1B | 1.31B | 350B | 54.99 | 53.82 |
| TinyLlama · 1T | 1.10B | 1T | 50.24 | 48.70 |
| TinyLlama · 1.5T | 1.10B | 1.5T | 51.17 | 49.94 |
| TinyLlama · 2T | 1.10B | 2T | 51.56 | 49.64 |
| TinyLlama · 2.5T | 1.10B | 2.5T | 53.89 | 52.38 |

## Paired comparisons that support the headline

| Baseline | L20 minus baseline, pp | Paired 95% interval, pp | Interpretation |
|---|---:|---:|---|
| TinyLlama · 21B | +7.62 | [+6.68, +8.55] | Higher in this protocol |
| Pythia-1B | +3.16 | [+2.30, +4.02] | Higher in this protocol |
| OPT-1.3B | +0.26 | [-0.64, +1.15] | Interval spans zero; not proof of equivalence |
| TinyLlama · 1T | +1.02 | [+0.16, +1.89] | Higher in this protocol |
| TinyLlama · 1.5T | +0.09 | [-0.78, +0.97] | Interval spans zero; not proof of equivalence |
| TinyLlama · 2T | -0.30 | [-1.20, +0.59] | Interval spans zero; not proof of equivalence |
| TinyLlama · 2.5T | -2.63 | [-3.49, -1.80] | Lower in this protocol |
| DataDecide · QC7/FW3 · 18B | -1.22 | [-2.05, -0.38] | Lower in this protocol |
| Phi-1.5 (synthetic-data reference) | -13.77 | [-14.78, -12.78] | Lower in this protocol |

The 20B-versus-1T comparison is a **50× difference in reported token exposure**,
not proof of 50× faster training, 50× lower total cost, or equal general capability.
The model also loses to stronger candidates; these are included below.

## All 36 frozen baselines

Training compute is a **6ND approximation**, relative to L20-1B's 20B run.
It excludes data curation/classifier/teacher and experiment-search compute.
Parameter counts are measured totals, which can differ from rounded model names.

| Checkpoint | Parameters | Training tokens | 6ND / ours | 7-task | 6-task | L20−baseline 7-task, pp | Paired 95% interval, pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pythia_1b` | 1.0118B | 300B | 13.797× | 48.1042 | 46.0145 | +3.1559 | [+2.3023, +4.0201] |
| `dd_dclm_baseline_qc_20p_10000` | 1.2799B | 14.42B | 0.839× | 49.8436 | 49.7004 | +1.4165 | [+0.5600, +2.2821] |
| `dd_dclm_baseline_qc_20p_12500` | 1.2799B | 18.02B | 1.048× | 51.3832 | 50.8951 | -0.1231 | [-0.9789, +0.7016] |
| `dd_dclm_baseline_qc_20p_15000` | 1.2799B | 21.63B | 1.258× | 52.0216 | 51.1812 | -0.7615 | [-1.6136, +0.0841] |
| `tinyllama_early_10b` | 1.1000B | 10B | 0.500× | 40.7694 | 39.5623 | +10.4907 | [+9.4971, +11.4444] |
| `tinyllama_early_21b` | 1.1000B | 21B | 1.050× | 43.6388 | 41.1769 | +7.6214 | [+6.6839, +8.5497] |
| `weborganizer_dclm` | 1.4399B | 28.8B | 1.885× | 54.4443 | 53.5132 | -3.1841 | [-4.0271, -2.3442] |
| `dd_fineweb_edu_12500` | 1.2799B | 18.02B | 1.048× | 51.1991 | 49.8495 | +0.0610 | [-0.7498, +0.8739] |
| `dd_fineweb_edu_15000` | 1.2799B | 21.63B | 1.258× | 52.0323 | 50.6279 | -0.7722 | [-1.5886, +0.0389] |
| `dd_dclm_baseline_qc_20p_20000` | 1.2799B | 28.84B | 1.677× | 52.0780 | 51.8280 | -0.8179 | [-1.6835, +0.0779] |
| `dd_dclm_baseline_qc_20p_42500` | 1.2799B | 61.28B | 3.565× | 56.7062 | 55.5201 | -5.4461 | [-6.2859, -4.6139] |
| `dd_fineweb_edu_10000` | 1.2799B | 14.42B | 0.839× | 50.8285 | 49.4528 | +0.4316 | [-0.3890, +1.2618] |
| `dd_fineweb_edu_20000` | 1.2799B | 28.84B | 1.677× | 52.4260 | 51.0261 | -1.1659 | [-2.0042, -0.3349] |
| `dd_fineweb_edu_42500` | 1.2799B | 61.28B | 3.565× | 55.0719 | 54.3117 | -3.8117 | [-4.6517, -2.9758] |
| `dd_dclm_baseline_10000` | 1.2799B | 14.42B | 0.839× | 49.0624 | 47.7951 | +2.1977 | [+1.3377, +3.0716] |
| `dd_dclm_baseline_12500` | 1.2799B | 18.02B | 1.048× | 50.6999 | 49.1499 | +0.5602 | [-0.2778, +1.3983] |
| `dd_dclm_baseline_15000` | 1.2799B | 21.63B | 1.258× | 51.2272 | 49.7549 | +0.0329 | [-0.7922, +0.8773] |
| `dd_dclm_baseline_20000` | 1.2799B | 28.84B | 1.677× | 51.4425 | 50.2507 | -0.1823 | [-1.0358, +0.6769] |
| `dd_dclm_baseline_42500` | 1.2799B | 61.28B | 3.565× | 55.0005 | 54.0092 | -3.7403 | [-4.5965, -2.8994] |
| `dd_dclm_baseline_qc_7p_fw3_10000` | 1.2799B | 14.42B | 0.839× | 52.0109 | 50.9801 | -0.7507 | [-1.6156, +0.1162] |
| `dd_dclm_baseline_qc_7p_fw3_12500` | 1.2799B | 18.02B | 1.048× | 52.4851 | 51.2479 | -1.2250 | [-2.0482, -0.3833] |
| `dd_dclm_baseline_qc_7p_fw3_15000` | 1.2799B | 21.63B | 1.258× | 52.8554 | 52.1386 | -1.5952 | [-2.4748, -0.7201] |
| `dd_dclm_baseline_qc_7p_fw3_20000` | 1.2799B | 28.84B | 1.677× | 53.7037 | 52.6136 | -2.4436 | [-3.2783, -1.6001] |
| `dd_dclm_baseline_qc_7p_fw3_42500` | 1.2799B | 61.28B | 3.565× | 56.3681 | 55.4468 | -5.1080 | [-5.9804, -4.2461] |
| `weborganizer_domain_mix` | 1.4399B | 28.8B | 1.885× | 55.1564 | 54.6652 | -3.8963 | [-4.7030, -3.0804] |
| `phi_1_5` | 1.4183B | 150B | 9.670× | 65.0262 | 63.4123 | -13.7660 | [-14.7773, -12.7804] |
| `opt_1_3b` | 1.3158B | 180B | 10.765× | 50.9955 | 49.8974 | +0.2646 | [-0.6409, +1.1547] |
| `falcon_rw_1b` | 1.3116B | 350B | 20.866× | 54.9939 | 53.8181 | -3.7338 | [-4.6379, -2.8591] |
| `tinyllama_early_31b` | 1.1000B | 31B | 1.550× | 44.0794 | 41.5432 | +7.1807 | [+6.2775, +8.0684] |
| `tinyllama_early_63b` | 1.1000B | 63B | 3.150× | 45.5028 | 43.3211 | +5.7573 | [+4.8572, +6.6346] |
| `tinyllama_early_105b` | 1.1000B | 105B | 5.250× | 46.1347 | 43.8850 | +5.1254 | [+4.2612, +5.9952] |
| `tinyllama_503b` | 1.1000B | 503B | 25.150× | 48.2627 | 46.8875 | +2.9974 | [+2.1388, +3.8651] |
| `tinyllama_1t` | 1.1000B | 1T | 50.001× | 50.2420 | 48.7023 | +1.0182 | [+0.1584, +1.8860] |
| `tinyllama_1_5t` | 1.1000B | 1.5T | 75.001× | 51.1691 | 49.9369 | +0.0910 | [-0.7771, +0.9745] |
| `tinyllama_2t` | 1.1000B | 2T | 100.001× | 51.5620 | 49.6358 | -0.3019 | [-1.1978, +0.5871] |
| `tinyllama_2_5t` | 1.1000B | 2.5T | 125.002× | 53.8918 | 52.3845 | -2.6317 | [-3.4891, -1.8006] |

## Task-level view of the same comparison

No task is newly selected for this display. The seven-task suite was already
fixed: HellaSwag, PIQA, WinoGrande, OpenBookQA, ARC-Easy, ARC-Challenge and BoolQ.
Six-task sensitivity removes BoolQ, which was not explicitly decontaminated
from the original corpus. Task-specific metrics are the original acc/acc_norm.

| Model | HellaSwag | PIQA | WinoGrande | OpenBookQA | ARC-Easy | ARC-Challenge | BoolQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **L20-1B** | **45.13** | **69.59** | **52.17** | **35.00** | **64.14** | **33.62** | **59.17** |
| DataDecide · QC7/FW3 · 18B | 45.03 | 67.03 | 51.38 | 38.40 | 67.68 | 37.97 | 59.91 |
| TinyLlama · 21B | 35.85 | 63.55 | 52.33 | 30.20 | 41.25 | 23.89 | 58.41 |
| WebOrganizer · domain mix | 61.62 | 76.99 | 56.51 | 35.60 | 62.29 | 34.98 | 58.10 |
| Phi-1.5 (synthetic-data reference) | 62.51 | 76.06 | 72.69 | 48.20 | 73.23 | 47.78 | 74.71 |
| OPT-1.3B | 53.76 | 72.31 | 59.12 | 33.20 | 51.30 | 29.69 | 57.58 |
| Pythia-1B | 47.17 | 69.42 | 52.57 | 31.40 | 48.91 | 26.62 | 60.64 |
| Falcon-RW-1B | 61.58 | 74.97 | 61.01 | 35.60 | 57.49 | 32.25 | 62.05 |
| TinyLlama · 1T | 52.59 | 69.86 | 56.35 | 33.20 | 52.57 | 27.65 | 59.48 |
| TinyLlama · 1.5T | 53.55 | 71.65 | 58.41 | 35.20 | 52.06 | 28.75 | 58.56 |
| TinyLlama · 2T | 54.59 | 70.57 | 56.51 | 33.40 | 54.42 | 28.33 | 63.12 |
| TinyLlama · 2.5T | 58.89 | 73.18 | 58.88 | 34.80 | 56.90 | 31.66 | 62.94 |

## Protocol and provenance

All numbers above are **our same-protocol re-evaluations**, not scores copied
from model cards: lm-eval 0.4.9, BF16, zero-shot, context 2,048, batch auto:4,
seeds 42/42/42/1234, no chat template. Training recipes and tokenizers differ.
Intervals cover benchmark-item sampling only, not training-seed variation, and
are not adjusted for multiple comparisons. Full raw model predictions are not
in the compact Git bundle; reproduction of aggregate evidence is not fresh inference.

- [Machine-readable presentation](comparison.json)
- [Original complete study and 252 task comparisons](../../pretraining/reports/metrics/efficiency-all-results-final-20260912.json)
- [Original report and claim boundaries](../../pretraining/reports/metrics/efficiency-all-results-final-20260912.md)
- [Language model card](../../pretraining/MODEL_CARD.md)
- [Reproduction instructions](../../docs/reproducibility.md)
- [Naming and presentation research](../../docs/presentation-research-2026-09-28.md)

Regenerate the summary with `python3 tools/build_language_showcase.py`.
Regenerate the figure with `python3 tools/plot_language_comparison.py` after
installing Matplotlib. `--check` validates the checked-in presentation and figure
hashes without Matplotlib or a GPU.
