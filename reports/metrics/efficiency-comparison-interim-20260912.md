# 1.1B compute-efficiency comparison: interim matched-protocol results

## Protocol and interpretation

All values below were recomputed from the remote per-example `lm_eval==0.4.9`
outputs under the same seven-task protocol: zero-shot, BF16, context 2048,
`batch_size=auto:4`, no chat template, fixed seeds 42/42/42/1234. The primary
metric is `acc_norm` except Winogrande and BoolQ, which use `acc`. Task scores
are equally weighted.

Differences are **ours minus baseline**, in percentage points. The 95% intervals
come from 10,000 paired, task-stratified bootstrap draws with seed 20260912. An
interval wholly above zero is a statistically significant win; wholly below
zero is a statistically significant loss; an interval containing zero is not
distinguishable under this interval. These intervals capture test-item sampling
uncertainty for fixed checkpoints, not training-seed variance. They are not
adjusted for multiple comparisons. BoolQ was not explicitly decontaminated from
the original 20B corpus, so the six-task sensitivity result is required.

## Macro results

| Baseline checkpoint | Parameters | Training tokens | Compute vs ours (6ND ratio) | Baseline 7-task | Ours−baseline, 95% CI | Inference | Baseline 6-task excl. BoolQ | Ours−baseline, 95% CI | Inference |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| TinyLlama, step-5k-token-10B | 1,100,048,384 | 10.000B nominal | 0.5000× | 40.7694 | **+10.4907 [+9.4971, +11.4444]** | significant win | 39.5623 | **+10.3788 [+9.2806, +11.4315]** | significant win |
| TinyLlama, step-10k-token-21B | 1,100,048,384 | 21.000B nominal | 1.0500× | 43.6388 | **+7.6214 [+6.6839, +8.5497]** | significant win | 41.1769 | **+8.7642 [+7.7160, +9.7908]** | significant win |
| Pythia-1B | 1,011,781,632 | 300.000B | 13.797× | 48.1042 | **+3.1559 [+2.3023, +4.0201]** | significant win | 46.0145 | **+3.9266 [+2.9813, +4.8736]** | significant win |
| DataDecide DCLM QC20, step 10k | 1,279,854,592 | 14.41792B | 0.8387× | 49.8436 | **+1.4165 [+0.5600, +2.2821]** | significant win | 49.7004 | +0.2407 [−0.6803, +1.1675] | inconclusive |
| DataDecide DCLM QC20, step 12.5k | 1,279,854,592 | 18.02240B | 1.0484× | 51.3832 | −0.1231 [−0.9789, +0.7016] | inconclusive | 50.8951 | **−0.9540 [−1.8819, −0.0504]** | significant loss |
| DataDecide DCLM QC20, step 15k | 1,279,854,592 | 21.62688B | 1.2581× | 52.0216 | −0.7615 [−1.6136, +0.0841] | inconclusive | 51.1812 | **−1.2401 [−2.1786, −0.3089]** | significant loss |

Our fixed checkpoint is 1,100,048,384 parameters and 19,999,703,040 training
tokens. Its seven-task macro is 51.2601%; the six-task value is 49.9411%.

## Per-task results

| Baseline | Task | Ours | Baseline | Difference pp | Paired 95% CI pp | Inference |
|---|---|---:|---:|---:|---:|---|
| TinyLlama 10B | HellaSwag | 45.1305 | 33.1109 | +12.0195 | [+11.2229, +12.8164] | win |
| TinyLlama 10B | PIQA | 69.5865 | 61.6431 | +7.9434 | [+6.0936, +9.7933] | win |
| TinyLlama 10B | Winogrande | 52.1705 | 51.3023 | +0.8682 | [−2.6835, +4.2620] | inconclusive |
| TinyLlama 10B | OpenBookQA | 35.0000 | 30.2000 | +4.8000 | [+1.0000, +8.4000] | win |
| TinyLlama 10B | ARC-Easy | 64.1414 | 38.7626 | +25.3788 | [+23.2744, +27.4832] | win |
| TinyLlama 10B | ARC-Challenge | 33.6177 | 22.3549 | +11.2628 | [+8.7031, +13.8225] | win |
| TinyLlama 10B | BoolQ | 59.1743 | 48.0122 | +11.1621 | [+9.0520, +13.2722] | win, contamination caveat |
| TinyLlama 21B | HellaSwag | 45.1305 | 35.8494 | +9.2810 | [+8.5342, +10.0279] | win |
| TinyLlama 21B | PIQA | 69.5865 | 63.5473 | +6.0392 | [+4.2437, +7.8346] | win |
| TinyLlama 21B | Winogrande | 52.1705 | 52.3283 | −0.1579 | [−3.4728, +3.2360] | inconclusive |
| TinyLlama 21B | OpenBookQA | 35.0000 | 30.2000 | +4.8000 | [+1.2000, +8.4000] | win |
| TinyLlama 21B | ARC-Easy | 64.1414 | 41.2458 | +22.8956 | [+20.7912, +24.9579] | win |
| TinyLlama 21B | ARC-Challenge | 33.6177 | 23.8908 | +9.7270 | [+7.1672, +12.3720] | win |
| TinyLlama 21B | BoolQ | 59.1743 | 58.4098 | +0.7645 | [−0.9480, +2.5076] | inconclusive, contamination caveat |
| Pythia | HellaSwag | 45.1305 | 47.1719 | −2.0414 | [−2.7186, −1.3643] | loss |
| Pythia | PIQA | 69.5865 | 69.4233 | +0.1632 | [−1.4146, +1.7410] | inconclusive |
| Pythia | Winogrande | 52.1705 | 52.5651 | −0.3946 | [−3.4747, +2.7624] | inconclusive |
| Pythia | OpenBookQA | 35.0000 | 31.4000 | +3.6000 | [+0.2000, +7.0000] | win |
| Pythia | ARC-Easy | 64.1414 | 48.9057 | +15.2357 | [+13.3838, +17.0875] | win |
| Pythia | ARC-Challenge | 33.6177 | 26.6212 | +6.9966 | [+4.5222, +9.5563] | win |
| Pythia | BoolQ | 59.1743 | 60.6422 | −1.4679 | [−3.3028, +0.3670] | inconclusive |
| DD QC20 10k | HellaSwag | 45.1305 | 48.4864 | −3.3559 | [−4.0430, −2.6688] | loss |
| DD QC20 10k | PIQA | 69.5865 | 68.9880 | +0.5985 | [−0.9249, +2.1219] | inconclusive |
| DD QC20 10k | Winogrande | 52.1705 | 54.9329 | −2.7624 | [−6.0773, +0.4736] | inconclusive |
| DD QC20 10k | OpenBookQA | 35.0000 | 34.2000 | +0.8000 | [−2.4000, +3.8000] | inconclusive |
| DD QC20 10k | ARC-Easy | 64.1414 | 59.4276 | +4.7138 | [+3.0724, +6.3552] | win |
| DD QC20 10k | ARC-Challenge | 33.6177 | 32.1672 | +1.4505 | [−0.7679, +3.6689] | inconclusive |
| DD QC20 10k | BoolQ | 59.1743 | 50.7034 | +8.4709 | [+6.1774, +10.7951] | win, contamination caveat |
| DD QC20 12.5k | HellaSwag | 45.1305 | 49.0042 | −3.8737 | [−4.5509, −3.2065] | loss |
| DD QC20 12.5k | PIQA | 69.5865 | 70.1306 | −0.5441 | [−2.1219, +1.0337] | inconclusive |
| DD QC20 12.5k | Winogrande | 52.1705 | 55.8800 | −3.7096 | [−7.0245, −0.4736] | loss |
| DD QC20 12.5k | OpenBookQA | 35.0000 | 35.6000 | −0.6000 | [−3.6000, +2.4000] | inconclusive |
| DD QC20 12.5k | ARC-Easy | 64.1414 | 62.2475 | +1.8939 | [+0.2525, +3.5354] | win |
| DD QC20 12.5k | ARC-Challenge | 33.6177 | 32.5085 | +1.1092 | [−1.1092, +3.2423] | inconclusive |
| DD QC20 12.5k | BoolQ | 59.1743 | 54.3119 | +4.8624 | [+2.7217, +7.0948] | win, contamination caveat |
| DD QC20 15k | HellaSwag | 45.1305 | 51.2149 | −6.0844 | [−6.7815, −5.4070] | loss |
| DD QC20 15k | PIQA | 69.5865 | 71.0011 | −1.4146 | [−2.9924, +0.1632] | inconclusive |
| DD QC20 15k | Winogrande | 52.1705 | 54.9329 | −2.7624 | [−6.1563, +0.5525] | inconclusive |
| DD QC20 15k | OpenBookQA | 35.0000 | 34.8000 | +0.2000 | [−3.0000, +3.4000] | inconclusive |
| DD QC20 15k | ARC-Easy | 64.1414 | 62.3737 | +1.7677 | [+0.2104, +3.4091] | win |
| DD QC20 15k | ARC-Challenge | 33.6177 | 32.7645 | +0.8532 | [−1.2799, +3.0717] | inconclusive |
| DD QC20 15k | BoolQ | 59.1743 | 57.0642 | +2.1101 | [−0.0612, +4.3425] | inconclusive |

## Evidence identity and current boundary

The accepted raw result SHA-256 values are:

- Pythia: `8afa2688f5d821ce908969c43bd536d97156abac07794e1ed32706ad7fadda32`
- TinyLlama 10B: `3ba80c8450665e358c5c06319a5c2242af97a7c6f86156559a56b44fb41a29b7`
- TinyLlama 21B: `648d27d1c01e092fbef23e73439baaaaa26489ad8c343661afc157e581a05b89`
- DataDecide 10k: `fff72a35d3897a8150d5aeb82e7273b7515b4031e50143aba55ae1c78d277cc0`
- DataDecide 12.5k: `64591a298c8a71481b53af8e0605ed9ef65cf97b9b225784f85654d77ccdf64a`
- DataDecide 15k: `9235e2c80ccf335513fe90b8b509b04bd4bb2758a2b7cf6594f18cbd14f8f827`

These are interim results: 6 of 36 declared new checkpoint evaluations are
complete. TinyLlama 10B was loaded only after an exact-hash-bound, statically
audited conversion in a networkless and capability-dropped container; tensor
equality was verified before evaluation. The same procedure was applied to the
TinyLlama 21B checkpoint, and its conversion receipt was archived before the
rebuildable weights were removed. The queue is now processing WebOrganizer
DCLM. No failed checkpoint score is included here, and the candidate set is not
a global census of all 1B models.
