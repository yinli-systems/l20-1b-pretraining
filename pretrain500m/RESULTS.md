# 529M experiment results

Status date: 2026-09-15 UTC.

## Immutable parent

| Item | Value |
| --- | ---: |
| Parameters | 528,748,800 |
| Parent step | 7,629 |
| Parent prediction tokens | 15,999,172,608 |
| Parent checkpoint SHA-256 | `13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf` |

## Data pipeline

The three-tranche raw audit covered 199,407 rows from 16 source categories and
assigned 199,372 byte-identical-unique documents. Physical row groups were
disjoint and bound input hashes were stable. Quality and family feature gates
retained 180,758 rows before final eligible deduplication and reserves. The
packed selected-source receipt covers 180,637 documents with separate train,
development, and confirmation streams.

The recipes combine general web, knowledge/PDF, math, five programming
languages, and six non-English language strata. Synthetic Cosmopedia data was
excluded from the frozen comparison because its recorded seed-source label was
not sufficient parent-document identity.

Key receipts:

- [three-tranche raw audit](reports/raw-intake-audit-three-tranche-20260913.json)
- [quality and family features](reports/quality-family-v1-completed-20260913.json)
- [selected-source packing](reports/pack-selected-v1-completed-20260914.json)
- [frozen screen selection](reports/frozen-two-seed-screen-selection-v1.json)

## Distributed-system qualification

The four-GPU recovery test reproduced model, optimizer, reader cursor, origin,
run fingerprint, and every rank's RNG state exactly across a process restart.
Its two branches achieved final 10-step median MFU of 0.77514 and 0.77531.

| GPUs | Nodes | Median tokens/s | Steady MFU | Scope |
| ---: | ---: | ---: | ---: | --- |
| 4 | 1 | about 162,000 | about 0.775 | long confirmation configuration |
| 8 | 2 | 290,323 | 0.6912 | throughput qualification |
| 16 | 4 | 467,716 | 0.5567 | throughput qualification |

The 8- and 16-GPU measurements are scaling qualifications. They do not establish
quality improvements.

## Frozen two-seed screen

Each screen run used 536,870,912 prediction tokens. The selection rule was the
lowest worst-seed equal-domain development loss, followed by the two-seed mean
and recipe ID.

| Recipe | Mean equal-domain loss | Worst seed | Seed spread |
| --- | ---: | ---: | ---: |
| F0 existing-web control | 3.312786 | 3.315484 | 0.005397 |
| F1 broad English | 2.702542 | 2.703114 | 0.001144 |
| F2 reasoning | 2.689510 | 2.689540 | 0.000059 |
| F3 broad multilingual | **2.571669** | **2.572069** | 0.000799 |

F3 reduced the screen's mean equal-domain loss by 22.37% relative to F0 and
advanced to confirmation. This result is a development-loss selection result.

## Long confirmation training

F2 and F3 were each run with two seeds. Every run completed 1,024 steps and
2,147,483,648 prediction tokens on four distinct RTX 5090 GPUs. The four runs
total 8,589,934,592 prediction tokens.

| Recipe | Seed | Slurm job | Final 10-step MFU | End-of-run equal-domain validation loss | Checkpoint SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| F2 reasoning | 20260914 | 1589567 | 0.766436 | 2.542319 | `69d9d8d3358f23815f2455f7e075645056f90fe01ed40a11525130a0e58ca085` |
| F2 reasoning | 20260915 | 1590004 | 0.766533 | 2.546584 | `f6a4ff43aeeb892ce7dd3dd7ed69f022cef6af688f21b3f005f67850f6041cbc` |
| F3 broad multilingual | 20260914 | 1590005 | 0.775366 | 2.378190 | `194e1da1ba1e74a77e28ae207d1dd74a4baa6889408e41c6dce36d551f1e222e` |
| F3 broad multilingual | 20260915 | 1589570 | 0.776009 | 2.379313 | `8ee551307118a52bc79e9268ce3a08be20c6776234051fc48e3192d5d3e09a21` |

The full machine-readable receipt is
[long-confirmation-training-20260914.json](reports/long-confirmation-training-20260914.json).

## Frozen held-out loss confirmation

Slurm job 1590660 evaluated the immutable Base and the exact F2/F3 two-seed
grid on 16,678,912 prediction tokens from the five frozen domains. The result
and selection receipts have SHA-256 values
`32d19eae4cf49b522a954bd9ab083466058f5386ed177ddf009c96da0bb070ef`
and `478086a7661899be25c7ff1843f1feb91700c6fb40be26d89ff8750a888409bc`.

| Model family | Two-seed mean equal-domain loss | Relative reduction versus Base |
| --- | ---: | ---: |
| Base | 3.304217 | -- |
| F2 reasoning | 2.567941 | 22.28% |
| F3 broad multilingual | **2.402608** | **27.29%** |

F3's mean loss was 6.44% below F2. F2 was slightly better on code, general
web, knowledge/reading, and math; F3's multilingual loss was about 24.26%
below F2. The frozen loss rule therefore selected F3. These results are in
[long-confirmation-heldout-results-1590660.json](reports/long-confirmation-heldout-results-1590660.json)
and [long-confirmation-heldout-selection-1590660.json](reports/long-confirmation-heldout-selection-1590660.json).

## Matched seven-task confirmation

Slurm job 1590907 reran Base and evaluated both F2 and both F3 checkpoints
with two distinct RTX 5090 GPUs under the same frozen zero-shot, BF16,
2,048-context lm-eval protocol. It completed in 28m38s with exit code `0:0`.
The two-GPU Base differed from the earlier four-GPU Base by only +0.0384
percentage points in the unweighted mean.

| Task | Matched Base | F2 two-seed mean | F2 delta | F3 two-seed mean | F3 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| HellaSwag | 42.3422% | 42.9347% | +0.5925 pp | 42.8102% | +0.4680 pp |
| PIQA | 66.8662% | 67.6551% | +0.7889 pp | 67.3830% | +0.5169 pp |
| WinoGrande | 53.6701% | 51.9732% | -1.6969 pp | 51.3023% | -2.3678 pp |
| OpenBookQA | 33.2000% | 33.7000% | +0.5000 pp | 33.4000% | +0.2000 pp |
| ARC-Easy | 55.2609% | 54.2508% | -1.0101 pp | 54.5455% | -0.7155 pp |
| ARC-Challenge | 28.3276% | 29.4795% | +1.1519 pp | 29.5222% | +1.1945 pp |
| BoolQ | 56.2997% | 59.7248% | +3.4251 pp | 59.6942% | +3.3945 pp |
| **Unweighted mean** | **47.9952%** | **48.5311%** | **+0.5359 pp** | **48.3796%** | **+0.3844 pp** |

F2 ranked first by the frozen higher-worst-seed rule. Its two seed scores were
48.5191% and 48.5432%, a spread of 0.0240 percentage points. F3's spread was
0.1670 points. The bound summary is
[seven-task-confirmation-2gpu-results-1590907.json](reports/seven-task-confirmation-2gpu-results-1590907.json),
with SHA-256
`7eddae38ce9b1b002ede388dc73c55e078ccbeb38269c5e893609d997e31ab99`.

## Current gate

F2 provides a stable positive direction and is about 1.47 percentage points
short of a 50% seven-task mean. F3 remains the held-out loss winner. Neither is
formally promoted because the accuracy gains are small relative to sampling
uncertainty and both recipes regress on WinoGrande and ARC-Easy. Future recipe
selection must use new contamination-screened development proxies because the
seven final task results have now been inspected.

The present evidence supports reproducible training, checkpoint integrity,
measured MFU, held-out loss improvement, and matched base-model multiple-choice
accuracy. It does not establish instruction following, safety, calibrated
generation, tool use, domain mastery, or superiority over current public
models. SFT, preference optimization, and RLVR have not started.
