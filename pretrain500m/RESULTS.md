# 529M experiment results

Status date: 2026-09-16 UTC.

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

## Paired F2 continuation and export diagnostic

Two F2 continuations started from their exact matched long-confirmation
checkpoints. Each completed 256 steps and 536,870,912 additional prediction
tokens on four RTX 5090 GPUs, with every ten-step median MFU gate above 0.70.
Both full-state checkpoint binaries were independently SHA-256 checked on
ParaCloud; the binaries remain outside Git history.

| Seed | Training job | Final ten-step mean MFU | Reserved five-domain masked-loss reduction versus matched parent | Seven-task mean |
| ---: | ---: | ---: | ---: | ---: |
| 20260914 | 1592301 | 0.76099 | 0.43185% | 48.5516% (parent 48.5191%) |
| 20260915 | 1592302 | 0.77360 | 0.40391% | Unknown: export gate failed |

The family-disjoint reserved comparison lowered masked loss in all five domains
for both seeds. Its evaluation job 1593179 completed with Slurm exit `0:0`.
The [training qualification](result-archive/f2-continuation-two-seed-qualification-20260916.json),
[formal evaluation exit](result-archive/f2-continuation-confirmation-scheduler-terminal-20260916.json),
and [paired masked-loss result](result-archive/f2-continuation-paired-confirmation-results-20260916.json)
preserve checkpoint identities, metrics, and source checksums. Seed 20260914's
training stderr gained a Slurm teardown diagnostic even though its job and step
accounting eventually showed `COMPLETED 0:0`; the final stderr is archived.

The adaptive matched seven-task rerun used two RTX 5090 GPUs and exactly
reproduced the previous Base and both F2 parent aggregates. Seed 20260914's
continuation passed the original HF export smoke and improved its matched
parent by 0.03249 percentage points. Seed 20260915's export reload smoke had
247/256 BF16 argmax matches, below the frozen 97% minimum; job 1593250 exited
`FAILED 1:0` before its seven-task aggregate was written. The
[negative evaluation receipt](result-archive/f2-continuation-seven-task-failed-export-20260916.json)
and [partial artifacts](result-archive/para-f2-seven-failed-20260916/SHA256SUMS)
are retained. No complete continuation two-seed score exists.

A separate unchanged-gate export diagnostic on one RTX 5090 reproduced the
failure at 247/256 for seed 20260915 and passed at 250/256 for seed 20260914.
Both reloads had bitwise-exact model tensors and passed the original maximum
and mean BF16 logit-drift bounds. All changed argmax positions had close
native top-token margins, at most 0.0625. The
[diagnostic result](result-archive/f2-export-parity-diagnostic-results-20260916.json)
and [raw JSON snapshots](result-archive/para-f2-export-diagnostic-failed-20260916/SHA256SUMS)
preserve this negative gate result. Numerical sensitivity is a plausible
explanation, but the original gate remains failed and has not been relaxed.

Two additional input seeds were then frozen before execution and applied to
both continuation checkpoints under the same unchanged gate. Only one of four
cells passed: seed 20260914 had 247/256 matches on both new samples; seed
20260915 had 252/256 and 248/256. Across the original and two new samples,
each exact checkpoint produced both a pass and a failure. All four new reloads
again had bitwise-exact tensors and stayed inside the original max/mean logit
drift bounds. This directly establishes that the 256-position argmax decision
is sample-sensitive under the fixed setup, while leaving every individual
failure valid. The [four-cell receipt](result-archive/f2-export-parity-independent-samples-results-20260916.json)
and [nine-file snapshot](result-archive/para-f2-export-independent-samples-failed-20260916/SHA256SUMS)
retain the complete diagnostic grid. No threshold has been changed.

An adaptive margin-aware protocol was then frozen on four additional unseen
input seeds and applied uniformly to both F2 parents and both continuations.
All 16 cells passed: every reload retained bitwise-exact tensors and finite
logits, all max/mean drift limits held, and every stable position preserved its
argmax. Across 2,048 positions per role, descriptive raw argmax agreement was
98.24% for the parents and 97.12% for the continuations; this raw rate is not
the new gate. Both continuation HF artifacts were independently streamed and
matched their export receipts. The [complete result receipt](result-archive/f2-margin-aware-export-results-20260916.json)
and [52-file small-artifact snapshot](result-archive/para-f2-margin-aware-export-pass-20260916/SHA256SUMS)
authorize completing the adaptive matched seven-task comparison. Because this
protocol was designed after inspecting the original failures, it is not sealed
evidence and does not itself establish a capability gain.

The missing seed-20260915 pair was then evaluated on two RTX 5090 GPUs. Its
matched parent exactly reproduced every prior task score, sample count, metric,
aggregate score, and bootstrap interval, permitting the cross-job two-seed
combination. Seed 20260914 improved by 0.03249 percentage points, while seed
20260915 regressed by 0.18428 points. The continuation two-seed mean was
48.45525%, versus 48.53115% for the matched parents, a decline of 0.07589
points. No individual task declined by more than two points, but the frozen
progression rule requiring both seeds to improve failed. The
[completion receipt](result-archive/f2-seven-task-completion-results-20260916.json)
and [ten-file snapshot](result-archive/para-f2-seven-completion-negative-20260916/SHA256SUMS)
retain the complete negative result. This continuation is not promoted.

## Current gate

The earlier F2 two-seed seven-task mean is 48.5311%, about 1.47 percentage
points short of 50%; the completed continuation mean is lower at 48.4553%.
F3 remains the earlier held-out loss winner. Neither recipe nor the F2
continuation is formally promoted: the continuation improved reserved masked
loss in both seeds but failed the two-seed capability progression rule. Future
recipe selection needs new contamination-screened development proxies because
the seven final task results have now been inspected.

The present evidence supports reproducible training, checkpoint integrity,
measured MFU, held-out loss improvement, and matched base-model multiple-choice
accuracy. It does not establish instruction following, safety, calibrated
generation, tool use, domain mastery, or superiority over current public
models. SFT, preference optimization, and RLVR have not started.

## GPT-6 Pro external research review

GPT-6 Pro at its maximum 5/5 reasoning setting reviewed this evidence and
primary public sources. The downloaded output is archived as an immutable
[research response](reports/gpt6-pro-max-research-response-20260915.md),
[proposed plan](reports/gpt6-pro-max-research-plan-20260915.json), and
[competitor registry](reports/gpt6-pro-max-competitor-registry-20260915.json).
These are recommendations, not completed training or matched competitor results.

The review corrected the model description to the actual `deep` configuration:
26 layers, hidden size 1,280, intermediate size 3,584, 20 query heads, five KV
heads, and head dimension 64. It also identified that the held-out receipt's
16,678,912 packed/processed prediction-token budget is distinct from the
10,103,815 mask-selected loss targets summed across the five domains. Future
receipts must report those quantities separately.

Its primary proposal is to retain F2 as a control and compare five fresh-corpus
recipes from the immutable parent, using new development and sealed evaluation
suites. The proposed funnel is 536,870,912-token pilots, two-seed
2,147,483,648-token confirmations, and only then a two-seed 20,000,538,624-token
run. The leading untrained hypothesis allocates 30% fresh FineWeb-Edu, 25%
DCLM, 20% English FinePDFs, 15% math, and 10% code, with peak LR 6e-5 as a
screened hypothesis. It is explicitly `PROPOSED_NOT_TRAINED` and blocked on
fresh-data admission, independent proxy/sealed suites, exact runtime binding,
full-state disk measurement, and fixed-horizon resume qualification.
