# 529M experiment results

Status date: 2026-09-14 UTC.

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

## Current gate

The four checkpoints passed exact plan construction and were submitted to a
frozen five-domain held-out loss evaluation as Slurm job 1590660. At the last
captured scheduler observation, the evaluation was pending allocation due to
priority. Recipe selection and model promotion therefore remain pending.

The present evidence supports reproducible training, checkpoint integrity,
measured MFU, and development-loss comparisons. It does not establish
generative task accuracy, instruction following, safety, post-training quality,
or superiority over public models. SFT, preference optimization, and RLVR have
not started.
