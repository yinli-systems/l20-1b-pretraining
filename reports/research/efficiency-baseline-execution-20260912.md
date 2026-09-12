# Same-protocol efficiency baselines: execution record

This work tests the supplied research hypothesis. It does **not** adopt its
claims of world #1, a verified global Pareto frontier, or 75x efficiency.
The original random-initialized 20B checkpoint is the comparison target;
continuation A/B checkpoints are not substituted for it.

## Frozen scope

The active plan is
[`efficiency-evaluation-plan-20260912-v3.json`](../receipts/efficiency-evaluation-plan-20260912-v3.json),
SHA-256 `a52debd19314c47f5517b1075ef30c5b586da4e65525a9cef4920a958d4f5a3c`.
It declares **36 new checkpoint evaluations** and reuses two verified reference
points: our original 20B model and the already evaluated TinyLlama 3T model.

| Family | New checkpoints | Training-token budgets |
| --- | ---: | --- |
| DataDecide | 20 | Four recipes, each at 14.42B / 18.02B / 21.63B / 28.84B / 61.28B |
| WebOrganizer | 2 | Global DCLM filter and domain-mixture baseline, each 28.80B |
| TinyLlama | 10 | Official 10B / 21B / 31B / 63B / 105B branches; 503B / 1T / 1.5T / 2T / 2.5T releases |
| Pythia, OPT, Falcon-RW, Phi-1.5 | 4 | Author-reported final budgets; Phi is a separate synthetic/teacher reference |

Each repository resolves to an immutable commit, with individual weight and
tokenizer artifact size/hash requirements. Exact identities and provenance
are in the [inventory](../receipts/efficiency-baseline-inventory-20260912-v5.json).
TinyLlama branch/release labels are rounded budgets, not exact token counters.
The official 105B branch is not silently relabeled as the old evaluation table's 103B point.

The lower-compute 14.42B DataDecide and 10B TinyLlama controls matter: if our
model were the cheapest candidate in the entire plot, its frontier membership
would be automatic, not evidence of superior efficiency. These controls were
added before any new same-protocol score was produced. The initial 31-job
queue was stopped while it was empty and waiting; its plan and receipt remain
archived, and the training process was untouched.

## DataDecide selection audit

The [official evaluation dataset](https://huggingface.co/datasets/allenai/DataDecide-eval-results)
was pinned to `9919b5a0e61e57a85021263918fa82d6ceaee038`.
All four upstream Parquet files were downloaded and SHA-256 verified.
The scan recovered 2,025 complete seven-task 1B result rows and compared
25 recipes at the 100B endpoint, using all three reported seeds.

The predeclared selector was the mean of the seven upstream primary metrics
across the three seeds at step 69,369. Its top recipe was
**DCLM-Baseline (QC 20%)**, with an upstream mean of 0.6254613.
This selects a strong terminal recipe; it does not establish the best recipe
at every lower budget. FineWeb-Edu, DCLM-Baseline, and QC-7%-FW3 are additional
declared anchors, not a claim that these are the top four recipes.
The [scan and its selection rule](../metrics/datadecide-upstream-scan-20260912.json)
are retained. Upstream OLMES results are **not** mixed with our harness scores.

DataDecide token budgets use the author's step count times 704 sequences
times 2,048 tokens, following the [upstream training configuration](https://github.com/allenai/DataDecide).
Its published 1.1768B parameter definition excludes input embeddings. The
author model has 1,279,854,592 total parameters, including 103,022,592 input
embedding parameters. The worker records actual parameter counts consistently
instead of using inconsistent model-name labels.

## Evaluation and uncertainty

The fixed primary suite is HellaSwag, PIQA, WinoGrande, OpenBookQA, ARC-E,
ARC-C and BoolQ. It uses `lm_eval==0.4.9`, BF16, context 2,048, zero shot,
no chat template, complete test sets, batch setting `auto:4`, seeds
42/42/42 and few-shot seed 1234.

Primary metrics follow the prior comparison: `acc_norm` for HellaSwag, PIQA,
OpenBookQA and ARC-E/C; `acc` for WinoGrande and BoolQ. Every accepted result
must match the original task configuration, document IDs/hashes and prompt/
target hashes, and its sample mean must reproduce its reported aggregate.
The paired bootstrap uses 10,000 repetitions, seed 20260912.

Both the seven-task mean and the six-task mean excluding BoolQ are reported.
The original data did not explicitly decontaminate BoolQ. Confidence intervals
describe benchmark-sample uncertainty, not training-seed uncertainty; they do
not correct for testing many models. This is not a complete ability, coding,
instruction-following or generation-quality evaluation.

The graph uses `6 × actual total parameters × training tokens` as an
**approximate training-compute proxy**, not measured hardware FLOPs. Data
curation, teacher-model generation, tokenizer/gate experiments and model-search
compute are not included. Phi stays visually separate and is excluded from
the natural-corpus candidate frontier. WebOrganizer's domain mixture was
optimized using related target benchmarks; that selection is disclosed.

## Execution and safeguards

### Evaluation-first update

The user subsequently requested: “先完成评测吧，训练可以等等”. B received
its supported graceful stop signal and saved the complete resumable checkpoint
at **step 61**, preserving 63,713,280 branch prediction tokens and the optimizer/
RNG state. Its checkpoint SHA-256 is
`51de78a08b15fde30042e47553ca4cb2e5d3e65f280be86b5b362683b4d049af`.
The [pause receipt](../receipts/efficiency-training-pause-20260912.json) verifies
the 13,200,791,363-byte checkpoint, `checkpointed_stop`, and released CUDA/GPU
lock. No rollback to step 50 was used.

Plan v3 admits only this exact user-authorized checkpointed pause, rather than
waiting for B step 190. The 36 candidates and scoring protocol are unchanged;
already downloaded/CPU-verified Pythia runs first to avoid waiting for another
download before the first evaluation. There is **no automatic training resume**.
The old empty v2 queue's rejection of the early stop was expected and is
archived; it was not a training or benchmark failure. The original v2 runner
source and receipt were retained before the priority change. Local tests after
the change: 122 passed, one target-host serialization test skipped.

At approximately 2026-09-12 00:48 Dubai, the
[evaluation-first launch receipt](../receipts/efficiency-evaluation-priority-start-20260912.json)
reported Pythia running, queue PID 3196358. Worker PID 3196371 had entered
69,106 log-likelihood requests on CUDA; the observed GPU utilization sample
was 100%. The old training PID was absent, and only the evaluation tmux session
remained. No new baseline result had completed at that snapshot.

### Earlier snapshot (superseded by the priority update)

Verified snapshot: **2026-09-12 00:38 Dubai / 2026-09-11 20:38 UTC**.
The [queue launch receipt](../receipts/efficiency-queue-start-20260912.json)
records PID 3194909, status `waiting_for_B_gpu_lock`, and **zero new completed
evaluations**. Both tmux sessions were present; CUDA still belonged only to B
(PID 3190905). B was at step 57/190, about 13,004 prediction tokens/s. Its
step-50 original-mixture validation PPL was 11.295613, essentially unchanged
from the original 11.298834 reference; this is not a downstream benchmark win.
This is a dated snapshot, not a live endpoint.

The [v1 supersession receipt](../receipts/efficiency-plan-v1-supersession-20260912.json)
records the replacement of the earlier empty waiting queue. No baseline
evaluation or training process was interrupted.

Remote bundle:
`/home/hhai/pretrain/evaluations/efficiency-20260912-v1`.
The directory is the experiment bundle; `plan-v3` is its current plan revision.
The queue runs in tmux session `efficiency-eval-v3`.

The original policy required normal B completion at step 190. Under the
explicit evaluation-first request, the queue instead admits the exact
hash-bound checkpointed pause described above, after the GPU lock releases
and no CUDA process remains. Other early stops or failed checkpoint checks
remain blocked. The queue never promotes a pilot or starts/resumes training.

Models are downloaded and evaluated serially. The planned downloads total
about **168.2 GB** cumulatively, not simultaneous disk occupancy; Pythia's
2.09 GB was already prefetched. This is inference-only and charges zero
additional **training tokens**, but requires GPU time and network transfer.
No new paid GPU instance was provisioned.

The downloader verifies every artifact and reserves 11 GiB beyond the pending
model download after B completes. It removes only this queue's explicitly
enumerated, hash-verified, reproducible downloads after the result and analysis
are verified. Training data, checkpoints, existing caches and unrelated files
are never cleanup targets. Failures retain evidence and stop the queue; there
is no silent rerun, changed protocol, unverified replacement or skipped failure.

Each worker must pass exact HF loading-key diagnostics and finite actual-
checkpoint CUDA logits before spending time on benchmarks. Author adapters
are isolated, with no modifications to B's installed training environment.
Runtime package versions, 13,660 harness/Transformers source files, and all
admitted author dependency files are hash-bound.

Completed preparation checks:

- DataDecide/OLMo CPU toy forward, causality, BF16 and exact save/load round trip.
- WebOrganizer/OpenLM equivalent CPU toy checks, using its declared PyTorch
  backends. The [two lazy-import compatibility edits](openlm-adapter-compatibility-20260912.md)
  preserve the author's numerical implementations.
- Actual Pythia weight hashes, exact loading keys, finite CPU logits and one
  invented, non-benchmark harness likelihood request. Actual parameter count:
  1,011,781,632. This is not a Pythia benchmark result.
- Local repository tests: 121 passed, one target-host serialization test skipped.
- Remote frozen-plan check and isolated OLMo runtime bindings passed.

These checks establish preparation/admission readiness, not completed GPU
evaluations. The first new verified result enables an incremental SVG graph
and JSON comparison. No unmeasured baseline point is inserted into the graph.
The initial chart is explicitly partial and candidate-set-only, with detailed
paired comparisons and six-task sensitivities in its analysis records.

## Still blocked

| Requested baseline | Evidence boundary |
| --- | --- |
| Cerebras-GPT-1.3B | Canonical repository API returned HTTP 401; no unverified third-party replacement was used. |
| MobileLLM-1B | Official weights/config require approved authenticated access; no cached HF token was available on the GPU host. |
| Original DCLM 1B-1x | Matching original competition checkpoint not located. WebOrganizer's DCLM-filtered models are explicitly separate research baselines. |
| Meta Lingua 1B/60B | Author-reported benchmark table located, but matching official released checkpoint not located. |

Primary sources: [DCLM leaderboard](https://www.datacomp.ai/dclm/leaderboard.html),
[Lingua](https://github.com/facebookresearch/lingua),
[WebOrganizer DCLM baseline](https://huggingface.co/WebOrganizer/LM-1b_1x-DCLMFasttext),
[TinyLlama official early checkpoint collection](https://huggingface.co/TinyLlama/tinyLlama-intermediate-checkpoints),
[MobileLLM](https://huggingface.co/facebook/MobileLLM-1B).

Even after this declared subset completes, it cannot establish a world rank.
It can establish measured tradeoffs among the evaluated candidates. Broader
low-budget recipe/seed coverage, complete access, contamination review, and
additional ability tests remain necessary for stronger claims.
