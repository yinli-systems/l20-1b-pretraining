# High-quality 7B MoE training programme — 2026-09-17

## Decision

The 8.376B-token FineWeb-Edu run remains a systems/stability pilot. Its tokens
stay outside the formal budget because it uses one source and a short learning
rate horizon. The formal programme is now a 5.1T-token candidate: 5T tokens of
base pretraining followed by a separately admitted 100B high-quality anneal.
The checkpoints at 150B, 300B, 600B, 1T, 2.6T, 5T and 5.1T are evaluation and
stop/revise points on one predeclared schedule. They are not invitations to
extend a completed short schedule after seeing its score.

This budget is a competitive research target, not a result. The closest public
architecture, OLMoE-1B-7B, has 6.9B total and 1.3B active parameters and was
trained on 5T tokens. DCLM's dense 7B reference used 2.6T tokens. A 150B run is
a useful first qualification checkpoint, but is only 3% of the OLMoE token
budget and cannot support a market-leading claim by itself.

The machine-readable contract is
[`../plans/high_quality_5p1t_program_v1.json`](../plans/high_quality_5p1t_program_v1.json).

## Evidence that changes the plan

- [OLMoE](https://arxiv.org/abs/2409.02060) is the closest public comparator.
  It used dropless token-choice routing, 64 fine-grained experts with eight
  active, about 4M tokens per optimizer step, AdamW with beta2 0.95, a 0.01
  load-balancing coefficient, FP32 optimizer state, and 5T pretraining tokens.
  Its controlled study reports a smaller 1–2% gain when moving from 32/4 to
  64/8 expert granularity at matched total and active compute. That makes our
  current 16/2 design an ablation candidate, not a proven optimum.
- [DataComp-LM](https://arxiv.org/abs/2406.11794) reports that model-based
  filtering was central to its data result; its 7B model trained on 2.6T tokens
  reached 64% MMLU 5-shot. This supports DCLM-style quality filtering and shows
  why raw corpus size is not enough.
- [Llama 3](https://arxiv.org/abs/2407.21783) selected its mixture with small
  scaling-law models and larger confirmation runs. Its reported mixture is
  roughly 50% general knowledge, 25% math/reasoning, 17% code and 8%
  multilingual data. It also used URL, global MinHash document and aggressive
  line deduplication, then tested a 40B-token anneal with 30% candidate data.
  These proportions are hypotheses for our proxy sweep, not constants copied
  into production.
- [OLMo 2](https://arxiv.org/abs/2501.00656) used 50B/100B/300B microanneals to
  choose late-stage data. Its 100B mix was about half filtered DCLM, with math,
  Q&A, science and reference sources; it also found that a small amount of
  domain data can help and that limited repetition can help, while checkpoint
  averaging was useful. This motivates our separate 100B anneal and a maximum
  of four presentations for scarce late-stage sources.
- [SmolLM2](https://arxiv.org/abs/2502.02737) trained its 1.7B model for about
  11T tokens and its 360M model for 4T. Its staged results show that math and
  code quality mattered more than merely adding low-quality domain data.
  FineMath-4+ beat less selective math corpora, while one repeated math source
  plateaued after roughly ten epochs. Our plan therefore caps replay and keeps
  FineMath-4+ distinct from the lower-scored remainder.
- [FineWeb](https://arxiv.org/abs/2406.17557) demonstrates that extraction,
  filtering and dedup choices change downstream results, while FineWeb-Edu
  shifts domain coverage toward academic and programming text. It should be
  mixed with broad web rather than treated as complete coverage.
- [The Stack](https://arxiv.org/abs/2211.15533) found large code gains from
  near-deduplication and reproduced prior results using permissively licensed
  code. Code admission therefore remains file-license-aware, provenance-bound,
  globally near-deduplicated and benchmark-decontaminated.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) provides positive large-scale
  evidence for auxiliary-loss-free batch-wise balancing and multi-token
  prediction, but at a radically different 671B/37B-active scale. We will test
  those ideas as matched ablations; neither enters the 7B production graph
  without a quality and numerical-parity win.
- NVIDIA's official [Megatron Core MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/moe.html)
  identifies the relevant systems path: expert parallelism, grouped GEMM,
  permutation/router fusion, all-to-all overlap and a distributed optimizer.
  It notes that unoptimized expert all-to-all can consume 30–40% of training
  time. These are the next performance candidates after the custom EP screen,
  with exact-update validation before promotion.

## What previous experiments permit

The 529M F2 family proved an engineering recipe, not a data recipe. Its runs
held roughly 76–77% rolling MFU, but the two-seed continuation reduced the
seven-task mean from 48.53115% to 48.45525%. The frozen selector therefore
retired F2 continuation. We reuse large microbatches, low accumulation,
constant tokens per update, deterministic resume and strict MFU telemetry. We
do not reuse the rejected F2 corpus mixture as evidence of quality.

The current exact 150B candidate is a useful middle mixture: 58% filtered broad
web, 17% permissive code, 10% math, 10% multilingual and 5% science/reference.
It becomes one arm of a proxy comparison against a more web-heavy OLMoE-style
mix and a capability-heavy Llama-style mix. The comparison starts from one
immutable 529M parent, uses 536,870,912 prediction tokens per arm, then requires
a two-seed confirmation. Adaptive seven-task scores can diagnose; they cannot
alone choose the production mixture because those tasks have already been
inspected repeatedly.

## Architecture and optimization sequence

1. Preserve the current 8xRTX5090 and 16xRTX4090 pilot allocations until their
   normal checkpoint boundaries. They provide durability and stability
   evidence only.
2. Run the 8-way EP screen already queued and the new 4-way EP screen. Both
   preserve 131,072 prediction tokens per optimizer step and require at least
   50% median causal-useful MFU, finite loss/gradients and at least 2 GiB of
   per-GPU headroom.
3. A systems pass advances only to numerical qualification: all-tensor update
   parity, FP32 master/optimizer-state semantics, checkpoint/reload and a
   sustained run. The pure-BF16 screen cannot initialize formal training.
4. Compare 16/2, 32/4 and 64/8 expert granularity at matched total and active
   expert capacity. OLMoE makes 64/8 the serious challenger, but the current
   16/2 graph remains the reference until a matched proxy and long confirmation
   win. A granularity change requires a fresh formal initialization.
5. After the custom EP graph is correct, qualify Megatron Core grouped GEMM,
   fused permutation/router operations and all-to-all overlap. Scale to EP8 ×
   DP2 when 16 cards are available; add DP replicas only after data order,
   optimizer scaling and checkpoint topology conversion are exact.

## Data and training gates

Every document must bind its source revision, byte hash, license/provenance,
quality decision, privacy decision and split. Admission uses global URL, exact,
near-document and repeated-line deduplication; exact/13-gram benchmark matching
plus a semantic clean-room review; and independent train/validation/sealed-test
isolation. No public benchmark training split is permitted in the final
anneal. The existing writer remains fail-closed until its receipt advances from
`RUNNING_NOT_ADMITTED` to `TRAINING_ADMITTED`.

The 100B late-stage candidate is 50% more-restrictively-filtered web, 20% math,
15% permissive code/technical Q&A, 10% science/reference and 5% high-quality
multilingual data. It is frozen only after microanneals. The inherited learning
rate decays linearly to zero, and any checkpoint averaging is preregistered and
evaluated against its individual members.

The first required capability threshold is the user's seven-task macro above
50%. A claim of exceeding another model is stricter: identical weights scope,
tokenizer/prompt/support setting, sample rows and scoring code, followed by a
positive paired-bootstrap lower bound against every named comparator. Held-out
loss, one benchmark, or a high MFU number cannot substitute for that test.

## Compute reality

Using the current strict causal-useful numerator (about 7.565 GFLOP per
prediction token) and the project's 209.5 BF16 TFLOP/s RTX5090 denominator, a
5.1T run takes an idealized 533 days on 8 cards at 50% MFU, 266 days on 16
cards, or 66.6 days on 64 cards. At 77% MFU those figures are about 346, 173
and 43.3 days. Queueing, evaluation, checkpointing, data I/O and failures are
additional. The plan therefore uses measured checkpoint gates and scales GPU
count; it does not pretend that MFU alone makes 5T a short run.
