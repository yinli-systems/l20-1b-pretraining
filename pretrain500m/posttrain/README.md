# 529M post-training program

This directory defines the next optimization stage without changing or
overwriting the admitted base checkpoint. The base artifact is immutable:

- checkpoint step: `7629`
- prediction tokens: `15,999,172,608`
- checkpoint SHA-256:
  `13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf`
- parameters: `528,748,800`

The data-mixture stage has advanced through a frozen two-seed screen and four
long confirmation runs. F2 reasoning and F3 broad multilingual were each
trained for 2,147,483,648 prediction tokens with two seeds on four RTX 5090
GPUs. All four runs completed and passed the MFU and checkpoint-integrity gates.
The frozen five-domain held-out evaluation remains the selection gate. No
candidate is promoted until that evaluation completes and its artifacts pass
the bound hash checks.

The earlier continued-pretraining learning-rate pilot compared `3e-5`, `1e-4`,
and `3e-4` from the exact immutable base checkpoint. Its selection evidence was
exploratory and was superseded by the frozen recipe screen and long-confirmation
protocols recorded in the result receipts.

## Optimization loop

1. Freeze independent train, development, and final-test data with revisions,
   licenses, hashes, deduplication, and benchmark-decontamination receipts.
2. Run short SFT learning-rate and mixture pilots against development metrics.
3. Promote one SFT candidate only when instruction and domain development
   metrics improve without a material base-capability or safety regression.
4. Run preference optimization only after SFT passes. Use preference data that
   is disjoint from all evaluation prompts and keep the SFT model as reference.
5. For domains that need new factual knowledge, compare domain-adaptive
   continued pretraining followed by SFT against SFT alone.
6. Use verifier-based optimization only for tasks with deterministic checkers.
7. Evaluate a promoted candidate once on the sealed final suites and retain all
   failures and negative ablations.

The seven-task base benchmark is a final regression and comparison suite. Its
test labels are not training data and are not used for repeated candidate
selection.

## Claim boundary

"Better than all market models" is not a valid result without a predefined
comparison class and matched reruns. Direct claims require exact competitor
checkpoints, the same prompts, precision, context length, seeds, chat template,
and metric implementation. The practical target is a statistically supported
Pareto improvement within the selected parameter and domain class, including
quality, latency, memory, and serving cost.
