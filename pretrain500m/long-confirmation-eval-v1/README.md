# Long-confirmation evaluation

This source waits for exactly one successful 2.147B-token model-only checkpoint
for each frozen F2/F3 recipe and seed, verifies its training status, manifest,
content hash, and checkpoint metadata, then submits a four-GPU evaluation on the
family-disjoint five-domain confirmation set. The selector applies the frozen
rule: lower worst-seed equal-domain loss, then lower two-seed mean, then recipe
identifier. Its output is a selection receipt, not a generative-capability or
market-superiority claim.
