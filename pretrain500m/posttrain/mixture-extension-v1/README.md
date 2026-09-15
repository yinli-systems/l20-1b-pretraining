# R4 code-guard two-seed extension v1

This bundle scales the current-corpus R4 code-guard recipe to exactly
1,073,741,824 prediction tokens for seeds 20260916 and 20260917. Each run starts
from the immutable 528,748,800-parameter Base checkpoint and uses four RTX 5090
GPUs, a 2,097,152-token global batch, 6e-5 peak learning rate, 67,108,864-token
warmup, deterministic execution, and a model-only final checkpoint.

R4 was selected as the only novel scale candidate. Its adaptive seven-task
score improved over Base by 0.001337 while remaining effectively tied with the
R0 control, and its development code, general, and multilingual losses were
better than R0. R3 1e-4 is excluded because its seven-task score fell below
Base. R0 is excluded because it is a control and a longer F2 incumbent already
exists.

`build_inputs.py` doubles the R4 pilot block quotas, verifies the admitted
parent manifest and capability result, and checks every source against its
cumulative repeat cap along each candidate lineage. `submit.py` requires at
least 75 GiB of shared space and a current public single-node four-GPU layout,
then submits both seeds exactly once. The training runner keeps the strict MFU
rule: after a five-step grace period, every checked rolling ten-step median must
remain strictly above 0.70.

This is an adaptive current-corpus experiment. It is not fresh-corpus training,
a sealed evaluation, a formal promotion, or evidence of market superiority.
