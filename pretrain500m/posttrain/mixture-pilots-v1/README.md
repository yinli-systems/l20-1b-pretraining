# Current-corpus mixture pilots v1

This bundle implements the seven P1 experiments from the archived GPT-6 Pro
research plan using the already-admitted packed corpus. It does not label the
data as fresh and it cannot make a market-superiority claim.

Each arm starts from the same immutable 528,748,800-parameter base checkpoint,
uses one four-RTX-5090 DDP allocation, processes exactly 536,870,912 prediction
tokens in 256 steps, and uses seed 20260916. R0 through R4 share a peak learning
rate of 6e-5. R3 also runs at 3e-5 and 1e-4. The warmup is 67,108,864 tokens,
and the rolling ten-step MFU median after a five-step grace period must remain
strictly above 0.70.

`build_inputs.py` derives new hash-bound manifests from the admitted F2 parent
source pool, verifies the base and parent identities, enforces every cumulative
source repeat cap, and emits matching admission receipts. `submit.py` refuses a
duplicate active campaign or less than 50 GiB free space before submitting all
seven jobs. `run.sbatch` verifies four physical RTX 5090 GPUs and uses the same
compiled DDP path that previously measured about 0.75--0.78 MFU.

The frozen development loss is only a screening signal. Independent capability
evaluation and at least two-seed confirmation are required before selecting a
long continuation candidate.
