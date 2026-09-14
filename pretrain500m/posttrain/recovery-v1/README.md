# Actual GPU recovery qualification

Slurm job 1588191 uses four RTX 5090 GPUs on one allocation. It compares 32
continuous optimizer steps with 16 steps plus a process restart and 16 resumed
steps, always starting from the immutable 529M base and the same full LR schedule.
Each branch consumes 67,108,864 prediction tokens from the existing FineWeb pack.

This is actual GPU training for recovery and efficiency qualification. It does
not introduce new knowledge data, select a new mixture, or establish a capability
gain. The existing v2 corpus-admission gates remain unchanged. This separate
runner accepts only the exact immutable original data manifest and its declared
shards, is capped at 134,217,728 tokens per branch, and cannot certify a new corpus.

The code archive SHA-256 is
`a653d8549f467aee7e2dd2f6b2772ea31cd9138e522678ade4edf2a9aff5b88a`.
The frozen remote source is
`/ssd/scxi253/pretrain500m-20260912-v1/source/recovery-v1`.
Outputs are under
`/ssd/scxi253/pretrain500m-20260912-v1/posttrain/recovery-v1-1588191`.
The final comparison requires exact model, optimizer, reader, origin, fingerprint
and all-rank RNG equality, plus measured rolling MFU strictly above 0.50.
`QUALIFICATION_COMPLETED` for one branch is not a passing paired comparison.

Final result: job 1588191 ended FAILED after both branches completed 32 steps.
Their MFU windows passed, but exact model comparison failed at the embedding
weights. Logged loss and gradient norm match through step 17 and differ from
step 18. The failed source and artifacts are preserved; recovery-v2 is a
separate candidate and must qualify independently.
