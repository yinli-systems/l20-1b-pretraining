# Four-GPU long confirmation launch

Each job starts from the frozen 529M base checkpoint and trains one F2/F3 seed
for exactly 2,147,483,648 prediction tokens on four RTX 5090 GPUs. The runner
requires exact admission, mixture, validation, protocol, source, and base-model
hashes. It aborts if the allocation is not four RTX 5090 GPUs or if the rolling
ten-step MFU is not strictly above 0.50 after the five-step grace period.

Four jobs can run concurrently on 16 GPUs. Their model-only finals are intended
for paired held-out recipe confirmation and do not constitute promotion.
