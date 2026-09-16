# F2 fresh-N1 continuation v1

This bundle reopens the F2 direction as a controlled checkpoint-continuation experiment. It does **not** repeat the retired old-F2 continuation mixture. Both arms start from the stronger original F2 seed-20260915 checkpoint and use the already admitted fresh `N1_new_pool_knowledge` mixture, which was the best of the first five new-pool arms on the matched seven-task screen.

The two arms differ only in peak learning rate (`3e-5` versus `6e-5`). Each processes exactly 536,870,912 prediction tokens on four RTX 5090 GPUs with the frozen 2,048-token geometry, deterministic execution, validation before and after training, and a rolling ten-step MFU median that must remain strictly above 0.70 after the five-step grace period. The final checkpoint retains optimizer state so the selected arm can be resumed exactly.

The parent is a model-only checkpoint, so the first launch necessarily creates a fresh AdamW optimizer. Scheduler submission, allocation, live MFU, completion, seven-task retention, a score above 0.50, and model superiority are separate gates. Neither arm is promoted by this bundle.
