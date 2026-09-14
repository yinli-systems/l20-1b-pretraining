# CPT v8 eight-GPU scale qualification

This is a 20-step, no-checkpoint, two-node qualification of CPT v8. It keeps the
exact 2,097,152 global prediction tokens per optimizer step with training
microbatch 4 and accumulation 32. Validation microbatch 2 covers all 7,824
frozen development blocks exactly.

The run passes only with eight distinct RTX 5090 devices, finite metrics, a
completed 20-step status, the frozen qualified DDP policy, no bucket rebuilds,
and a final ten-step median MFU strictly above 0.50. The result qualifies
throughput and integration only.
