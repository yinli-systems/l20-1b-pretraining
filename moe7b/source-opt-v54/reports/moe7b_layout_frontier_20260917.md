# 7B MoE 8xRTX5090 layout frontier

The active 8-GPU systems pilot uses microbatch 1, accumulation 8, and synchronizes
gradients on every microbatch.  Its recent steady state is approximately 11,260
prediction tokens/s and 5.085% causal useful-matmul MFU.  This screen keeps the
model, source data, global 131,072 prediction tokens per optimizer step, precision,
optimizer, activation checkpointing, gradient synchronization policy, and seed fixed.

The three candidate layouts are microbatch/accumulation 2/4, 4/2, and 8/1.  Each
candidate runs directly for 12 optimizer steps without a 56 GB checkpoint, so the
allocation measures construction, memory feasibility, and steady-state throughput
rather than checkpoint I/O.  It cannot establish training quality.

A candidate is retained only if it completes on exactly eight RTX 5090 devices,
has finite loss and gradients, preserves 131,072 prediction tokens per optimizer
step, leaves at least 2 GiB device-memory headroom, and improves median throughput
by at least 10%.  A retained candidate must then pass matched all-tensor update
parity before any long run changes layout.  OOM, non-finite values, identity drift,
or insufficient speedup is a rejection, not a reason to weaken the gate.

The active 8x5090 and 16x4090 jobs remain untouched during these queued screens.
The formal 150B run remains gated on completion and admission of the frozen staged
data pack; pilot tokens are outside that formal budget.
