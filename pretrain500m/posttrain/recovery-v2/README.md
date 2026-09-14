# Recovery candidate with stable DDP reduction buckets

The preceding job 1588191 finished all training steps but failed exact checkpoint
comparison. Logged loss and gradient norm agreed through step 17; divergence
appeared in step 18, after restart at step 16. Aggregate agreement does not prove
that every gradient, parameter or optimizer value matched.

This isolated candidate tests a specific explanation: DDP normally rebuilds its
initial reduction buckets after the first backward pass, while a reconstructed
DDP object begins with its initial buckets again. Different floating-point
reduction layouts can change the first resumed update. PyTorch 2.11.0's
[reducer implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/distributed/c10d/reducer.hpp)
disables rebuilding when `find_unused_parameters=True` and `static_graph=False`.
This is a hypothesis until the actual GPU comparisons pass.

The only numerical-policy change is enabling unused-parameter detection to keep
the 25 MiB reduction layout fixed. The policy and diagnostic source hashes are
included in the run identity. Every step records the bucket layout and rejects
unexpected rebuilding. No model, data, optimizer, LR, target-token or acceptance
criterion is changed from recovery-v1. Before the full run, a small synthetic
FP32 four-GPU probe compares reducer recreation under default and fixed buckets;
it must pass for fixed buckets before the full-model test starts. This probe
is not a full-model process-restart or efficiency qualification.

The actual 529M test retains 32 uninterrupted steps versus 16 plus a process
restart plus 16 steps, 67,108,864 prediction tokens per branch, and rolling MFU
strictly above 0.50. Exact model, optimizer, reader, provenance and all-rank RNG
comparison is unchanged. At the first resumed update, full parameter/optimizer
and per-rank gradient content hashes are recorded without extra tensor files.
Gradient and post-update diagnostic time is included in that step's MFU timing.
The original engineering-only corpus admission remains restricted to the frozen
FineWeb pack. This run does not admit the new corpus or establish quality gains.

Source and output use separate `recovery-v2` paths. The previous failure, base,
formal data and other project artifacts are preserved. The existing immutable
`recovery-v1-inputs` manifests and protocol are reused by content hash.
