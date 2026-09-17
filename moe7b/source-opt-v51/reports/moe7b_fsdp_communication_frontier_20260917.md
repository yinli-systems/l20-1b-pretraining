# 7B MoE FSDP2 communication frontier

The active layout all-gathers each layer again across eight accumulation
microbatches.  It uses approximately 19.6 GB reserved memory on each 32.6 GB
RTX 5090, leaving enough measured headroom to test retaining full BF16
parameters after non-final backwards.

The primary candidate keeps unsharded parameters between accumulation
microbatches, keeps the root parameter group unsharded between forward and
backward, synchronizes sharded gradients every microbatch, and retains FP32
reduce-scatter.  The final microbatch restores reshard-after-backward, so every
optimizer step ends in the ordinary sharded state.  A secondary candidate uses
BF16 reduction and is treated as a separate numerical change.

Each candidate is a bounded fresh-initialization screen.  OOM, non-finite
metrics, less than 2 GiB headroom, or less than 10% throughput improvement is a
rejection.  Any performance pass still requires matched all-tensor update
parity and sustained confirmation before it can define the fresh 150B formal
run.  The active pilot checkpoint fingerprint is unchanged and cannot silently
resume under a different communication policy.
