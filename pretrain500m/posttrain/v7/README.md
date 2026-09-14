# CPT runner v7

V7 tests the fixed static-graph DDP fast path. The v5 benchmark showed that
`find_unused_parameters=false, static_graph=false` lets DDP rebuild buckets on
the next iteration, which violates the qualified recovery invariant. V7
therefore defaults to `find_unused_parameters=false, static_graph=true` and
keeps explicit switches for a same-allocation comparison with the v4 policy.

The separate validation microbatch remains available so all 7,824 frozen
development blocks divide exactly at 4, 8, or 16 GPUs. The `none` checkpoint
mode remains metrics-only. V7 is qualification-only until fixed-bucket,
numerical, MFU, and speed gates pass on real GPUs.
