# CPT v8 qualified-reduction throughput runner

V8 adds exact validation microbatching and a metrics-only checkpoint mode to the
qualified v4 runner so the same 2,097,152-token optimizer step can be tested at
4, 8, and 16 GPUs. It deliberately freezes the reduction policy used by the
healthy formal screens: `find_unused_parameters=true`, `static_graph=false`,
gradient bucket views enabled, and a 25 MiB bucket cap.

The alternative `find_unused_parameters=false, static_graph=true` is excluded.
Job 1588605 reproduced PyTorch issue 143580: the first accumulated backward in
`no_sync()` can fail with `expect_autograd_hooks_ INTERNAL ASSERT FAILED` when
`static_graph=true`. The dynamic false/false variant is also excluded because
job 1588569 rebuilt buckets and violated the fixed-reduction invariant.

The `none` checkpoint mode exists only for short throughput qualification. V8
does not create a new formal promotion path.
