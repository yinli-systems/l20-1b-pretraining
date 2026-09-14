# CPT runner v5

V5 is a prospective speed candidate. It defaults DDP
`find_unused_parameters` to false after every rank in the completed v4 screens
reported that no parameter was unused. The explicit boolean switch is retained
for an otherwise-identical A/B benchmark. `static_graph` remains false so this
change does not assume more than the observed parameter-use invariant.

The new `none` checkpoint mode is for short metrics-only throughput benchmarks;
it still runs validation and the rolling-median MFU gate but writes no model or
optimizer state. This version remains qualification-only until a paired GPU
benchmark passes numerical, MFU, and identity checks.
