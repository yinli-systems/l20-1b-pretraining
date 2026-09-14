# DDP fixed-static-graph speed benchmark v2

This is a same-allocation BAAB comparison of the qualified v4 DDP policy
(`find_unused_parameters=true, static_graph=false`) and the prospective fixed
static-graph policy (`find_unused_parameters=false, static_graph=true`). It
replaces v1, whose false/dynamic arm correctly failed when DDP rebuilt buckets
before the second step.

Every segment uses the same four RTX 5090 GPUs, seed, initialization, F3 data,
optimizer, schedule, and 20-step token budget. The static path is admitted only
when no bucket rebuild occurs, every final ten-step median MFU is above 0.50,
loss and gradient traces match within 1e-12, and median speedup exceeds 0.5%.
