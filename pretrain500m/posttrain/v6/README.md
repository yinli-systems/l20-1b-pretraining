# CPT runner v6

V6 extends the prospective v5 speed candidate with a separate validation
microbatch. The frozen development set contains 7,824 blocks, so four-GPU,
eight-GPU, and sixteen-GPU training can evaluate the exact same set with
validation microbatches 4, 2, and 1 respectively. Training microbatch and global
prediction tokens per optimizer step remain unchanged.

DDP `find_unused_parameters` still defaults to false with an explicit A/B
switch, `static_graph` remains false, and `none` remains a metrics-only
checkpoint mode. V6 is qualification-only until real multi-node GPU checks pass.
