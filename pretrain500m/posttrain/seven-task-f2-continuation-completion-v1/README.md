# F2 seven-task missing-pair completion

This job evaluates the missing seed-20260915 continuation and reruns its exact
matched parent on the same two RTX 5090 GPUs. The parent must reproduce every
previous task score, sample count, metric name, aggregate score, and bootstrap
interval before the new pair may be combined with the completed seed-20260914
pair.

The task suite and export protocol are adaptive and already inspected. The
result can measure progression and retention under the same seven-task setup;
it cannot serve as sealed evidence or automatically promote the model.
