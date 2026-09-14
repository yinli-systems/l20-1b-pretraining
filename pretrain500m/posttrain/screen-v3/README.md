# Four-GPU mixture screen v3

Each frozen recipe starts from the same 529M checkpoint, consumes exactly
536,870,912 prediction tokens, evaluates the same document-masked five-domain
development set, and saves model weights only after all 256 steps complete.
The run stops without a checkpoint when the rolling ten-step median MFU is at
or below 0.50 after five grace steps. The run seed is an explicit scheduler
input. A prospective resource-only amendment permits at most four independent
four-GPU jobs concurrently after the aggregate storage gate passes.
