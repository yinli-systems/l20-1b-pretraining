# CPT long confirmation runner

This is the qualified four-GPU CPT runner with the launch step cap extended to
2048 so a 2,147,483,648-token confirmation can finish in one allocation. It
retains the exact admission, initialization, validation, deterministic DDP,
rolling ten-step MFU, finite-metric, data-epoch, and atomic output checks.

Confirmation outputs are model-only and non-resumable. They select the recipe;
a later single finalist run must use the separately qualified full-checkpoint
path before promotion.
