# F2 fresh-N1 independent confirmation v1

This bundle spends the previously untouched confirmation halves of four public
proxies exactly once.  It compares the seed-20260915 long-F2 parent with the
frozen `3e-5`, 256-step fresh-N1 continuation on matched RTX 4090 hardware.

The candidate was frozen from the matched seven-task screen at git commit
`6498ccb`.  The four confirmation datasets, split manifests, prompts, metrics,
tokenizer, checkpoint identities, and decision rule are frozen in
`gate-plan.json` and the four domain `plan.json` files before scoring.

Each domain is one 2-GPU job: one GPU evaluates the parent and one evaluates the
candidate.  `submit.py` verifies all bundle files, both multi-gigabyte
checkpoints, storage headroom, scheduler capacity visibility, and duplicate-job
absence before submitting.  `run-domain.sbatch` verifies physical RTX 4090
identity and all inputs again inside each allocation.

Passing this screen only authorizes a second-seed replication.  It is not a
formal model promotion or evidence of broad or market superiority.
