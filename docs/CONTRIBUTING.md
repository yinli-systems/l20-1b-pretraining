# Contributing

Create changes on `kevin/<topic>` and open a focused pull request. Do not commit
raw datasets, credentials, model weights, optimizer states or machine-local logs.

Before requesting review:

```bash
python3 tools/verify_repository.py
python3 pretraining/reproducibility/recompute.py
python3 -m pytest -q
```

For new results, include data/endpoint identity, exact question counts, metric
and generation settings, measured versus planned compute, negative outcomes,
retention status and statistical limitations. Keep results from different
splits and wrappers separate. Never overwrite a failed run with a corrected
successful one without preserving the original evidence.

Historical files bound by manifests are immutable. Add a new dated record or
an explicit migration; do not update hashes merely to suppress verification
failures. A clean README is not permission to discard inconvenient results.
