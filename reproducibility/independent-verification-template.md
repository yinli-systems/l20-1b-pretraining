# Independent verification record

Status: **pending independent verifier**

## Identity

- Verifier:
- Organization or affiliation (optional):
- Verification UTC date:
- Repository commit:
- Model revision: `7fc9ab0f50e7faf34c2487fefdf46aa99f4a50b8`
- OS, architecture, and Python version:
- GPU and software environment for evaluation, if run:

## Required clean-checkout verification

```bash
python3 reproducibility/recompute.py > reproduction-output.json
sha256sum reproduction-output.json
```

- Exit code:
- Output SHA-256:
- Bundle status:
- Full-run time-weighted MFU:
- Full-run step-weighted MFU:
- Covered-interval tokens/s:
- Final validation loss and perplexity:
- Full-run energy remains `null`: yes / no

## Optional source-event verification

- Source TensorBoard event SHA-256:
- Re-exported CSV SHA-256:
- Matches `d656e87efa759f34036bf59bcf6eb950adf5f22abbfc9375178d1111da9070c3`: yes / no

## Optional exact-weight verification

- `verify_hf_release.py` exit code:
- Parameter count:
- Tensor count:
- All managed hashes and safetensors checks passed: yes / no

## Optional independent evaluation

- lm-eval version and full command:
- Raw prediction/sample-log location and SHA-256:
- Task-config hashes:
- Regenerated seven-task scores:
- Regenerated paired interval, if item-level outcomes were retained:

## Exceptions and conclusion

- Deviations from the documented protocol:
- Failed or skipped checks:
- Claim(s) independently verified:
- Claim(s) not independently verified:
- Verifier signature or signed commit:
