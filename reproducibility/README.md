# Independent reproduction package

This directory is the smallest fail-closed package for checking the published
L20-1B-20B-Base training and evaluation claims. It separates three evidence
classes that must not be conflated:

| Evidence | Result | Measurement boundary |
|---|---:|---|
| Full-run, time-weighted MFU | 71.06% | 19,144/19,148 optimizer steps; model FLOP/s divided by 132 TFLOP/s |
| Full-run, step-weighted MFU | 71.09% | Same logger samples, equal weight per optimizer step |
| Step-10,595 MFU snapshot | 71.17% | One observation only; not the run average |
| Covered-interval throughput | 12,824 tokens/s | 19,144 logger intervals; excludes steps 1-4 |
| Successful supervisor elapsed time | 433.24 h | One L20; elapsed GPU-hours, not a billing receipt |
| Data preparation | 22.97 h calendar | First full-data attempt to admitted receipt; final admitted attempt was 11.15 h |
| Training-launch recovery | 24.23 min | Failed before optimizer step 1; zero completed updates lost |
| Full-run energy | unavailable | No run-spanning power time series was preserved |

`GPU utilization = 100%` and `MFU = 100%` are different statements. The former
is a busy-time counter; the latter compares measured model arithmetic to a
declared peak denominator. The 348.3 W point sample and the short gate trace are
not extrapolated into training kWh.

## 1. Pin the repository and verify offline

Use the exact public commit containing this package. The manifest records its
parent commit because a file cannot bind the hash of the commit that first adds
itself.

```bash
git clone https://github.com/yinli-systems/l20-1b-pretraining.git
cd l20-1b-pretraining
git checkout l20-1b-repro-v1
python3 reproducibility/recompute.py
```

The verifier uses only the Python standard library. It checks every
manifest-bound file hash, the surviving execution-source hashes, timing
arithmetic, data token totals, validation loss/perplexity, full-run telemetry
aggregates, the compact benchmark result, the TinyLlama-1T paired sufficient
statistics, and the fixed model-release identity. Any hash or numeric mismatch
returns a non-zero status.

Expected headline output includes:

```text
"status": "verified"
"time_weighted_mfu": 0.710558979886...
"aggregate_tokens_per_second": 12824.189835...
"final_validation_perplexity": 11.299003601...
```

The checked-in evaluation files are compact results, not all raw model
responses. Task fingerprints and paired outcome counts are retained for the
frozen comparison; the reported bootstrap CI is hash-bound but cannot be
independently replayed exactly without the original item-level outcome files.
An independent evaluator should therefore regenerate raw predictions with
`--log_samples` before treating the CI as independently reproduced.

## 2. Re-export the full-run telemetry

The original TensorBoard event file is 16,473,400 bytes with SHA-256:

```text
295a8a4cfa71709450b559029cf13830781dc7fa9d505e93d6f1ed227a333407
```

In a clean environment, copy that exact file without renaming it—the filename
must contain `tfevents`—then run:

```bash
python3 -m venv .venv-event
. .venv-event/bin/activate
python -m pip install --requirement reproducibility/requirements-event-export.txt
python reproducibility/export_training_telemetry.py \
  /path/to/events.out.tfevents.1787553807.hhai-zijun.2957779.0 \
  --output /tmp/full-training-telemetry.csv
sha256sum /tmp/full-training-telemetry.csv
```

The expected CSV SHA-256 is:

```text
d656e87efa759f34036bf59bcf6eb950adf5f22abbfc9375178d1111da9070c3
```

The exporter verifies the event hash, contiguous optimizer-step scalars, final
token count, aligned throughput/FLOP samples, and positive telemetry before it
writes an output. It refuses to overwrite an existing file.

## 3. Retrieve and verify the exact weights

The model identity is:

```text
AliceYin/L20-1B-20B-Base@7fc9ab0f50e7faf34c2487fefdf46aa99f4a50b8
```

With Git LFS installed:

```bash
git clone https://huggingface.co/AliceYin/L20-1B-20B-Base model-release
git -C model-release checkout 7fc9ab0f50e7faf34c2487fefdf46aa99f4a50b8
git -C model-release lfs pull
python3 verify_hf_release.py model-release
```

The standard-library verifier checks every managed file digest, all three
safetensors layouts, the index-to-tensor mapping, 201 tensors, and
1,100,048,384 parameters. This verifies release identity and structure, not
benchmark quality.

## 4. Regenerate evaluation results in a clean GPU environment

The recorded evaluation environment used Python 3.12.3, PyTorch 2.12.1+cu130,
Transformers 4.56.2, and lm-eval 0.4.9. The frozen seven-task comparison used
BF16, context length 2,048, zero-shot prompting, `batch_size=auto:4`, no chat
template, and seeds `42/42/42/1234` for Python/NumPy/PyTorch/few-shot sampling.

Run lm-eval 0.4.9 against the exact local release for:

```text
hellaswag,piqa,winogrande,openbookqa,arc_easy,arc_challenge,boolq
```

Enable sample logging and retain both the raw output hash and task-config hash.
Compare the regenerated task metrics to
`reports/metrics/efficiency-all-results-final-20260912.json`. BoolQ must also be
reported as a sensitivity exclusion because it was not explicitly included in
the original corpus decontamination list.

## Provenance boundary

`training-source-manifest.json` hashes the source files recovered from the
surviving execution host. Six files differ from the earlier environment
receipt because operational fixes occurred after that receipt. Their host
mtimes place the recovered copies before the relevant stage, but mtimes are not
cryptographic pre-registration. Accordingly, this package calls them a
reconstructed execution snapshot, not an immutable pre-run commit.

The surviving `pipeline_config.json` is an early planning snapshot. Source
counts in `reports/receipts/data-full-receipt.json`, not that planning file, are
authoritative for the actual packed corpus.

## Independent sign-off

The author-side clean-directory check is recorded, but it is not a substitute
for another person's verification. A reviewer should copy
[`independent-verification-template.md`](independent-verification-template.md),
fill it from a fresh checkout, attach the verifier output, and commit the signed
record without changing the frozen evidence manifest.

## Energy collection for a future run

For a future training run, begin GPU power sampling before process launch and
stop it only after the final checkpoint is durable. Record monotonic and UTC
timestamps, `power.draw`, the sampling interval, GPU UUID, process lifecycle,
missing intervals, and host/restart events. Integrate only measured intervals
and report coverage; do not silently fill gaps. GPU telemetry still excludes
CPU, RAM, storage, networking, cooling, and facility overhead, which require a
separately identified PDU or provider measurement. A single `nvidia-smi`
reading must never be multiplied by wall time and presented as run energy.
