# Stage D Learning-Rate Screen

Status: complete and independently audited
Selection data: 250 fresh development families per trial
Selection strata: 50 per task, exactly 25 `base_answer=no` and 25
`base_answer=yes`
Test predictions produced: 0

## Results

| Arm | LR multiplier | Family joint | Ordinary base/edited | Train wall | Input+visual tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| F_matched_196 | 0.5x | **72.8%** | **86.2%** | 502.9 s | 6,119.2 | 7.900 GiB |
| F_matched_196 | 1.0x | 72.4% | 86.0% | 502.9 s | 6,119.6 | 7.900 GiB |
| F_matched_196 | 2.0x | 65.2% | 82.0% | 502.7 s | 6,121.2 | 7.900 GiB |
| A_spatial_49 | 0.5x | 54.4% | 76.8% | 138.5 s | 6,934.2 | 4.338 GiB |
| A_spatial_49 | 1.0x | **62.4%** | **81.2%** | 138.3 s | 6,943.0 | 4.338 GiB |
| A_spatial_49 | 2.0x | 61.6% | 80.0% | 138.4 s | 6,939.5 | 4.338 GiB |

The frozen lexicographic rule selects `F_matched_196=0.5x` and
`A_spatial_49=1.0x`. The screen used 150 optimizer steps and one seed. These
scores select optimization settings only; they are not evidence that the
full-token arm wins after 875 steps or across seeds.

The 49-token arm used 45.1% less peak allocated VRAM and had 13.5% higher
reported input-plus-visual token throughput in this short training screen. The
token-throughput numerator differs by visual-token count, so it is not by
itself an end-to-end speedup measurement.

## Cost and audit

- Training GPU time: 0.5475 hours.
- Development evaluation GPU time: 0.0614 hours.
- Total: 0.6089 L20 GPU-hours, below the 2-hour cap.
- Every run completed 150 steps with no teacher cache.
- All six development files contain exactly the frozen task-answer strata.
- All selected checkpoints were hash-bound and optimizer/nonselected state was
  removed under a receipt.
- Runner-selected multipliers were independently recomputed from the frozen
  rule.

## Integrity anchors

- LR protocol: `2fec2dfc6dc6c98bc6f805a6ef62649061ea22a383594a4acf08b2cc5d2a93dd`
- Training smoke: `b1356dfb7149ee5db7d77eded04a40f92654b3c53d7138e4038cbd27348f6352`
- LR summary: `a69087907e9649c264dc7f2e4f7618a4f369f98455e39e2478402f24664017c8`
- Independent LR audit: `997aa30b18f60192a4a642dacde00306973e68bd253cb5257ed90949b2a2ef9d`
- Frozen D1 protocol: `dd04741069c4be356156b61dfba9105bccbcdb7efe78030a08a996b546f86d5e`

## Claim boundary

This is a development-only hyperparameter screen. It does not establish
efficacy, multi-seed stability, a compression-caused accuracy gain, natural
image transfer, or production performance. The old Stage C test and the new
IID/OOD tests were not used.
