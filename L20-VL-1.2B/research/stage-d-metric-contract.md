# Stage D Metric Contract

Status: frozen before Stage D model selection
Old Stage C test: regression only; forbidden for Stage D tuning or checkpoint selection

## Primary unit and score

The atomic statistical unit is one `scene_family_id`, containing the base,
answer-changing edited, and answer-invariant variants. The primary score is
base-plus-edited family-joint accuracy: a family is correct only when both the
base and edited answers are correct. Families, not individual variants, are
resampled for paired confidence intervals.

The primary D1 comparison is `A_spatial_49 - F_matched_196` on the new IID test.
It is evaluated only after both arms, all five paired seeds, learning rates, and
checkpoint choices are frozen from development data.

The LR screen and checkpoint-selection subset contains exactly 50 families per
task, stratified to 25 `base_answer=no` and 25 `base_answer=yes`. Selection is
deterministic after sorting by `scene_family_id` within each task-answer
stratum. The first task-only-prefix LR attempt was invalidated before any
development inference and is excluded from all summaries.

## Required companion metrics

Every evaluation must also report:

- Ordinary per-query accuracy over base and edited queries separately counted.
- Ordinary accuracy over all base, edited, and invariant variants.
- Per-variant accuracy.
- Base-plus-invariant family-joint accuracy.
- Valid-answer rate. In candidate scoring this means the selected answer is one
  of the manifest candidates; it is not a free-generation parser metric.
- Prediction distribution for each condition and variant.
- Per-task versions of the same metrics.

## No-image interpretation

Base and edited examples share a question but have opposite answers. A
deterministic question-only scorer normally emits the same answer for both, so
its family-joint accuracy is mathematically zero even when ordinary per-query
accuracy is nonzero. Therefore no-image family-joint, ordinary accuracy,
same-prediction rate, validity, and answer distribution must be read together.

## Random-image controls

Legacy Stage C evidence used a global cyclic image shift. That result remains
unchanged but its random-image interpretation is limited because donors could
cross task strata.

Stage D uses a deterministic donor rotation within each task and expected-answer
stratum. It requires zero self-matches, 100% same-task donors, and 100%
answer-stratum matching. This is a conservative nuisance control that breaks
exact-family pairing without adding cross-task render shifts or forced label
anticorrelation. It is not a natural-image performance estimate.

## Integrity and invariance gates

Before D1 training, the selected reference scorer must pass exact prediction
agreement for a same-checkpoint rerun, batch size 1 versus the reference batch,
reversed candidate order, and reversed manifest order. Every candidate answer
span must be non-empty. Every evaluated image requested by the audit must match
its manifest SHA-256.

Stage D confirmation scoring uses float32 weights and activations with TF32
disabled. The original Stage C bfloat16 metrics remain historical evidence and
are not silently rewritten. This higher-precision scorer is used consistently
for every new arm and seed because D0 found seven of 250 development families
changed decisions between batch size 1 and 8 under the legacy bfloat16 path.

The legacy teacher cache is required to match family, task, candidate order, and
expected candidate index. It does not contain image hashes. D1 answer-only
training must not rely on teacher scores, and image integrity is checked against
the manifest independently.

`F_matched_196` matches training opportunity but does not match the additional
4,726,274 compressor parameters in `A_spatial_49`. Their D1 difference is a
package-level contrast. A capacity-matched `spatial_196` arm is required before
describing the result as a token-count-only effect.

## Claim boundary

Passing these gates validates scorer and control integrity. It does not establish
efficacy, compression-caused accuracy gains, multi-seed robustness, natural-image
transfer, or production serving performance.
