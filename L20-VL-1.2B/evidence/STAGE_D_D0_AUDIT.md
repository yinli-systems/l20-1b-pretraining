# Stage D D0 Audit — Scoring, Controls, Adaptation and Fresh Holdouts

Status: D0 passed with a mandatory matched-training replication
Audit date: 2026-09-13
Hardware for scorer audit: one NVIDIA L20 (46,068 MiB)
Test inference during D0: none

## What D0 changed

The Stage C checkpoint difference remains a valid observation, but it cannot be
attributed to compression. The selected `answer_49` branch starts from the
selected full-token checkpoint and then receives 875 additional optimizer
updates on 14,000 training families. The selected `full_196` checkpoint did not
receive that continuation. Stage D therefore replaces the old causal wording
with a matched two-arm comparison.

The two current interfaces also differ in bridge capacity:

| Interface | Bridge parameters | Compressor parameters | Language LoRA parameters |
|---|---:|---:|---:|
| selected full_196 | 5,774,848 | 0 | 12,615,680 |
| selected answer_49 | 10,501,122 | 4,726,274 | 12,615,680 |

`F_matched_196` versus `A_spatial_49` will match data, updates, LR-search
allowance, seed and checkpoint selection. It remains a package-level contrast;
a later `spatial_196` control is required to isolate token count from compressor
capacity and layout.

## Frozen metric contract

The primary metric is base-plus-edited family-joint accuracy. Ordinary
base/edited query accuracy, all-variant accuracy, per-variant accuracy, valid
answer rate, prediction distribution and base/invariant joint accuracy are also
reported for every condition.

This resolves the no-image ambiguity. On the audited 250-family development
subset, no-image family-joint accuracy is 0.0%, while ordinary primary-query
accuracy is 50.0% and valid-answer rate is 100.0%. The scorer emits the same
answer for base and edited questions 100% of the time; because those labels are
opposites, joint correctness is mathematically impossible even though per-query
accuracy is nonzero.

The new random-image control rotates donors within the same task and expected
answer stratum, with zero self-matches. Its family-joint accuracy is 29.6% and
ordinary primary-query accuracy is 50.2%. This supersedes the old global-shift
control for Stage D; historical Stage C numbers are not rewritten.

## Scorer invariance audit

The first BF16 audit is retained as a negative receipt: 7/250 predictions
differed between batch size 1 and batch size 8. Stage D scoring was therefore
frozen to FP32 with TF32 disabled. Under that path, all gates passed exactly:

- same checkpoint rerun;
- batch size 1 versus 8;
- candidate order reversal;
- manifest order reversal;
- 500 nonempty candidate answer spans;
- 750 image files matching manifest hashes;
- deterministic same-task, answer-stratified random control with zero self-links.

The FP32 reference result is 73.2% true-image family-joint accuracy and 86.4%
ordinary primary-query accuracy. These are scorer-audit development figures,
not a new efficacy claim.

## Fresh Stage D data

The first holdout build was rejected before model use because human review found
combined binding/geometry interventions and sampled clipping/overlap. The v2
build separates `binding_swap_only` from
`binding_swap_plus_boundary_geometry`, forbids overlap and requires all objects
to be fully visible.

The accepted v2 manifest contains 7,000 scene families and 21,000 unique images:

| Partition | Families | Status |
|---|---:|---|
| development | 2,000 | admitted for LR and checkpoint selection |
| IID test | 3,000 | sealed |
| OOD binding | 2,000 | sealed |

There is zero exact family or image overlap with the Stage C corpus. Twelve
contact sheets covering every split-task pair were manually checked (144
families / 432 rendered images). Labels, answer-changing edits, invariant edits,
binding strata, visibility and non-overlap passed. Human review is sample-based;
it is not exhaustive proof of semantic correctness.

## Integrity anchors

- Static audit: `6f58c55ed26a172320dc56b5d9ac570bc6ae3e33ecfe4fe5f65f9cfd7d154014`
- Initial BF16 scorer audit: `a30c93b54d073be6f30e7882b3893277fa7765660e48e8c37fd594adbcef3530`
- Passing FP32 scorer audit: `83451584380db70431cc20ffc8bbb781438a374f9ecd444e39715c25ea30cec7`
- v2 data protocol: `0e776b0ac5cf466a3ffdc98768c41a843fffd1ffd9188d0fac53bea8b416a4fe`
- v2 data admission: `dd33bd6fdd472b7b32cc56f9a70cb072c1158b03a7cfec6c2bdf645195b12d1c`
- v2 generation receipt: `0d3776303b08d28a401f06705cf8cd38f834124ef91db18d366773aee885cfe2`
- v2 manifest: `efd51701c19379ce77af83272acd732c93cc7e4428ca67c6cb147bea0572c6cb`
- v2 human audit: `318a04a9ed2858babc22ae1de8360553f8214c4372533b490f30ae02028589ac`
- Superseded LR-screen v1 protocol: `e583176abb8b04a649b00c7ecbf5ebc6449629687147c48dee0a8059d4cd2b1c`
- Answer-balanced LR-screen v2 protocol: `2fec2dfc6dc6c98bc6f805a6ef62649061ea22a383594a4acf08b2cc5d2a93dd`

## Claim boundary

D0 validates the scorer, controls, training-opportunity diagnosis and sampled
synthetic holdouts. It does not establish an accuracy gain from compression,
multi-seed stability, natural-image transfer, or broad VLM quality. IID and OOD
test inference remains prohibited until all ten D1 runs and checkpoint choices
are frozen under a separately hashed unseal receipt.
