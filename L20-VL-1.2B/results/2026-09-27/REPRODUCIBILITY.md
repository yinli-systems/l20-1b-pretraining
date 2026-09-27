# Reproducibility and Evidence Hashes — 2026-09-27

This file pins the final large unique-image scaling run and its reporting artifacts.

## Final scale run

- parent checkpoint manifest SHA-256: `e53cff13cb1e53e18310152eba26b419a245b77201fffcc80ea33079f05c622d`
- data-admission SHA-256: `1789a53c27fe4a01a77e77b98de6b9d6ec33097f6ee4662bd87d9f2ff6fee79b`
- canonical training-plan SHA-256: `2dc21bf0ce79bfa4e8c860fe97345e268cf244ea6716c9e646929631943a892f`
- frozen evaluation selection SHA-256: `dfe3c7cff11851b115a6323e1d56374eee75956b4002a5e5ef9d36b840683d17`
- trainer source SHA-256: `a001cf8f32f962d5efd20825c9f7d686ff988ef6b66357f17332a57491a5cb80`
- final step-001934 checkpoint manifest SHA-256: `11fce0dc36e742fba02457aaed43fae7dd5a40f942734ea5215b4c592e26353c`
- final run summary SHA-256: `ae256755807839f867adcd9dcf37c5b383aae8cc53b916873e0e122c00642ff1`
- parent-comparison SHA-256: `59fded4693979b96662db3f710d2563925e79c28a48fac34e47efd78cd5c3814`

## Frozen run shape

- steps: **1,934**
- global image batch: **64**
- new high-resolution images per update: **32**
- new high-resolution route: **448px / 784 visual tokens**
- old retention route: **224px / 196 visual tokens**
- trainable parameters: **10,280,448**
- new-image events: **61,888**
- global image events: **123,776**
- planned unique photos: **31,178**
- unique train image clusters: **30,958**
- maximum cluster exposure: **2**
- training seed: **2026092718**
- checkpoints: 32, 256, 512, 768, 1024, 1280, 1536, 1792, 1934

## Evaluation discipline

The final endpoint was frozen before the final confirmation results were accessed. Confirmation rows were not used for stopping or checkpoint selection.

The final comparison uses exactly the same confirmation image/question rows for the immutable parent and the final endpoint. Paired cluster bootstrap is applied to the per-row score difference.

## Retention discipline

The training protocol stopped only after two consecutive retention-gate failures. The final run had **zero** consecutive retention failures.

Frozen retention tolerances:

- old QA absolute drop <= 0.05
- old caption NLL increase <= 0.08
- text NLL increase <= 0.05

## Data provenance boundary

The scale run uses pinned `HuggingFaceM4/the_cauldron` source shards at revision:

`847a98a779b1652d65111daf20c972dfcd333605`

All 21 acquired whole source shards were verified against upstream LFS SHA-256 metadata before admission.

The project additionally checks recorded historical image IDs, exact hashes, 224px pixel hashes and registered near-image clusters. This is not an exhaustive semantic/document-level contamination proof.

## Public-repo boundary

The public repository intentionally does not contain:

- raw source images;
- raw source Parquet shards;
- language-model or vision-model weights;
- incremental checkpoints;
- optimizer states;
- private machine paths as reproducibility requirements.

The public results package is therefore an evidence and protocol record, not a self-contained weight release.
