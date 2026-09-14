# Stage-A Full-Token Foundation Data Plan

**Frozen 2026-09-13 — acquisition authorized, training conditional on audit**

## Decision

Use two complementary sources for the bounded 25,000-example Stage-A bridge
pilot:

1. **CLEVR v1.0** for exact, programmatically verifiable attributes, counting,
   comparison, and spatial reasoning.
2. **Open Images Localized Narratives** for natural-image diversity and
   human-authored descriptions, restricted to a deterministic subset whose
   image-level license and attribution metadata are retained.

The pilot is deliberately small and balanced: 12,500 training examples from
each source. Its purpose is to determine whether the frozen SigLIP2 encoder can
be aligned to the released 1.1B language model strongly enough to support a
later compression diagnosis. It is not intended to produce a general-purpose
or release-ready VLM.

## Primary-source basis

The [official CLEVR page](https://cs.stanford.edu/people/jcjohns/clevr/)
describes 70,000 training images, 15,000 validation images, scene graphs,
functional programs, and a CC BY 4.0 release. The official 18GB v1.0 archive is
pinned by URL and exact Content-Length before acquisition. Only train and
validation are allowed; the test split is explicitly prohibited.

The [Localized Narratives project](https://google.github.io/localized-narratives/)
states that its annotations are CC BY 4.0 and that the final captions were
manually produced by annotators. The Open Images portion provides over 500,000
training narratives. The [Open Images V7 documentation](https://storage.googleapis.com/openimages/web/download_v7.html)
provides image IDs, original landing pages, author and title fields, image-level
license URLs, and an official subset downloader. Open Images images in this
pipeline must carry the exact CC BY 2.0 URL plus non-empty author and landing
page metadata.

These statements support bounded research use and attribution preservation;
they are not a warranty that every underlying record is free of rights,
privacy, or factuality problems. That residual risk is why the source is
filtered, sampled, manually audited, and not automatically approved for
release.

## Why other candidates were not selected

- The NVIDIA Llama-Nemotron VLM collection has useful annotations and a
  collection-level permissive framing, but its own card warns that distributed
  models trained on some generated material may inherit Meta Llama terms. Two
  previously sampled synthetic OCR components also failed this project's human
  language-quality audit.
- The Cauldron is a collection whose sub-datasets retain component-specific
  licenses. A collection-level label cannot replace component-level review.
- PixMo-Cap metadata and a 288-image sample were audited earlier, but unresolved
  image-origin rights, privacy, factuality, and benchmark-family overlap prevent
  training admission.

## Frozen automated gates

For Localized Narratives, the pipeline:

- accepts 12–100-word predominantly ASCII captions;
- rejects email, phone, URL, person-related, sexual, violent, medical, identity,
  and other high-risk markers;
- requires exact per-image license, author, and original landing-page metadata;
- requires successful image decoding, both dimensions at least 224 pixels,
  aspect ratio at most 4:1, and a non-blank pixel-variance floor;
- removes exact images, normalized exact captions, and images within dHash
  Hamming distance four;
- preserves caption, annotator, image URL, original URL, landing page, author,
  title, license, hashes, dimensions, and deterministic selection hash.

For CLEVR, the pipeline selects at most one deterministic question per image,
preserves the functional program and question family, verifies image decoding,
and keeps official split boundaries.

Both sources are checked for exact image-hash overlap with the controlled
counterfactual diagnostic. Two deterministic 100-image contact sheets are then
reviewed manually. Training remains blocked until the review receipt pins both
manifest hashes and explicitly returns
`pass_for_stage_a_research_training`.

## Budget and stop rules

The frozen maximum acquisition envelope is 27,798,530,964 bytes, below the
user-authorized 50GB limit. That envelope includes the 19,021,600,724-byte CLEVR
archive, Localized Narratives captions, Open Images metadata, and an 8GB ceiling
for selected image files.

The full-token training pilot has a hard 12-hour wall cap, one L20, 25,000
examples, 781 optimizer steps, BF16, effective batch size 32, and only the
projector plus visual boundary embeddings trainable. The language and vision
parents remain frozen and are hash-checked before and after.

Training stops on non-finite loss or gradient, OOM, artifact mismatch, disk
boundary failure, or the wall-time cap. Completion of the optimization run does
not authorize compression experiments: the full-token model must still pass
the predeclared paired-joint and true/no/random-image visual-use floor.
