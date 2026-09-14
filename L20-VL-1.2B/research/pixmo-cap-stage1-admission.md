# PixMo-Cap Stage-1 Admission Review

## Decision

PixMo-Cap is the strongest current candidate for the first real-image alignment pilot, but the full collection is not admitted automatically. The planned path is: verify all frozen metadata, construct a conservative candidate pool, download a small stratified image audit set, measure image availability and image-text integrity, remove benchmark/source-family overlap, complete human review, and only then authorize bounded projector/compressor pilots.

The initial OCR components are excluded from this path. They passed file-integrity checks but failed the natural-English quality screen.

## Why PixMo-Cap leads

Ai2 describes PixMo-Cap as 712,000 distinct images with approximately 1.3 million dense captions collected from 60–90 second human spoken descriptions across roughly 70 topics.^1 The frozen Hugging Face revision contains 717,042 rows and four Parquet metadata shards totaling 1,101,527,794 bytes.^2 The Molmo paper attributes the system's performance primarily to carefully collected PixMo data rather than web-scale noisy image-text pairs.^3

This is unusually relevant to a single-L20 project: dense captions provide substantially more grounded supervision per downloaded image than short alt-text. The official Molmo repository describes captions as roughly 200 words on average, supports captioner pretraining, and warns that URL-hosted images may fail to download.^4 Current Molmo2 training code still uses PixMo-Cap, trains for four passes over the caption set, and mixes transcript-style examples in some configurations.^5

## Risks and claim boundaries

1. **Underlying images.** The frozen schema stores image URLs rather than images. An ODC-BY license on the database does not by itself establish a uniform license for every externally hosted image. Image acquisition therefore remains a separate, source-family-level decision.
2. **Claude-generated captions.** The dataset card states that the cleaned caption was produced from human transcripts using Claude and is subject to Anthropic terms.^2 Anthropic's current commercial terms assign outputs to the customer but also prohibit customers from accessing the service to train competing models.^6 Those terms do not clearly resolve every downstream-dataset question, so this project treats caption use as research-only pending a separate rights decision; the original human transcripts remain a required comparison arm.
3. **Opt-out signals.** The current Hugging Face page reports that some elements have creator opt-out signals, but the frozen three-column schema contains no row-level opt-out field.^2 No row may be represented as opt-out-clean without a separate row-level resolution process.
4. **Availability.** Ai2 notes that URL downloads can fail, and an official repository issue reports one user's observed availability at 77.52%; this is operational evidence, not a guaranteed current rate.^7 The audit must measure availability on the selected frozen sample.
5. **Contamination.** URL and source-family names can expose known benchmark families, but string filtering cannot establish pixel-level non-overlap. Exact and perceptual image comparisons against frozen evaluation artifacts remain mandatory.
6. **Safety and PII.** Metadata regexes are only candidate detectors. They do not prove that images or captions are free of personal information, harmful content, or sensitive documents.

## Frozen admission sequence

### Metadata gate

- Verify all four pinned Parquet sizes and SHA-256 hashes.
- Require exactly 717,042 rows with the frozen `image_url`, `caption`, and `transcripts` schema.
- Measure duplicate URLs and normalized captions, URL schemes and hosts, caption/transcript lengths, English-character coverage, caption-transcript lexical agreement, obvious refusal text, email/phone candidates, and benchmark-family markers.
- Produce counts only in Git; keep raw corpus rows remote.

### Image-audit gate

- Select a deterministic, host-stratified sample from conservative metadata candidates.
- Cap bytes and file count before acquisition; accept only decoded RGB images with recorded source URL, HTTP status, content type, size, SHA-256, dimensions, and perceptual hash.
- Reject redirects to unrelated content, placeholders, tiny images, exact/near duplicates, caption-image mismatches, identifiable sensitive documents, and benchmark-family images.
- Generate a deterministic contact sheet and retain every rejection reason.

### Stage-1 training gate

- Freeze the 1.100B language model and SigLIP2 encoder; train only compressor/projector parameters.
- Screen compression ratios 1, 4, 9, and 16 under identical prediction-token caps.
- Compare true images against no-image and deterministic random-image controls.
- Select by the lower 95% clustered-bootstrap confidence bound of true-vs-random gain per added training FLOP; throughput cannot select a winner by itself.
- Do not advance if any capability bucket fails, if a single source supplies most of the gain, or if image controls show the model is ignoring pixels.

## Current operational status

The pinned metadata acquisition and full-corpus metadata audit are complete. All 717,042 rows passed schema/integrity checks, but metadata alone cannot establish image rights or visual factuality. A deterministic 288-image audit set (32 images from each of nine AI2-hosted families) was downloaded: all 288 decoded successfully, with zero exact duplicates and zero dHash pairs at Hamming distance 1 or 2.

Manual contact-sheet review found mostly good coarse image-caption alignment, but also material family-specific defects: one Stanford Dogs sample depicts a cat while the caption calls it a dog; some captions over-assert species, dishes, gender, or provenance; several compare captions retain formatting residue; and egocentric images require stronger privacy screening. Contact sheets show only caption prefixes and therefore cannot establish sentence-level factuality for full dense captions.

The 288 images are admitted only to a forward-only systems test. Formal training remains blocked, and no training token has been consumed under this review. A forward-only pass can validate decoding, preprocessing, vision encoding, bridge shape, Base integration, memory, and throughput; it cannot demonstrate learning or data quality.

That test is now complete. All 288 real images produced finite outputs through
the frozen SigLIP2 encoder, frozen random bridge, and frozen released Base. At
batch size 4 and a 256-token caption cap, target compression arms 1x, 4x, 9x,
and 16x measured 72.1, 97.9, 103.3, and 108.9 images/s respectively; effective
non-padding caption throughput was 14.7k, 19.9k, 21.0k, and 22.2k tokens/s.
Peak allocated memory ranged from 2.82GiB to 2.61GiB. Every parent and bridge
hash was identical before and after, with no optimizer, backward call, or
training token. Random-bridge losses are intentionally excluded from quality
interpretation; only controlled trained pilots can choose a compression ratio.

## Sources

1. Ai2. “[Molmo: PixMo Data Quality Wins over Quantity](https://allenai.org/blog/molmo).” 2024.
2. Ai2. “[allenai/pixmo-cap Dataset Card](https://huggingface.co/datasets/allenai/pixmo-cap).” Frozen project revision `edce6390d9d5be6c8db0d863fbe62718c88988a4`; current page checked separately for later metadata.
3. Deitke et al. “[Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](https://arxiv.org/abs/2409.17146).” 2024.
4. Ai2. “[allenai/molmo](https://github.com/allenai/molmo).” Official code and data-loading documentation.
5. Ai2. “[Molmo2 Pretraining Configuration](https://github.com/allenai/molmo2/blob/main/launch_scripts/pretrain.py).” Official code.
6. Anthropic. “[Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms).” Effective June 17, 2025.
7. Ai2 Molmo issue tracker. “[Percentage of available images in pixmo](https://github.com/allenai/molmo/issues/37).” User-reported operational observation; not treated as a dataset guarantee.
