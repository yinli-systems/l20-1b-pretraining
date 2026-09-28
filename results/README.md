# Results

Different rows below use different evaluation scopes. They are a lineage, not
a single comparable leaderboard or a fitted scaling law.

| Study | Scope | Outcome / entry point |
|---|---|---|
| Released 1.1B language base | Frozen language benchmarks, 20B prediction tokens | [Model card](../pretraining/MODEL_CARD.md), [full evidence](../pretraining/reports/README.md) |
| Earlier VLM pilots | General vision, replay, capacity and resolution controls | [September 27 archive](../L20-VL-1.2B/results/2026-09-27/RESULTS_SO_FAR.md); negative and inconclusive arms retained |
| ~31k-photo two-pass expansion | Local image-disjoint confirmation from source training corpora | Same archive; TextVQA +4.38 pp, DocVQA +3.43 pp, ChartQA +5.86 pp; not full official splits |
| Streamed expansion | Full published splits, 15,937 questions | [September 28 report](2026-09-28/README.md); TextVQA/DocVQA improve, ChartQA unresolved, AI2D format confound, retention guard fails |

The latest package distinguishes 165,376 prepared images from 163,840 used
new-image events and a 250,000-image ceiling. No result in this index implies
a released competitive general-purpose VLM.
