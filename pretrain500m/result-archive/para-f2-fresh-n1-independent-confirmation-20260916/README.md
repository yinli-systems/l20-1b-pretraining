# F2 fresh-N1 independent confirmation — 2026-09-16

The frozen `f2n1-lr3e5` step-256 continuation was compared with its matched
seed-20260915 F2 parent on the previously untouched confirmation halves of four
public proxies. Each domain used one 2×RTX 4090 job, one GPU per checkpoint.
All four scheduler jobs completed with exit code 0, empty stderr, distinct
physical RTX 4090 UUIDs, and verified output manifests.

| Domain | Samples | Parent primary | Candidate primary | Delta | Frozen gate |
|---|---:|---:|---:|---:|---|
| Belebele reading `acc_norm` | 450 | 0.337778 | 0.326667 | −0.011111 | fail; paired 95% delta CI [−0.024444, 0.002222] crosses the −0.01 margin |
| MGSM exact numeric match | 125 | 3/125 (0.024) | 2/125 (0.016) | −0.008 | fail; one fewer exact answer, despite NLL improving 7.917026→7.810106 |
| MBPP execution pass@1 | 129 | 0/129 | 0/129 | 0 | fail; candidate did not break the zero-pass floor, despite code NLL improving 1.755244→1.743311 |
| SciQ `acc_norm` | 490 | 0.624490 | 0.634694 | +0.010204 | pass; paired 95% delta CI [−0.002041, 0.022449] clears the −0.01 margin |

The pre-frozen cross-domain rule required all four domain gates and at least two
strict primary improvements. Only knowledge improved and passed. The resulting
status is `HOLD_CANDIDATE_NO_AUTOMATIC_CONTINUATION`; this checkpoint is not
promoted and should not be continued automatically.

Jobs: reading 1595612 (2:01), math 1595613 (4:30), code 1595614 (22:43), and
knowledge 1595615 (1:41). The exact source protocol is commit
`1f87f7ad0f4e1686b867b12a8dede220db07f64e` on `codex/529m-results`.

These are one-time public proxy confirmation results. They do not establish
sealed generalization, instruction following, safety, multilingual quality, or
market superiority.
