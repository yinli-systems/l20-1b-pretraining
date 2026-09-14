# Unique-data expansion intake v1

This intake covers the measured unique-token deficits for the frozen F2 and F3
2.147B-token, two-seed confirmation runs. `plan.py` derives every target from
the exact short-screen manifests and the two-epoch per-source cap. Acquisition
adds a 35% reserve for downstream quality, decontamination, family splitting,
and packing loss; raw outputs remain `NOT_ADMITTED` until those gates pass.

The intake reuses completed DCLM/PDF v3 raw extensions, reads unused row groups
from the existing pinned shards, and adds one FineMath shard at the same pinned
revision because shard 53 does not contain enough unused groups. Code content
is accepted only for the allowlisted permissive licenses, rehydrated from
Software Heritage, checked against its SHA-1 blob identity, and syntax parsed
for Python. The live implementation uses 64 code-object workers per language
and ten source workers after the 16-worker launch measured excessive small-file
latency. Each output records exact physical provenance and range hashes.

Completed segments are idempotent. If a segment failed, its partial receipt and
file are moved into `attempts/` before retry; the process-wide file lock prevents
two writers. The program enforces finite row, byte, network, code-attempt, code
payload, and disk-headroom bounds.
