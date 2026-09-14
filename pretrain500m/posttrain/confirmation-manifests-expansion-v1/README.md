# Long confirmation input builder

Builds exact 2,147,483,648-token F2 and F3 manifests by multiplying the frozen
screen block quotas by four. It uses the original frozen FineWeb shards and the
new combined selected-source packs. Every non-FineWeb quota must remain within
two passes over its packed training blocks.

Packed `.npy` inputs are validated through their NumPy header, exact one-dimensional
token shape, `uint16` dtype, bound SHA-256, and block metadata. The container header
is not counted as token payload.

The builder binds all audit, contamination, exclusion, quality, family, pack,
design, validation, and parent-screen evidence. It refuses shortages and writes
content-addressed admission receipts for the long confirmation runner.
