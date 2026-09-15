# Paired F2 continuation pilot

Each Slurm job requests one node with four public RTX 5090 GPUs. It starts from
the exact model-only final checkpoint of one completed F2 seed, initializes a
new optimizer, and trains 536,870,912 prediction tokens with the frozen F2
source quotas. The runner validates the hash-bound pilot admission and new
document-masked development data before training. The rolling ten-step MFU must
stay strictly above 0.70 after five grace steps; loss must remain finite. Full
optimizer/reader/RNG checkpoints are saved at steps 128 and 256.

The two seeds are comparable paired pilot runs. Their completion is not a
model-promotion or market-superiority claim; reserved confirmation and
independent capability/retention checks follow.
