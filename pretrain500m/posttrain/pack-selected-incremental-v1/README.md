# Incremental selected-source packing

This packer can reuse a completed source artifact only after verifying the prior
pack report hash, tokenizer hash, every array/index/tail hash, NumPy shape and
dtype, token/block accounting, and the exact selected row, partition, family,
offset, and EOS sequence against the new family assignment. Any changed source
is repacked in the fresh output directory; unchanged source paths remain bound
to their prior immutable artifacts. No pack report grants training admission.
