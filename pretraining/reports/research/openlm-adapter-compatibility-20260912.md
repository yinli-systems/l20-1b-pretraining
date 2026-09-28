# OpenLM inference adapter admission

Upstream: [mlfoundations/open_lm](https://github.com/mlfoundations/open_lm),
commit `9bb92ef1689333534b7057942a20d18a46d1fa52`.

The two pinned WebOrganizer checkpoints explicitly select `torch_attn` and
`swiglu_torch`. Upstream nevertheless imports `xformers.ops` at module import
time. The available xFormers wheel targets a different PyTorch/CUDA ABI and
fails before either of those PyTorch implementations can be used.

The isolated copy makes exactly these changes:

1. Move `import xformers.ops as xops` from module scope in `attention.py`
   to the start of `xformers_attn`.
2. Move the same import from module scope in `model.py` into the
   `args.ffn_type == "swiglu"` branch.

There is no catch-and-fallback, model-type rename, architecture change or
weight conversion. Requests selecting an xFormers backend remain unsupported.
The plan rejects OpenLM checkpoints not explicitly selecting both PyTorch
implementations. An AST comparison against the pinned upstream source was
identical after ignoring only those two import placements.

The clean isolated adapter passed CPU import, finite FP32 and BF16 forward,
prefix-causality, exact save/load-key and numerical round-trip tests. These
are tiny-model checks, not real-checkpoint benchmark results. The worker
must additionally pass actual weight hashes, exact HF loading diagnostics,
and finite CUDA logits before evaluating any benchmark.

The original extraction contained macOS auxiliary `._*.json` files, which
upstream attempted to parse as model configs. A separate clean extraction
excludes these files. The previous extraction and failed diagnostics were
retained; training files and its installed environment were not modified.

Exact admitted files are bound in
`../receipts/efficiency-adapter-environment-20260912.json`. Original packages,
unused optional wheels, and generated toy artifacts remain isolated under
the efficiency evaluation directory; no changes were made to global Torch,
NumPy, CUDA, drivers, or the continuation training runtime.
