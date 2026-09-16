#!/usr/bin/env python3
"""GPU compatibility check for xFormers attention on the target runtime."""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
import xformers
from xformers.ops import memory_efficient_attention


def main() -> None:
    torch.manual_seed(20260917)
    device = torch.device("cuda")
    shape = (16, 257, 24, 64)  # batch, tokens, heads, head width
    tensors = [
        torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    query, key, value = tensors
    reference = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)
    candidate = memory_efficient_attention(query, key, value)
    max_absolute_error = (candidate.float() - reference.float()).abs().max().item()
    candidate.float().square().mean().backward()
    gradients_finite = all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all().item()
        for tensor in tensors
    )

    def xformers_step() -> None:
        memory_efficient_attention(query, key, value).float().square().mean().backward()

    def sdpa_step() -> None:
        F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).float().square().mean().backward()

    def benchmark(step_function) -> float:
        for _ in range(4):
            step_function()
            for tensor in tensors:
                tensor.grad = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            step_function()
            for tensor in tensors:
                tensor.grad = None
        torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000 / iterations

    for tensor in tensors:
        tensor.grad = None
    xformers_milliseconds = benchmark(xformers_step)
    sdpa_milliseconds = benchmark(sdpa_step)
    print(
        json.dumps(
            {
                "status": "PASS" if gradients_finite else "FAIL_NONFINITE",
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "xformers": xformers.__version__,
                "shape_bmhd": shape,
                "max_absolute_error_vs_sdpa": max_absolute_error,
                "gradients_finite": gradients_finite,
                "xformers_forward_backward_milliseconds": xformers_milliseconds,
                "sdpa_forward_backward_milliseconds": sdpa_milliseconds,
                "xformers_speedup_over_sdpa": sdpa_milliseconds / xformers_milliseconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
