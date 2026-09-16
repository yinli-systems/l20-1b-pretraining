#!/usr/bin/env python3
"""Reconstruct a MobileLLM PyTorch snapshot from its public full-precision ONNX export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import numpy_helper
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected-graph-sha256", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    args = parser.parse_args()
    graph_path = args.snapshot / "onnx/model.onnx"
    data_path = args.snapshot / "onnx/model.onnx_data"
    if digest(graph_path) != args.expected_graph_sha256:
        raise ValueError("ONNX graph digest mismatch")
    if digest(data_path) != args.expected_data_sha256:
        raise ValueError("ONNX external-data digest mismatch")

    graph = onnx.load(graph_path, load_external_data=True)
    initializers = {item.name: item for item in graph.graph.initializer}
    nodes = {item.name: item for item in graph.graph.node}
    config = AutoConfig.from_pretrained(
        args.snapshot, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    expected = model.state_dict()
    converted = {}
    mapping = {}
    for key, reference in expected.items():
        if key in initializers:
            array = numpy_helper.to_array(initializers[key])
            source = key
            transpose = False
        else:
            if not key.endswith(".weight"):
                raise ValueError(f"unmapped state entry: {key}")
            module = key.removesuffix(".weight")
            components = module.split(".")
            if components[:2] != ["model", "layers"] or len(components) < 5:
                raise ValueError(f"unexpected linear-module path: {module}")
            node_name = f"/model/layers.{components[2]}/" + "/".join(components[3:]) + "/MatMul"
            node = nodes.get(node_name)
            if node is None or len(node.input) != 2 or node.input[1] not in initializers:
                raise ValueError(f"missing ONNX MatMul weight for {key}")
            source = node.input[1]
            array = numpy_helper.to_array(initializers[source]).T
            transpose = True
        tensor = torch.from_numpy(array.copy())
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(
                f"converted tensor mismatch for {key}: {tensor.shape}/{tensor.dtype} "
                f"!= {reference.shape}/{reference.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite converted tensor: {key}")
        converted[key] = tensor.contiguous()
        mapping[key] = {"onnx_initializer": source, "transpose": transpose}
    loaded = model.load_state_dict(converted, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("strict reconstructed state load failed")
    if sum(item.numel() for item in converted.values()) != 603188352:
        raise ValueError("unexpected reconstructed parameter count")

    from safetensors.torch import save_file
    output = args.snapshot / "model.safetensors"
    save_file(converted, output)
    receipt = {
        "schema": "mobilellm-600m-onnx-to-pytorch-reconstruction-v1",
        "status": "STRICT_STATE_RECONSTRUCTION_COMPLETE",
        "source": {
            "repo_id": "onnx-community/MobileLLM-600M",
            "revision": "25723a2c3beb2698878a620e4edf7bd17f5bd339",
            "declared_base_model": "facebook/MobileLLM-600M",
            "onnx_graph_sha256": args.expected_graph_sha256,
            "onnx_external_data_sha256": args.expected_data_sha256,
        },
        "official_code_blob_identity": {
            "configuration_mobilellm.py": "e291643d7582176f3c3da500e73a171e49b313e7",
            "modeling_mobilellm.py": "e1558bd14d9d976cd206055764cd7008beb2a1fd",
            "matched_via_public_code_mirror": "RoyArkh/02-12-facebook-MobileLLM-125M_xsum_client7_round1@5b76806d7421687982486908821da2439291e57a",
        },
        "state_entries": len(converted),
        "parameters": sum(item.numel() for item in converted.values()),
        "model_safetensors_sha256": digest(output),
        "mapping_sha256": hashlib.sha256(json.dumps(
            mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "validation": {
            "all_state_keys_and_shapes_matched": True,
            "strict_pytorch_state_load": True,
            "all_parameters_finite": True,
        },
        "claim_boundary": (
            "The public ONNX repository declares facebook/MobileLLM-600M as its base. "
            "The reconstruction is structurally exact to that full-precision ONNX graph, "
            "but the gated official PyTorch weight file was unavailable for a direct hash comparison."
        ),
    }
    (args.snapshot / "reconstruction-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
