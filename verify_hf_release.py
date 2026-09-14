#!/usr/bin/env python3
"""Verify a downloaded Hugging Face safetensors release without loading weights.

The verifier is intentionally standard-library-only.  It checks the release
manifest, every managed file digest, safetensors byte layout, tensor/index
coverage, dtype/shape sizes, and the aggregate tensor invariants recorded at
export time.  It does not claim that a local directory proves a remote Hub
revision; callers must record that revision separately when downloading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any


DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

REQUIRED_MANAGED_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


class VerificationError(ValueError):
    """Raised for a malformed or non-matching release artifact."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON file {path.name}: {exc}") from exc


def _managed_path(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or len(pure.parts) != 1 or name in {"", ".", ".."}:
        raise VerificationError(f"unsafe managed filename: {name!r}")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"managed file is missing or not regular: {name}")
    return path


def _shape_size(shape: Any, tensor_name: str) -> int:
    if not isinstance(shape, list) or any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
        for dimension in shape
    ):
        raise VerificationError(f"invalid shape for tensor {tensor_name}")
    return math.prod(shape)


def inspect_safetensors(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    if file_size < 10:
        raise VerificationError(f"safetensors file is too short: {path.name}")

    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise VerificationError(f"missing safetensors header length: {path.name}")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length < 2 or header_length > file_size - 8:
            raise VerificationError(f"invalid safetensors header length: {path.name}")
        header_bytes = handle.read(header_length)

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid safetensors header: {path.name}: {exc}") from exc
    if not isinstance(header, dict):
        raise VerificationError(f"safetensors header is not an object: {path.name}")

    data_bytes = file_size - 8 - header_length
    intervals: list[tuple[int, int, str]] = []
    tensors: dict[str, dict[str, Any]] = {}
    parameter_count = 0
    tensor_bytes = 0
    dtypes: set[str] = set()
    for name, metadata in header.items():
        if name == "__metadata__":
            if not isinstance(metadata, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata.items()
            ):
                raise VerificationError(f"invalid __metadata__ in {path.name}")
            continue
        if not isinstance(name, str) or not name or not isinstance(metadata, dict):
            raise VerificationError(f"invalid tensor entry in {path.name}")
        dtype = metadata.get("dtype")
        if dtype not in DTYPE_BYTES:
            raise VerificationError(f"unsupported dtype {dtype!r} for tensor {name}")
        count = _shape_size(metadata.get("shape"), name)
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise VerificationError(f"invalid data offsets for tensor {name}")
        start, end = offsets
        expected_bytes = count * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > data_bytes or end - start != expected_bytes:
            raise VerificationError(f"byte extent mismatch for tensor {name}")
        intervals.append((start, end, name))
        tensors[name] = {"dtype": dtype, "shape": metadata["shape"], "bytes": expected_bytes}
        parameter_count += count
        tensor_bytes += expected_bytes
        dtypes.add(dtype)

    # Safetensors requires one fully indexed, hole-free data buffer.  This also
    # detects overlapping tensor extents without reading multi-gigabyte payloads.
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise VerificationError(
                f"non-contiguous or overlapping data before tensor {name} in {path.name}"
            )
        cursor = end
    if cursor != data_bytes:
        raise VerificationError(f"unindexed trailing data in {path.name}")

    return {
        "file": path.name,
        "tensor_count": len(tensors),
        "parameter_count": parameter_count,
        "tensor_bytes": tensor_bytes,
        "stored_dtypes": sorted(dtypes),
        "tensors": tensors,
    }


def verify_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise VerificationError(f"release directory does not exist: {root}")

    manifest_path = _managed_path(root, "release-manifest.json")
    manifest = _json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise VerificationError("release manifest has no files object")
    managed = manifest["files"]
    missing_required = sorted(REQUIRED_MANAGED_FILES - set(managed))
    if missing_required:
        raise VerificationError(f"manifest omits required files: {missing_required}")

    verified_files: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(managed.items()):
        if not isinstance(expected, dict):
            raise VerificationError(f"invalid manifest entry for {name}")
        path = _managed_path(root, name)
        actual_bytes = path.stat().st_size
        actual_hash = sha256(path)
        if expected.get("bytes") != actual_bytes:
            raise VerificationError(f"size mismatch for {name}")
        if expected.get("sha256") != actual_hash:
            raise VerificationError(f"SHA-256 mismatch for {name}")
        verified_files[name] = {"bytes": actual_bytes, "sha256": actual_hash}

    index = _json(_managed_path(root, "model.safetensors.index.json"))
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise VerificationError("safetensors index has no weight_map object")
    weight_map = index["weight_map"]
    if not weight_map or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise VerificationError("safetensors weight_map is empty or malformed")

    shard_names = sorted(set(weight_map.values()))
    if any(not name.endswith(".safetensors") for name in shard_names):
        raise VerificationError("weight_map references a non-safetensors file")
    unmanaged_shards = sorted(set(shard_names) - set(managed))
    if unmanaged_shards:
        raise VerificationError(f"weight_map references unmanaged shards: {unmanaged_shards}")

    all_tensors: dict[str, str] = {}
    shard_summaries = []
    parameter_count = 0
    tensor_bytes = 0
    stored_dtypes: set[str] = set()
    for shard_name in shard_names:
        summary = inspect_safetensors(_managed_path(root, shard_name))
        shard_summaries.append({key: value for key, value in summary.items() if key != "tensors"})
        for tensor_name in summary["tensors"]:
            if tensor_name in all_tensors:
                raise VerificationError(f"duplicate tensor across shards: {tensor_name}")
            all_tensors[tensor_name] = shard_name
        parameter_count += summary["parameter_count"]
        tensor_bytes += summary["tensor_bytes"]
        stored_dtypes.update(summary["stored_dtypes"])

    if weight_map != all_tensors:
        missing = sorted(set(weight_map) - set(all_tensors))[:5]
        extra = sorted(set(all_tensors) - set(weight_map))[:5]
        misplaced = sorted(
            name for name in set(weight_map) & set(all_tensors) if weight_map[name] != all_tensors[name]
        )[:5]
        raise VerificationError(
            f"index/tensor mismatch; missing={missing} extra={extra} misplaced={misplaced}"
        )

    expected_dtype = sorted(
        value.removeprefix("torch.")
        .replace("float32", "F32")
        .replace("float16", "F16")
        .replace("bfloat16", "BF16")
        for value in manifest.get("stored_dtype", [])
    )
    aggregate_checks = {
        "parameter_count": (manifest.get("parameter_count"), parameter_count),
        "tensor_count": (manifest.get("tensor_count"), len(all_tensors)),
        "tensor_bytes": (manifest.get("tensor_bytes"), tensor_bytes),
        "stored_dtype": (expected_dtype, sorted(stored_dtypes)),
        "index_total_size": (index.get("metadata", {}).get("total_size"), tensor_bytes),
    }
    for field, (expected, actual) in aggregate_checks.items():
        if expected != actual:
            raise VerificationError(f"aggregate {field} mismatch: expected={expected} actual={actual}")

    return {
        "valid": True,
        "release_dir": str(root),
        "managed_file_count": len(verified_files),
        "managed_bytes": sum(item["bytes"] for item in verified_files.values()),
        "shard_count": len(shard_names),
        "tensor_count": len(all_tensors),
        "parameter_count": parameter_count,
        "tensor_bytes": tensor_bytes,
        "stored_dtypes": sorted(stored_dtypes),
        "shards": shard_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    try:
        result = verify_release(args.release_dir)
    except VerificationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
