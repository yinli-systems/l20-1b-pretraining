#!/usr/bin/env python3
"""Frozen-feature evaluation for the bounded ViT-g/14 real-image pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-per-rank", type=int, default=128)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune-no-cudagraphs"),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("random", "teacher", "student"),
        default=("random", "teacher", "student"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_idx(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        zero_a, zero_b, dtype_code, dimensions = struct.unpack(">BBBB", handle.read(4))
        if (zero_a, zero_b, dtype_code) != (0, 0, 8):
            raise RuntimeError(f"unsupported IDX header in {path}")
        shape = tuple(struct.unpack(">I", handle.read(4))[0] for _ in range(dimensions))
        payload = handle.read()
    expected = 1
    for value in shape:
        expected *= value
    if len(payload) != expected:
        raise RuntimeError(f"IDX payload length mismatch in {path}: {len(payload)} != {expected}")
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(shape)


def load_dataset(root: Path, expected_train: int, expected_test: int) -> tuple[torch.Tensor, ...]:
    train_images = read_idx(root / "train-images-idx3-ubyte.gz")
    train_labels = read_idx(root / "train-labels-idx1-ubyte.gz").long()
    test_images = read_idx(root / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx(root / "t10k-labels-idx1-ubyte.gz").long()
    if train_images.shape != (expected_train, 28, 28) or train_labels.shape != (expected_train,):
        raise RuntimeError(f"unexpected training shapes: {train_images.shape}, {train_labels.shape}")
    if test_images.shape != (expected_test, 28, 28) or test_labels.shape != (expected_test,):
        raise RuntimeError(f"unexpected test shapes: {test_images.shape}, {test_labels.shape}")
    if train_labels.min() != 0 or train_labels.max() != 9:
        raise RuntimeError("training labels must cover classes 0 through 9")
    if test_labels.min() != 0 or test_labels.max() != 9:
        raise RuntimeError("test labels must cover classes 0 through 9")
    return train_images, train_labels, test_images, test_labels


def preprocess(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = images.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
    value = value.repeat(1, 3, 1, 1).div_(255.0)
    value = F.interpolate(value, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return value.sub_(mean).div_(std)


def partition_bounds(size: int, rank: int, world_size: int) -> tuple[int, int]:
    if size % world_size:
        raise RuntimeError(f"dataset size {size} must be divisible by world size {world_size}")
    per_rank = size // world_size
    return rank * per_rank, (rank + 1) * per_rank


@torch.no_grad()
def extract_features(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
    start, end = partition_bounds(len(images), rank, world_size)
    local_images = images[start:end]
    local_labels = labels[start:end]
    feature_chunks: list[torch.Tensor] = []
    began = time.monotonic()
    for offset in range(0, len(local_images), batch):
        raw = local_images[offset : offset + batch]
        actual = len(raw)
        if actual < batch:
            raw = torch.cat((raw, raw[-1:].expand(batch - actual, -1, -1)), dim=0)
        pixels = preprocess(raw, device)
        _logits, features = model(pixels, return_features=True)
        feature_chunks.append(features[:actual].detach().to(dtype=torch.float16))
    local_features = torch.cat(feature_chunks, dim=0)
    if local_features.shape != (len(local_images), 1536):
        raise RuntimeError(f"unexpected local feature shape: {local_features.shape}")
    gathered_features = torch.empty(
        (len(images), local_features.shape[1]), device=device, dtype=local_features.dtype
    )
    gathered_labels = torch.empty((len(labels),), device=device, dtype=torch.long)
    dist.all_gather_into_tensor(gathered_features, local_features.contiguous())
    dist.all_gather_into_tensor(gathered_labels, local_labels.to(device).contiguous())
    elapsed_tensor = torch.tensor(time.monotonic() - began, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed = float(elapsed_tensor.item())
    if rank == 0:
        return gathered_features.cpu(), gathered_labels.cpu(), elapsed
    return None, None, elapsed


def topk_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    predictions = scores.topk(min(5, scores.shape[1]), dim=1).indices
    top1 = predictions[:, 0].eq(labels).float().mean().item()
    top5 = predictions.eq(labels[:, None]).any(dim=1).float().mean().item()
    return top1, top5


@torch.inference_mode()
def weighted_knn(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    k: int,
    temperature: float,
    device: torch.device,
) -> dict:
    train = F.normalize(train_features.to(device=device, dtype=torch.float32), dim=1)
    labels = train_labels.to(device)
    correct1 = 0
    correct5 = 0
    total = 0
    for start in range(0, len(test_features), 256):
        query = F.normalize(test_features[start : start + 256].to(device=device, dtype=torch.float32), dim=1)
        target = test_labels[start : start + 256].to(device)
        similarities, indices = (query @ train.T).topk(k, dim=1)
        neighbor_labels = labels[indices]
        weights = torch.exp(similarities / temperature)
        votes = torch.zeros((len(query), 10), device=device)
        votes.scatter_add_(1, neighbor_labels, weights)
        predictions = votes.topk(5, dim=1).indices
        correct1 += int(predictions[:, 0].eq(target).sum().item())
        correct5 += int(predictions.eq(target[:, None]).any(dim=1).sum().item())
        total += len(query)
    return {"top1": correct1 / total, "top5": correct5 / total, "k": k, "temperature": temperature}


def stratified_split(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    fit, validation = [], []
    for class_id in range(10):
        indices = torch.where(labels == class_id)[0]
        generator = torch.Generator().manual_seed(seed + class_id)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        validation_count = len(indices) // 5
        validation.append(indices[:validation_count])
        fit.append(indices[validation_count:])
    return torch.cat(fit).sort().values, torch.cat(validation).sort().values


def fit_ridge(
    features: torch.Tensor,
    labels: torch.Tensor,
    regularization: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = F.normalize(features.to(device=device, dtype=torch.float32), dim=1)
    y = F.one_hot(labels.to(device), num_classes=10).float()
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x / len(x)
    rhs = centered_x.T @ centered_y / len(x)
    gram.diagonal().add_(regularization)
    weights = torch.linalg.solve(gram, rhs)
    return weights, x_mean, y_mean


@torch.inference_mode()
def ridge_scores(
    features: torch.Tensor,
    parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    weights, x_mean, y_mean = parameters
    x = F.normalize(features.to(device=device, dtype=torch.float32), dim=1)
    return (x - x_mean) @ weights + y_mean


def ridge_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    lambdas: list[float],
    seed: int,
    device: torch.device,
) -> dict:
    fit_indices, validation_indices = stratified_split(train_labels, seed)
    candidates = []
    for value in lambdas:
        parameters = fit_ridge(train_features[fit_indices], train_labels[fit_indices], value, device)
        scores = ridge_scores(train_features[validation_indices], parameters, device)
        top1, top5 = topk_accuracy(scores.cpu(), train_labels[validation_indices])
        candidates.append({"lambda": value, "validation_top1": top1, "validation_top5": top5})
    selected = max(candidates, key=lambda row: (row["validation_top1"], -row["lambda"]))
    parameters = fit_ridge(train_features, train_labels, selected["lambda"], device)
    scores = ridge_scores(test_features, parameters, device)
    top1, top5 = topk_accuracy(scores.cpu(), test_labels)
    return {
        "top1": top1,
        "top5": top5,
        "selected_lambda": selected["lambda"],
        "validation_candidates": candidates,
    }


def verify_inputs(arguments: argparse.Namespace, protocol: dict) -> dict:
    checkpoint_metadata_path = arguments.checkpoint / "metadata.json"
    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text())
    expected_metadata = protocol["checkpoint"]["metadata_identity"]
    for key, expected in expected_metadata.items():
        if checkpoint_metadata.get(key) != expected:
            raise RuntimeError(f"checkpoint identity mismatch for {key}")
    if checkpoint_metadata.get("completed_step") != protocol["checkpoint"]["completed_step"]:
        raise RuntimeError("checkpoint step mismatch")
    checkpoint_specification = protocol["checkpoint"]
    receipt_sha256 = sha256(arguments.checkpoint_sha256_receipt)
    if receipt_sha256 != checkpoint_specification["full_sha256_receipt_sha256"]:
        raise RuntimeError("checkpoint SHA-256 receipt identity mismatch")
    receipt_entries = {}
    for line in arguments.checkpoint_sha256_receipt.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        receipt_entries[Path(path.strip()).name] = digest
    if receipt_entries != checkpoint_specification["artifact_sha256"]:
        raise RuntimeError("checkpoint SHA-256 receipt contents mismatch")
    for filename, expected_size in checkpoint_specification["artifact_sizes_bytes"].items():
        path = arguments.checkpoint / filename
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"checkpoint artifact size mismatch for {path}")
    for filename in (".metadata", "metadata.json"):
        if sha256(arguments.checkpoint / filename) != checkpoint_specification["artifact_sha256"][filename]:
            raise RuntimeError(f"checkpoint metadata hash mismatch for {filename}")
    dataset_files = {}
    for dataset, specification in protocol["datasets"].items():
        dataset_files[dataset] = {}
        for filename, expected_md5 in specification["files"].items():
            path = arguments.data_root / dataset.replace("_", "-") / filename
            actual_md5 = md5(path)
            if actual_md5 != expected_md5:
                raise RuntimeError(f"dataset identity mismatch for {path}: {actual_md5}")
            dataset_files[dataset][filename] = {
                "bytes": path.stat().st_size,
                "md5": actual_md5,
                "sha256": sha256(path),
            }
    return {
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_metadata_json_sha256": sha256(checkpoint_metadata_path),
        "checkpoint_sha256_receipt_sha256": receipt_sha256,
        "dcp_metadata_sha256": sha256(arguments.checkpoint / ".metadata"),
        "dataset_files": dataset_files,
    }


def load_checkpoint_variant(model, checkpoint: Path, variant: str) -> None:
    state = {variant: get_model_state_dict(model)}
    dcp.load(state, checkpoint_id=str(checkpoint))
    set_model_state_dict(model, state[variant])


def main() -> None:
    arguments = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, 2, 4):
        raise RuntimeError(f"this evaluation supports one, two, or four FSDP2 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(20260917)
    torch.cuda.manual_seed_all(20260917)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "PREREGISTERED_FROZEN_FEATURE_DIAGNOSTIC":
        raise RuntimeError("a preregistered frozen-feature protocol is required")
    if protocol["evaluation"]["seed"] != 20260917:
        raise RuntimeError("unexpected evaluation seed")
    if rank == 0:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    identities = verify_inputs(arguments, protocol)

    from train_dino_real import DINOModel, parse_args as training_parse_args, shard_dino_model

    original_argv = list(os.sys.argv)
    try:
        os.sys.argv = [original_argv[0], "--manifest", "unused", "--protocol", "unused", "--acquisition-receipt", "unused", "--output-dir", "unused"]
        model_arguments = training_parse_args()
    finally:
        os.sys.argv = original_argv
    model = DINOModel(model_arguments, activation_checkpointing=False).to(device)
    model.requires_grad_(False)
    shard_dino_model(model, reshard_after_forward=True, compute_dtype=torch.float16)
    model.eval()
    model_run = model
    if arguments.compile_mode != "none":
        model_run = torch.compile(model, mode=arguments.compile_mode)

    datasets = {}
    for dataset, specification in protocol["datasets"].items():
        values = load_dataset(
            arguments.data_root / dataset.replace("_", "-"),
            specification["train_examples"],
            specification["test_examples"],
        )
        train_images, train_labels, test_images, test_labels = values
        if arguments.max_train:
            train_images, train_labels = train_images[: arguments.max_train], train_labels[: arguments.max_train]
        if arguments.max_test:
            test_images, test_labels = test_images[: arguments.max_test], test_labels[: arguments.max_test]
        datasets[dataset] = (train_images, train_labels, test_images, test_labels)

    report = {
        "claim_boundary": protocol["claim_boundary"],
        "checkpoint": str(arguments.checkpoint),
        "compile_mode": arguments.compile_mode,
        "data_root": str(arguments.data_root),
        "identities": identities,
        "protocol_sha256": sha256(arguments.protocol),
        "results": {},
        "source_sha256": sha256(Path(__file__)),
        "status": "RUNNING",
        "world_size": world_size,
    }
    if rank == 0:
        atomic_json(arguments.output_dir / "results.json", report)

    for variant in arguments.variants:
        if variant in ("teacher", "student"):
            load_checkpoint_variant(model, arguments.checkpoint, variant)
            dist.barrier()
        report["results"][variant] = {}
        for dataset, values in datasets.items():
            train_images, train_labels, test_images, test_labels = values
            train_features, gathered_train_labels, train_seconds = extract_features(
                model_run,
                train_images,
                train_labels,
                batch=arguments.batch_per_rank,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            test_features, gathered_test_labels, test_seconds = extract_features(
                model_run,
                test_images,
                test_labels,
                batch=arguments.batch_per_rank,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if rank == 0:
                assert train_features is not None and test_features is not None
                assert gathered_train_labels is not None and gathered_test_labels is not None
                evaluation = protocol["evaluation"]
                result = {
                    "feature_extraction_seconds_max_rank": {
                        "train": train_seconds,
                        "test": test_seconds,
                    },
                    "knn": weighted_knn(
                        train_features,
                        gathered_train_labels,
                        test_features,
                        gathered_test_labels,
                        k=evaluation["knn"]["k"],
                        temperature=evaluation["knn"]["temperature"],
                        device=device,
                    ),
                    "ridge_probe": ridge_probe(
                        train_features,
                        gathered_train_labels,
                        test_features,
                        gathered_test_labels,
                        lambdas=evaluation["ridge_probe"]["lambda_grid"],
                        seed=evaluation["seed"],
                        device=device,
                    ),
                    "test_examples": len(test_features),
                    "train_examples": len(train_features),
                }
                report["results"][variant][dataset] = result
                atomic_json(arguments.output_dir / "results.json", report)
                del train_features, test_features, gathered_train_labels, gathered_test_labels
                torch.cuda.empty_cache()
            dist.barrier()

    if rank == 0:
        gate = protocol["promotion_gate"]
        random_scores, teacher_scores = [], []
        if "random" in report["results"] and "teacher" in report["results"]:
            for dataset in protocol["datasets"]:
                random_scores.extend(
                    [
                        report["results"]["random"][dataset]["knn"]["top1"],
                        report["results"]["random"][dataset]["ridge_probe"]["top1"],
                    ]
                )
                teacher_scores.extend(
                    [
                        report["results"]["teacher"][dataset]["knn"]["top1"],
                        report["results"]["teacher"][dataset]["ridge_probe"]["top1"],
                    ]
                )
            gain = 100.0 * (sum(teacher_scores) - sum(random_scores)) / len(random_scores)
            report["promotion"] = {
                "minimum_gain_percentage_points": gate[
                    "minimum_teacher_mean_top1_gain_over_random_percentage_points"
                ],
                "teacher_mean_top1_gain_over_random_percentage_points": gain,
                "status": "PASS" if gain >= gate["minimum_teacher_mean_top1_gain_over_random_percentage_points"] else "FAIL",
            }
        report["status"] = "COMPLETE"
        atomic_json(arguments.output_dir / "results.json", report)
        files = [arguments.output_dir / "results.json"]
        (arguments.output_dir / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in files)
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
