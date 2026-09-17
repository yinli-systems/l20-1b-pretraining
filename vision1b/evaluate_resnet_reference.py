#!/usr/bin/env python3
"""Evaluate the official supervised torchvision ResNet-50 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from evaluate_frozen_features import (
    atomic_json,
    load_dataset,
    md5,
    ridge_probe,
    sha256,
    weighted_knn,
)
from evaluate_reference_backbone import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--torchvision-wheel", type=Path, required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    return parser.parse_args()


def verify_inputs(arguments: argparse.Namespace, reference: dict, base: dict) -> dict:
    if reference.get("status") != "PREREGISTERED_SUPERVISED_REFERENCE":
        raise RuntimeError("the frozen supervised-reference protocol is required")
    if sha256(arguments.base_protocol) != reference["base_protocol_sha256"]:
        raise RuntimeError("base protocol identity mismatch")
    specification = reference["model"]
    weight_sha256 = sha256(arguments.weights)
    if weight_sha256 != arguments.expected_weights_sha256 or weight_sha256 != specification["weights_sha256"]:
        raise RuntimeError("weight identity mismatch")
    if arguments.weights.stat().st_size != specification["weights_bytes"]:
        raise RuntimeError("weight length mismatch")
    wheel_sha256 = sha256(arguments.torchvision_wheel)
    if wheel_sha256 != arguments.expected_wheel_sha256 or wheel_sha256 != reference["torchvision"]["wheel_sha256"]:
        raise RuntimeError("torchvision wheel identity mismatch")
    dataset_files = {}
    for dataset, dataset_specification in base["datasets"].items():
        dataset_files[dataset] = {}
        for filename, expected_md5 in dataset_specification["files"].items():
            path = arguments.data_root / dataset.replace("_", "-") / filename
            actual_md5 = md5(path)
            if actual_md5 != expected_md5:
                raise RuntimeError(f"dataset identity mismatch for {path}")
            dataset_files[dataset][filename] = {
                "bytes": path.stat().st_size,
                "md5": actual_md5,
                "sha256": sha256(path),
            }
    return {
        "dataset_files": dataset_files,
        "torchvision_wheel_sha256": wheel_sha256,
        "weights_bytes": arguments.weights.stat().st_size,
        "weights_sha256": weight_sha256,
    }


def main() -> None:
    arguments = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260917)
    torch.cuda.manual_seed_all(20260917)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    reference = json.loads(arguments.reference_protocol.read_text())
    base = json.loads(arguments.base_protocol.read_text())
    identities = verify_inputs(arguments, reference, base)
    specification = reference["model"]

    import torchvision
    from torchvision.models import resnet50

    if torchvision.__version__ != reference["torchvision"]["version"]:
        raise RuntimeError("torchvision version mismatch")
    if not torchvision.extension._has_ops():
        raise RuntimeError("torchvision compiled operators are unavailable")
    model = resnet50(weights=None)
    state_dict = torch.load(arguments.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != specification["parameters"]:
        raise RuntimeError(f"parameter count mismatch: {parameter_count}")
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)
    model.requires_grad_(False)

    datasets = {}
    for dataset, dataset_specification in base["datasets"].items():
        values = load_dataset(
            arguments.data_root / dataset.replace("_", "-"),
            dataset_specification["train_examples"],
            dataset_specification["test_examples"],
        )
        train_images, train_labels, test_images, test_labels = values
        if arguments.max_train:
            train_images, train_labels = train_images[: arguments.max_train], train_labels[: arguments.max_train]
        if arguments.max_test:
            test_images, test_labels = test_images[: arguments.max_test], test_labels[: arguments.max_test]
        datasets[dataset] = (train_images, train_labels, test_images, test_labels)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "base_protocol_sha256": sha256(arguments.base_protocol),
        "batch_size": arguments.batch_size,
        "claim_boundary": reference["claim_boundary"],
        "evaluation_source_sha256": sha256(Path(__file__)),
        "feature_dimension": specification["feature_dimension"],
        "identities": identities,
        "model_id": specification["id"],
        "parameters": parameter_count,
        "reference_protocol_sha256": sha256(arguments.reference_protocol),
        "results": {},
        "status": "RUNNING",
        "torchvision_version": torchvision.__version__,
    }
    atomic_json(arguments.output_dir / "results.json", report)
    evaluation = base["evaluation"]
    for dataset, values in datasets.items():
        train_images, train_labels, test_images, test_labels = values
        train_features, train_targets, train_seconds = extract_features(
            model,
            train_images,
            train_labels,
            batch_size=arguments.batch_size,
            device=device,
            expected_dimension=specification["feature_dimension"],
        )
        test_features, test_targets, test_seconds = extract_features(
            model,
            test_images,
            test_labels,
            batch_size=arguments.batch_size,
            device=device,
            expected_dimension=specification["feature_dimension"],
        )
        report["results"][dataset] = {
            "feature_extraction_seconds": {"train": train_seconds, "test": test_seconds},
            "knn": weighted_knn(
                train_features,
                train_targets,
                test_features,
                test_targets,
                k=evaluation["knn"]["k"],
                temperature=evaluation["knn"]["temperature"],
                device=device,
            ),
            "ridge_probe": ridge_probe(
                train_features,
                train_targets,
                test_features,
                test_targets,
                lambdas=evaluation["ridge_probe"]["lambda_grid"],
                seed=evaluation["seed"],
                device=device,
            ),
            "test_examples": len(test_features),
            "train_examples": len(train_features),
        }
        atomic_json(arguments.output_dir / "results.json", report)
        del train_features, test_features
        torch.cuda.empty_cache()
    report["status"] = "COMPLETE"
    atomic_json(arguments.output_dir / "results.json", report)
    (arguments.output_dir / "SHA256SUMS").write_text(
        f"{sha256(arguments.output_dir / 'results.json')}  results.json\n"
    )


if __name__ == "__main__":
    main()
