#!/usr/bin/env python3
"""Evaluate the official OpenAI CLIP ViT-B/32 visual tower."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_frozen_features import atomic_json, load_dataset, md5, ridge_probe, sha256, weighted_knn
from evaluate_reference_backbone import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--wheel-root", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    return parser.parse_args()


def verify_inputs(arguments: argparse.Namespace, reference: dict, base: dict) -> dict:
    if reference.get("status") != "PREREGISTERED_OPENCLIP_REFERENCE":
        raise RuntimeError("the frozen OpenCLIP-reference protocol is required")
    if sha256(arguments.base_protocol) != reference["base_protocol_sha256"]:
        raise RuntimeError("base protocol identity mismatch")
    specification = reference["model"]
    weight_sha256 = sha256(arguments.weights)
    if weight_sha256 != arguments.expected_weights_sha256 or weight_sha256 != specification["weights_sha256"]:
        raise RuntimeError("weight identity mismatch")
    if arguments.weights.stat().st_size != specification["weights_bytes"]:
        raise RuntimeError("weight length mismatch")
    wheel_files = {}
    for filename, expected in reference["python_wheels"].items():
        path = arguments.wheel_root / filename
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"wheel identity mismatch for {filename}")
        wheel_files[filename] = {"bytes": path.stat().st_size, "sha256": actual}
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
        "python_wheels": wheel_files,
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

    import ftfy
    import open_clip
    import timm
    import torchvision

    versions = {
        "ftfy": ftfy.__version__,
        "open_clip": open_clip.__version__,
        "timm": timm.__version__,
        "torchvision": torchvision.__version__,
    }
    if versions != reference["python_versions"]:
        raise RuntimeError(f"Python package version mismatch: {versions}")
    model = open_clip.create_model("ViT-B-32", pretrained=None, force_quick_gelu=True)
    scripted = torch.jit.load(str(arguments.weights), map_location="cpu").eval()
    state_dict = dict(scripted.state_dict())
    del scripted
    for key in ("input_resolution", "context_length", "vocab_size"):
        state_dict.pop(key)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    visual_parameter_count = sum(parameter.numel() for parameter in model.visual.parameters())
    if parameter_count != specification["parameters"]:
        raise RuntimeError(f"full parameter count mismatch: {parameter_count}")
    if visual_parameter_count != specification["visual_parameters"]:
        raise RuntimeError(f"visual parameter count mismatch: {visual_parameter_count}")
    visual = model.visual.eval().to(device)
    visual.requires_grad_(False)
    del model

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
        "python_versions": versions,
        "reference_protocol_sha256": sha256(arguments.reference_protocol),
        "results": {},
        "status": "RUNNING",
        "visual_parameters": visual_parameter_count,
    }
    atomic_json(arguments.output_dir / "results.json", report)
    evaluation = base["evaluation"]
    for dataset, values in datasets.items():
        train_images, train_labels, test_images, test_labels = values
        train_features, train_targets, train_seconds = extract_features(
            visual,
            train_images,
            train_labels,
            batch_size=arguments.batch_size,
            device=device,
            expected_dimension=specification["feature_dimension"],
        )
        test_features, test_targets, test_seconds = extract_features(
            visual,
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
