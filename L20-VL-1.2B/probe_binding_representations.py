#!/usr/bin/env python3
"""Localize color-binding information across frozen VLM representation stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, SiglipVisionModel

from counterfactual_losses import paired_cluster_bootstrap
from modeling import bridge_from_architecture, freeze, vision_features
from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


REPRESENTATIONS = (
    "vision_196_local",
    "vision_196_global",
    "compressed_49_local",
    "compressed_49_global",
    "projected_49_local",
    "projected_49_global",
)


def build_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {}
    for row in rows:
        if row.get("question_index") == 0:
            selected[(row["split"], row["scene_pair_id"])] = row
    samples = []
    for (split, pair_id), row in sorted(selected.items()):
        objects = {item["id"]: item for item in row["scene_state"]["objects"]}
        target = objects["target"]
        partner = objects["partner"]
        for variant, color in (("base", target["color"]), ("edited", partner["color"])):
            samples.append({
                "sample_id": f"{pair_id}-{variant}",
                "scene_pair_id": pair_id,
                "split": split,
                "challenge": row["challenge"],
                "variant": variant,
                "image_path": row[f"{variant}_image_path"],
                "image_sha256": row[f"{variant}_image_sha256"],
                "target_x": target["x"],
                "target_y": target["y"],
                "target_color": color,
            })
    return samples


def grid_index(coordinate: int, grid: int) -> int:
    return min(grid - 1, max(0, int(coordinate * grid / 256)))


def extract_representations(
    samples: list[dict[str, Any]],
    vision,
    bridge,
    processor,
    batch_size: int,
    vision_feature_layer: int,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = []
            for sample in batch:
                path = Path(sample["image_path"])
                if sha256_file(path) != sample["image_sha256"]:
                    raise RuntimeError(f"image hash mismatch: {path}")
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
                "cuda", dtype=torch.bfloat16
            )
            features = vision_features(vision, pixels, vision_feature_layer)
            compressed = bridge.compress(features)
            projected = bridge.projector(compressed)
            indices_14 = torch.tensor(
                [
                    grid_index(sample["target_y"], 14) * 14
                    + grid_index(sample["target_x"], 14)
                    for sample in batch
                ],
                device="cuda",
            )
            indices_7 = torch.tensor(
                [
                    grid_index(sample["target_y"], 7) * 7
                    + grid_index(sample["target_x"], 7)
                    for sample in batch
                ],
                device="cuda",
            )
            batch_indices = torch.arange(len(batch), device="cuda")
            values = {
                "vision_196_local": features[batch_indices, indices_14],
                "vision_196_global": features.mean(dim=1),
                "compressed_49_local": compressed[batch_indices, indices_7],
                "compressed_49_global": compressed.mean(dim=1),
                "projected_49_local": projected[batch_indices, indices_7],
                "projected_49_global": projected.mean(dim=1),
            }
            for name, value in values.items():
                outputs[name].append(value.float().cpu())
    return {name: torch.cat(outputs[name]) for name in REPRESENTATIONS}


def fit_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    train = features[train_indices]
    mean = train.mean(dim=0)
    std = train.std(dim=0).clamp_min(1e-5)
    normalized = ((features - mean) / std).cuda(non_blocking=True)
    labels_cuda = labels.cuda(non_blocking=True)
    train_indices_cuda = train_indices.cuda(non_blocking=True)
    torch.manual_seed(seed)
    classifier = torch.nn.Linear(features.shape[1], int(labels.max()) + 1).cuda()
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(steps):
        positions = torch.randint(
            len(train_indices), (batch_size,), generator=generator, device="cuda"
        )
        selected = train_indices_cuda[positions]
        batch_features = normalized[selected]
        batch_labels = labels_cuda[selected]
        loss = F.cross_entropy(classifier(batch_features), batch_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        predictions = classifier(normalized).argmax(dim=-1).cpu()
    state = {
        "weight": classifier.weight.detach().float().cpu(),
        "bias": classifier.bias.detach().float().cpu(),
        "mean": mean.float(),
        "std": std.float(),
    }
    return predictions, state


def accuracy_summary(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    samples: list[dict[str, Any]],
    indices: list[int],
    chance: float,
) -> dict[str, Any]:
    correct = (predictions == labels).float()

    def summarize(selected: list[int]) -> dict[str, Any]:
        differences = [float(correct[index]) - chance for index in selected]
        clusters = [samples[index]["scene_pair_id"] for index in selected]
        interval = paired_cluster_bootstrap(
            differences, clusters, resamples=10_000, seed=20260914
        )
        return {
            "samples": len(selected),
            "scene_pairs": len(set(clusters)),
            "accuracy_percent": 100 * float(correct[selected].mean()),
            "minus_chance": {
                "estimate_pp": 100 * interval.estimate,
                "lower_95_ci_pp": 100 * interval.lower,
                "upper_95_ci_pp": 100 * interval.upper,
                "clusters": interval.clusters,
                "resamples": interval.resamples,
            },
        }

    result = {"overall": summarize(indices), "by_challenge": {}}
    for challenge in sorted({samples[index]["challenge"] for index in indices}):
        selected = [index for index in indices if samples[index]["challenge"] == challenge]
        result["by_challenge"][challenge] = summarize(selected)
    return result


def paired_representation_difference(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    labels: torch.Tensor,
    samples: list[dict[str, Any]],
    indices: list[int],
) -> dict[str, Any]:
    candidate_correct = candidate == labels
    baseline_correct = baseline == labels
    differences = [
        float(candidate_correct[index]) - float(baseline_correct[index]) for index in indices
    ]
    clusters = [samples[index]["scene_pair_id"] for index in indices]
    interval = paired_cluster_bootstrap(
        differences, clusters, resamples=10_000, seed=20260914
    )
    return {
        "estimate_pp": 100 * interval.estimate,
        "lower_95_ci_pp": 100 * interval.lower,
        "upper_95_ci_pp": 100 * interval.upper,
        "clusters": interval.clusters,
        "resamples": interval.resamples,
    }


def diagnose_local_information(
    metrics: dict[str, Any], minimum_accuracy_percent: float,
    minimum_lower_ci_minus_chance_pp: float,
) -> dict[str, Any]:
    stages = ("vision_196_local", "compressed_49_local", "projected_49_local")
    availability = {}
    for name in stages:
        result = metrics[name]["mechanism_dev"]["overall"]
        availability[name] = (
            result["accuracy_percent"] >= minimum_accuracy_percent
            and result["minus_chance"]["lower_95_ci_pp"]
            >= minimum_lower_ci_minus_chance_pp
        )
    if not availability["vision_196_local"]:
        localization = "vision_encoder_or_linear_readout_limit"
    elif not availability["compressed_49_local"]:
        localization = "196_to_49_compression_bottleneck"
    elif not availability["projected_49_local"]:
        localization = "vision_to_language_projection_bottleneck"
    else:
        localization = "downstream_query_binding_or_language_readout_bottleneck"
    return {
        "localization": localization,
        "availability_by_stage": availability,
        "thresholds": {
            "minimum_accuracy_percent": minimum_accuracy_percent,
            "minimum_lower_95_ci_minus_chance_pp": minimum_lower_ci_minus_chance_pp,
        },
        "boundary": (
            "This rule localizes linearly decodable target-color information only; "
            "it cannot establish causal sufficiency or natural-image transfer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "authorized_binding_representation_probe_v1":
        raise SystemExit("representation probe protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    source_code = protocol["source_code"]
    if sha256_file(Path(__file__)) != source_code["probe_sha256"]:
        raise SystemExit("probe source hash mismatch")
    if sha256_file(Path(__file__).with_name("modeling.py")) != source_code["modeling_sha256"]:
        raise SystemExit("probe modeling hash mismatch")
    if sha256_file(Path(__file__).with_name("counterfactual_losses.py")) != source_code["counterfactual_losses_sha256"]:
        raise SystemExit("probe bootstrap source hash mismatch")
    if sha256_file(Path(__file__).with_name("train_stage_a_full_token.py")) != source_code["stage_a_utility_sha256"]:
        raise SystemExit("probe utility source hash mismatch")
    prerequisite = protocol["prerequisite"]
    prerequisite_path = Path(prerequisite["evaluation_summary"])
    if sha256_file(prerequisite_path) != prerequisite["evaluation_summary_sha256"]:
        raise SystemExit("probe prerequisite summary hash mismatch")
    prerequisite_summary = json.loads(prerequisite_path.read_text())
    if prerequisite_summary.get("status") != prerequisite["required_status"]:
        raise SystemExit("probe prerequisite status mismatch")
    if prerequisite_summary.get("final_test_used") is not False:
        raise SystemExit("probe prerequisite used final test")
    if any(
        arm["visual_floor"]["passes_all"]
        for arm in prerequisite_summary["full_mechanism"].values()
    ):
        raise SystemExit("representation probe is not authorized after a passing repair")
    manifest = Path(protocol["data"]["manifest"])
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("probe manifest hash mismatch")
    for evidence in protocol["data"]["evidence"].values():
        if sha256_file(Path(evidence["path"])) != evidence["sha256"]:
            raise SystemExit(f"probe data evidence hash mismatch: {evidence['path']}")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if {row["split"] for row in rows} != {"train", "mechanism_dev", "selection_dev"}:
        raise SystemExit("probe partition set mismatch")
    samples = build_samples(rows)
    colors = sorted({sample["target_color"] for sample in samples})
    color_index = {color: index for index, color in enumerate(colors)}
    labels = torch.tensor([color_index[sample["target_color"]] for sample in samples])
    split_indices = {
        split: [index for index, sample in enumerate(samples) if sample["split"] == split]
        for split in ("train", "mechanism_dev", "selection_dev")
    }
    parents = protocol["parents"]
    if sha256_file(Path(parents["bridge"])) != parents["bridge_sha256"]:
        raise SystemExit("probe parent bridge hash mismatch")
    processor = AutoImageProcessor.from_pretrained(
        parents["vision"], local_files_only=True, use_fast=False
    )
    vision = SiglipVisionModel.from_pretrained(
        parents["vision"], dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    bridge = bridge_from_architecture(protocol["architecture"]).to(
        device="cuda", dtype=torch.bfloat16
    )
    bridge.load_state_dict(load_file(parents["bridge"], device="cpu"), strict=True)
    freeze(bridge)
    features = extract_representations(
        samples,
        vision,
        bridge,
        processor,
        protocol["extraction"]["batch_size"],
        protocol["architecture"]["vision_feature_layer"],
    )
    probe = protocol["probe"]
    train_indices = torch.tensor(split_indices["train"])
    predictions = {}
    states = {}
    metrics = {}
    for name in REPRESENTATIONS:
        predictions[name], states[name] = fit_probe(
            features[name],
            labels,
            train_indices,
            probe["optimizer_steps"],
            probe["batch_size"],
            probe["learning_rate"],
            probe["seed"],
        )
        metrics[name] = {
            split: accuracy_summary(
                predictions[name],
                labels,
                samples,
                split_indices[split],
                1.0 / len(colors),
            )
            for split in ("train", "mechanism_dev", "selection_dev")
        }
    local_global = {}
    for stage in ("vision_196", "compressed_49", "projected_49"):
        local_name = f"{stage}_local"
        global_name = f"{stage}_global"
        local_global[stage] = {
            split: paired_representation_difference(
                predictions[local_name],
                predictions[global_name],
                labels,
                samples,
                split_indices[split],
            )
            for split in ("mechanism_dev", "selection_dev")
        }
    stage_transitions = {}
    for candidate, baseline in (
        ("compressed_49_local", "vision_196_local"),
        ("projected_49_local", "compressed_49_local"),
    ):
        stage_transitions[f"{candidate}_minus_{baseline}"] = {
            split: paired_representation_difference(
                predictions[candidate],
                predictions[baseline],
                labels,
                samples,
                split_indices[split],
            )
            for split in ("mechanism_dev", "selection_dev")
        }
    decision_rules = protocol["decision_rules"]
    diagnosis = diagnose_local_information(
        metrics,
        decision_rules["minimum_accuracy_percent"],
        decision_rules["minimum_lower_95_ci_minus_chance_pp"],
    )
    weights_path = args.output.with_suffix(".safetensors")
    save_file(
        {
            f"{name}.{key}": value.contiguous()
            for name, state in states.items()
            for key, value in state.items()
        },
        weights_path,
    )
    write_json_atomic(args.output, {
        "schema_version": "2026-09-14-v1",
        "status": "complete_diagnostic_only",
        "completed_at": utc_now(),
        "protocol_sha256": sha256_file(args.protocol),
        "probe_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": sha256_file(manifest),
        "parent_bridge_sha256": sha256_file(Path(parents["bridge"])),
        "samples": len(samples),
        "samples_by_split": {split: len(indices) for split, indices in split_indices.items()},
        "color_classes": colors,
        "representations": metrics,
        "local_minus_global": local_global,
        "stage_transitions": stage_transitions,
        "diagnosis": diagnosis,
        "probe_weights": str(weights_path),
        "probe_weights_sha256": sha256_file(weights_path),
        "final_test_used": False,
        "training_prediction_tokens": 0,
        "claim_boundary": (
            "Oracle-local color decoding diagnoses frozen representation availability only. "
            "It does not show that the VLM can bind a language query to an object, and selection_dev is reused development data."
        ),
    })


if __name__ == "__main__":
    main()
