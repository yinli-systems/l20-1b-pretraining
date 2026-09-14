#!/usr/bin/env python3
"""Train a fixed-49-token query reader with optional address supervision."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from counterfactual_losses import masked_sequence_logprob
from modeling import bridge_from_architecture, freeze, load_bridge_parent, vision_features
from train_stage_a_full_token import cosine_lr, release_records, sha256_file, utc_now, vision_records, write_json_atomic
from train_stage_c_counterfactual import FamilyBatchBuilder, FamilyDataset, save_checkpoint


VARIANTS = ("base", "edited", "invariant")


def grid_index(coordinate: int, grid: int = 7) -> int:
    return min(grid - 1, max(0, int(coordinate * grid / 256)))


@dataclass
class RoutingBatchBuilder(FamilyBatchBuilder):
    output_grid: int = 7

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(rows)
        address_targets = []
        address_masks = []
        for row in rows:
            target = next(
                item for item in row["scene_state"]["objects"] if item["id"] == "target"
            )
            address_targets.append(
                grid_index(target["y"], self.output_grid) * self.output_grid
                + grid_index(target["x"], self.output_grid)
            )
            address_masks.append(row["question_role"].startswith("affected_"))
        batch["address_targets"] = torch.tensor(address_targets, dtype=torch.long)
        batch["address_supervision_mask"] = torch.tensor(address_masks, dtype=torch.bool)
        return batch


def address_objective(
    attention: torch.Tensor,
    targets: torch.Tensor,
    supervised: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Score address accuracy and base/edit consistency at scene-family level."""
    if attention.ndim != 4 or attention.shape[1:3] != (3, 2):
        raise ValueError("attention must be [families, variants, candidates, visual_tokens]")
    if targets.shape != attention.shape[:1] or supervised.shape != targets.shape:
        raise ValueError("address target/mask shape mismatch")
    if not supervised.any():
        raise ValueError("batch contains no address-supervised families")
    candidate_gap = (attention[:, :, 0] - attention[:, :, 1]).abs().max()
    mean_attention = attention.float().mean(dim=2)
    selected = mean_attention[supervised]
    selected_targets = targets[supervised]
    expanded_targets = selected_targets[:, None].expand(-1, len(VARIANTS))
    address_loss = F.nll_loss(
        selected.clamp_min(1e-8).log().flatten(0, 1),
        expanded_targets.flatten(),
    )
    base = selected[:, 0]
    edited = selected[:, 1]
    midpoint = 0.5 * (base + edited)
    pair_js = 0.5 * (
        F.kl_div(midpoint.clamp_min(1e-8).log(), base, reduction="batchmean")
        + F.kl_div(midpoint.clamp_min(1e-8).log(), edited, reduction="batchmean")
    )
    accuracy = (selected.argmax(dim=-1) == expanded_targets).float().mean()
    target_probability = selected.gather(
        -1, expanded_targets.unsqueeze(-1)
    ).mean()
    entropy = -(
        selected * selected.clamp_min(1e-8).log()
    ).sum(dim=-1).mean()
    return {
        "address_loss": address_loss,
        "address_pair_js": pair_js,
        "address_accuracy": accuracy,
        "address_target_probability": target_probability,
        "address_entropy": entropy,
        "candidate_attention_max_gap": candidate_gap,
    }


def counterfactual_value_features(
    features: torch.Tensor,
    families: int,
) -> torch.Tensor:
    """Swap base/edit value sources while preserving each address-side example."""
    if features.ndim != 3 or features.shape[0] != families * len(VARIANTS) * 2:
        raise ValueError("value features must be [families * variants * candidates, tokens, width]")
    shaped = features.reshape(families, len(VARIANTS), 2, *features.shape[1:])
    swapped = torch.stack((shaped[:, 1], shaped[:, 0], shaped[:, 2]), dim=1)
    return swapped.reshape_as(features)


def value_swap_objective(
    scores: torch.Tensor,
    targets: torch.Tensor,
    supervised: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Require base-address/edit-value and edit-address/base-value answer exchange."""
    if scores.ndim != 3 or scores.shape[1:] != (len(VARIANTS), 2):
        raise ValueError("value-swap scores must be [families, variants, candidates]")
    if targets.shape != scores.shape[:2] or supervised.shape != targets.shape[:1]:
        raise ValueError("value-swap target/mask shape mismatch")
    if not supervised.any():
        raise ValueError("batch contains no value-swap-supervised families")
    swapped_targets = torch.stack(
        (targets[:, 1], targets[:, 0], targets[:, 2]), dim=1
    )
    selected_scores = scores[supervised, :2].flatten(0, 1)
    selected_targets = swapped_targets[supervised, :2].flatten()
    return {
        "value_swap_loss": F.cross_entropy(selected_scores, selected_targets),
        "value_swap_accuracy": (
            selected_scores.argmax(dim=-1) == selected_targets
        ).float().mean(),
    }


def validate_data(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    data = protocol["data"]
    manifest = Path(data["manifest"])
    if sha256_file(manifest) != data["manifest_sha256"]:
        raise SystemExit("router manifest hash mismatch")
    for name, item in data["evidence"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"router data evidence hash mismatch: {name}")
    all_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if {row["split"] for row in all_rows} != {"train", "mechanism_dev", "selection_dev"}:
        raise SystemExit("router partition set mismatch")
    rows = [row for row in all_rows if row["split"] == data["split"]]
    if len(rows) != data["scene_families"]:
        raise SystemExit("router training-family count mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("statistical_cluster_id") != row.get("scene_pair_id"):
            raise SystemExit("router statistical cluster mismatch")
        grouped.setdefault(row["scene_pair_id"], []).append(row)
    if len(grouped) != data["unique_scene_pairs"]:
        raise SystemExit("router scene-pair count mismatch")
    if any(len(pair) != data["questions_per_scene_pair"] for pair in grouped.values()):
        raise SystemExit("router questions-per-scene-pair mismatch")
    rng = random.Random(protocol["optimization"]["seed"])
    pair_ids = sorted(grouped)
    rng.shuffle(pair_ids)
    return [
        row
        for pair_id in pair_ids
        for row in sorted(grouped[pair_id], key=lambda item: item["question_index"])
    ]


def main() -> None:
    process_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "authorized_counterfactual_address_value_routing_development_v1":
        raise SystemExit("counterfactual address-value routing protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    if args.arm not in protocol["arms"]:
        raise SystemExit("router arm is not declared")
    source_code = protocol["source_code"]
    root = Path(__file__).parent
    for filename, key in (
        ("train_counterfactual_address_value_router.py", "trainer_sha256"),
        ("modeling.py", "modeling_sha256"),
        ("train_stage_c_counterfactual.py", "stage_c_trainer_sha256"),
        ("counterfactual_losses.py", "counterfactual_losses_sha256"),
    ):
        if sha256_file(root / filename) != source_code[key]:
            raise SystemExit(f"router source hash mismatch: {filename}")
    prerequisite = protocol["prerequisite"]
    prerequisite_path = Path(prerequisite["tuned_plain_ce_summary"])
    if sha256_file(prerequisite_path) != prerequisite["tuned_plain_ce_summary_sha256"]:
        raise SystemExit("router prerequisite hash mismatch")
    prerequisite_summary = json.loads(prerequisite_path.read_text())
    if prerequisite_summary.get("decision") != prerequisite["required_decision"]:
        raise SystemExit("tuned plain CE did not admit router development")
    if prerequisite_summary.get("final_test_used") is not False:
        raise SystemExit("tuned plain-CE prerequisite used final test")
    arm = protocol["arms"][args.arm]
    weights = arm["loss_weights"]
    required_weights = {"ordinary_answer", "address", "address_pair", "value_swap"}
    if set(weights) != required_weights or any(float(value) < 0 for value in weights.values()):
        raise SystemExit("router loss weights must be complete and nonnegative")
    if weights["ordinary_answer"] != 1.0:
        raise SystemExit("router requires ordinary answer supervision")
    if args.arm == "query_ce" and any(
        weights[name] != 0.0 for name in ("address", "address_pair", "value_swap")
    ):
        raise SystemExit("query-CE control must not use causal routing objectives")
    if args.arm == "cavr" and any(
        weights[name] <= 0.0 for name in ("address", "address_pair", "value_swap")
    ):
        raise SystemExit("CAVR requires address, address-pair, and value-swap supervision")
    architecture = arm["architecture"]
    if (
        architecture.get("compressor") != "spatial_query"
        or int(architecture.get("visual_tokens", 0)) != 196
        or int(architecture.get("output_visual_tokens", 0)) != 49
        or int(architecture.get("query_readout_rank", 0)) != 64
    ):
        raise SystemExit("router architecture must be rank-64 on the 196-to-49 path")
    rows = validate_data(protocol)
    optimization = protocol["optimization"]
    batch_size = int(optimization["micro_batch_families"])
    if int(optimization["gradient_accumulation_steps"]) != 1:
        raise SystemExit("router trainer currently requires gradient_accumulation_steps=1")
    if batch_size % int(protocol["data"]["questions_per_scene_pair"]):
        raise SystemExit("router microbatch must preserve complete scene pairs")
    expected_steps = len(rows) // (
        batch_size * int(optimization["gradient_accumulation_steps"])
    )
    if expected_steps != optimization["max_optimizer_steps"]:
        raise SystemExit("router optimizer-step count mismatch")
    max_steps = min(args.max_optimizer_steps or expected_steps, expected_steps)
    max_wall = min(
        args.max_wall_seconds or optimization["max_wall_seconds"],
        optimization["max_wall_seconds"],
    )

    parents = protocol["parents"]
    base = Path(parents["language"])
    vision_path = Path(parents["vision"])
    parent_bridge = Path(parents["bridge"])
    parent_adapter = Path(parents["language_adapter"])
    if sha256_file(parent_bridge) != parents["bridge_sha256"]:
        raise SystemExit("router parent bridge hash mismatch")
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(parent_adapter / name) != expected:
            raise SystemExit(f"router parent adapter hash mismatch: {name}")
    parent_before = {"language": release_records(base), "vision": vision_records(vision_path)}

    torch.manual_seed(optimization["seed"])
    torch.cuda.manual_seed_all(optimization["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(
        vision_path, local_files_only=True, use_fast=False
    )
    loader = DataLoader(
        FamilyDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=RoutingBatchBuilder(
            tokenizer,
            processor,
            optimization["text_tokens_max"],
            teacher=None,
            deduplicate_images=True,
            output_grid=7,
        ),
        drop_last=True,
    )
    language = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language.config.use_cache = False
    from peft import PeftModel

    language = PeftModel.from_pretrained(
        language, parent_adapter, is_trainable=True, local_files_only=True
    )
    language.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    language.train()
    vision = SiglipVisionModel.from_pretrained(
        vision_path, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    bridge = bridge_from_architecture(architecture).to(
        device="cuda", dtype=torch.bfloat16
    )
    missing = load_bridge_parent(
        bridge, load_file(parent_bridge, device="cpu"), allow_new_compressor=True
    )
    if not missing or not all(name.startswith("query_readout.") for name in missing):
        raise RuntimeError(f"unexpected query-readout parent initialization: {missing}")
    bridge.train()
    assert bridge.query_readout is not None
    lora_parameters = [
        parameter
        for name, parameter in language.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    unexpected_language = [
        name
        for name, parameter in language.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if not lora_parameters or unexpected_language:
        raise RuntimeError(f"unexpected trainable language parameters: {unexpected_language}")
    readout_parameters = list(bridge.query_readout.parameters())
    readout_ids = {id(parameter) for parameter in readout_parameters}
    compressor_parameters = list(bridge.compressor.parameters())
    compressor_ids = {id(parameter) for parameter in compressor_parameters}
    bridge_parameters = list(bridge.parameters())
    core_bridge_parameters = [
        parameter
        for parameter in bridge_parameters
        if id(parameter) not in readout_ids and id(parameter) not in compressor_ids
    ]
    optimizer_groups = [
        ("adapter", lora_parameters, optimization["adapter_learning_rate"]),
        ("bridge", core_bridge_parameters, optimization["bridge_learning_rate"]),
        ("compressor", compressor_parameters, optimization["compressor_learning_rate"]),
        ("query_readout", readout_parameters, optimization["query_readout_learning_rate"]),
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters, "lr": learning_rate}
            for _, parameters, learning_rate in optimizer_groups
        ],
        weight_decay=optimization["weight_decay"],
        fused=True,
    )
    warmup = max(1, round(max_steps * optimization["warmup_ratio"]))
    trainable = lora_parameters + bridge_parameters

    args.output.mkdir(parents=True)
    run = {
        "schema_version": "2026-09-14-v1",
        "status": "running",
        "started_at": utc_now(),
        "arm": args.arm,
        "device": torch.cuda.get_device_name(),
        "protocol_sha256": sha256_file(args.protocol),
        "runner_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": protocol["data"]["manifest_sha256"],
        "parent_bridge_sha256": parents["bridge_sha256"],
        "missing_bridge_parent_keys": missing,
        "loss_weights": weights,
        "language_forward_passes_per_batch": 1 + int(weights["value_swap"] > 0),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "query_readout_parameters": sum(parameter.numel() for parameter in readout_parameters),
        "output_visual_tokens": bridge.spec.output_tokens,
        "max_optimizer_steps": max_steps,
        "optimizer_step": 0,
        "parent_before": parent_before,
    }
    write_json_atomic(args.output / "run.json", run)
    log_path = args.output / "train.jsonl"
    started = time.monotonic()
    optimizer_step = 0
    stop_reason = "data_exhausted"
    try:
        for batch in loader:
            if time.monotonic() - started >= max_wall:
                stop_reason = "wall_time_cap"
                break
            pixels = batch["pixel_values"].to(
                "cuda", dtype=torch.bfloat16, non_blocking=True
            )
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            answer_targets = batch["target_indices"].to("cuda", non_blocking=True)
            address_targets = batch["address_targets"].to("cuda", non_blocking=True)
            address_mask = batch["address_supervision_mask"].to("cuda", non_blocking=True)
            with torch.no_grad():
                unique_features = vision_features(
                    vision,
                    pixels,
                    int(architecture.get("vision_feature_layer", -1)),
                )
                text = language.get_input_embeddings()(input_ids)
            feature_indices = batch["image_feature_indices"].to(
                "cuda", non_blocking=True
            )
            features = unique_features[feature_indices.reshape(-1)].repeat_interleave(
                2, dim=0
            )
            inputs, mask, targets, readout_attention = bridge.inject_with_readout(
                text.detach(), attention_mask, labels, features
            )
            assert readout_attention is not None
            logits = language(inputs_embeds=inputs, attention_mask=mask).logits[:, :-1]
            shifted = targets[:, 1:]
            token_mask = shifted != -100
            scores = masked_sequence_logprob(
                logits, shifted.masked_fill(~token_mask, 0), token_mask
            ).reshape(answer_targets.shape[0], len(VARIANTS), 2)
            ordinary_loss = F.cross_entropy(
                scores.flatten(0, 1), answer_targets.flatten()
            )
            address = address_objective(
                readout_attention.reshape(
                    answer_targets.shape[0], len(VARIANTS), 2, -1
                ),
                address_targets,
                address_mask,
            )
            value_swap = {
                "value_swap_loss": ordinary_loss.new_zeros(()),
                "value_swap_accuracy": ordinary_loss.new_zeros(()),
            }
            if weights["value_swap"] > 0:
                swapped_value_features = counterfactual_value_features(
                    features, answer_targets.shape[0]
                )
                swapped_inputs, swapped_mask, swapped_targets, _ = (
                    bridge.inject_with_readout(
                        text.detach(),
                        attention_mask,
                        labels,
                        features,
                        query_readout_value_features=swapped_value_features,
                    )
                )
                assert swapped_targets is not None
                swapped_logits = language(
                    inputs_embeds=swapped_inputs, attention_mask=swapped_mask
                ).logits[:, :-1]
                shifted_swapped = swapped_targets[:, 1:]
                swapped_token_mask = shifted_swapped != -100
                swapped_scores = masked_sequence_logprob(
                    swapped_logits,
                    shifted_swapped.masked_fill(~swapped_token_mask, 0),
                    swapped_token_mask,
                ).reshape(answer_targets.shape[0], len(VARIANTS), 2)
                value_swap = value_swap_objective(
                    swapped_scores, answer_targets, address_mask
                )
            loss = (
                weights["ordinary_answer"] * ordinary_loss
                + weights["address"] * address["address_loss"]
                + weights["address_pair"] * address["address_pair_js"]
                + weights["value_swap"] * value_swap["value_swap_loss"]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite router loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nonfinite_gradient_names = [
                name
                for name, parameter in list(language.named_parameters())
                + list(bridge.named_parameters())
                if parameter.requires_grad
                and parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite_gradient_names:
                raise RuntimeError(
                    "non-finite router gradients: "
                    + ",".join(nonfinite_gradient_names[:16])
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, optimization["gradient_clip_norm"]
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite router gradient")
            optimizer_step += 1
            scale = cosine_lr(
                optimizer_step,
                max_steps,
                warmup,
                optimization["minimum_learning_rate_ratio"],
            )
            learning_rates = {
                name: base_lr * scale for name, _, base_lr in optimizer_groups
            }
            for group, (name, _, _) in zip(optimizer.param_groups, optimizer_groups):
                group["lr"] = learning_rates[name]
            optimizer.step()
            elapsed = time.monotonic() - started
            event = {
                "optimizer_step": optimizer_step,
                "total_loss": float(loss.detach()),
                "ordinary_loss": float(ordinary_loss.detach()),
                **{name: float(value.detach()) for name, value in address.items()},
                **{name: float(value.detach()) for name, value in value_swap.items()},
                "candidate_accuracy": float(
                    (scores.argmax(dim=-1) == answer_targets).float().mean().detach()
                ),
                "gradient_norm": float(grad_norm),
                "learning_rates": learning_rates,
                "elapsed_seconds": elapsed,
                "families_per_second": optimizer_step * batch_size / elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "timestamp": utc_now(),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            run.update(event)
            write_json_atomic(args.output / "run.json", run)
            if optimizer_step % optimization["checkpoint_every_steps"] == 0:
                save_checkpoint(args.output, bridge, language, optimizer, run)
            if optimizer_step >= max_steps:
                stop_reason = "optimizer_step_cap"
                break
        run["status"] = "complete"
        run["stop_reason"] = stop_reason
        run["completed_at"] = utc_now()
        run["wall_seconds"] = time.monotonic() - started
        run["total_wall_seconds"] = time.monotonic() - process_started
        if optimizer_step and optimizer_step % optimization["checkpoint_every_steps"]:
            save_checkpoint(args.output, bridge, language, optimizer, run)
        parent_after = {"language": release_records(base), "vision": vision_records(vision_path)}
        if parent_after != parent_before or sha256_file(parent_bridge) != parents["bridge_sha256"]:
            raise RuntimeError("immutable router parent changed")
        for name, expected in parents["language_adapter_files"].items():
            if sha256_file(parent_adapter / name) != expected:
                raise RuntimeError(f"immutable router adapter changed: {name}")
        run["parent_after"] = parent_after
        write_json_atomic(args.output / "run.json", run)
    except Exception as error:
        run["status"] = "failed"
        run["failed_at"] = utc_now()
        run["error"] = f"{type(error).__name__}: {error}"
        run["total_wall_seconds"] = time.monotonic() - process_started
        write_json_atomic(args.output / "run.json", run)
        raise


if __name__ == "__main__":
    main()
