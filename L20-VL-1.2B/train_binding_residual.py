#!/usr/bin/env python3
"""Train the rank-16 binding residual with ordinary or interchange supervision."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from counterfactual_losses import masked_sequence_logprob
from modeling import bridge_from_architecture, freeze, load_bridge_parent, vision_features
from train_stage_a_full_token import cosine_lr, release_records, sha256_file, utc_now, vision_records, write_json_atomic
from train_stage_c_counterfactual import FamilyBatchBuilder, FamilyDataset


VARIANTS = ("base", "edited", "invariant")


def intervention_feature_pairs(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return base/edited cores and the opposite residual source for each row."""
    if features.ndim != 4 or features.shape[1] != len(VARIANTS):
        raise ValueError("features must be [families, 3, patches, width]")
    cores = torch.stack((features[:, 0], features[:, 1]), dim=1)
    residual_sources = torch.stack((features[:, 1], features[:, 0]), dim=1)
    return cores, residual_sources


def intervention_target_pairs(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 2 or targets.shape[1] != len(VARIANTS):
        raise ValueError("targets must be [families, 3]")
    return torch.stack((targets[:, 1], targets[:, 0]), dim=1)


def candidate_scores(
    language,
    bridge,
    text_embeddings: torch.Tensor,
    attention: torch.Tensor,
    labels: torch.Tensor,
    core_features: torch.Tensor,
    residual_features: torch.Tensor | None = None,
) -> torch.Tensor:
    inputs, mask, targets = bridge.inject(
        text_embeddings,
        attention,
        labels,
        core_features,
        binding_residual_features=residual_features,
    )
    logits = language(inputs_embeds=inputs, attention_mask=mask).logits[:, :-1]
    shifted = targets[:, 1:]
    token_mask = shifted != -100
    return masked_sequence_logprob(
        logits, shifted.masked_fill(~token_mask, 0), token_mask
    )


def save_checkpoint(output: Path, bridge, optimizer, state: dict[str, Any]) -> None:
    checkpoint = output / f"step-{state['optimizer_step']:06d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()},
        checkpoint / "bridge.safetensors",
    )
    partial = checkpoint / "optimizer.pt.partial"
    torch.save(optimizer.state_dict(), partial)
    os.replace(partial, checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    write_json_atomic(output / "latest.json", {
        "checkpoint": str(checkpoint),
        "optimizer_step": state["optimizer_step"],
        "bridge_sha256": sha256_file(checkpoint / "bridge.safetensors"),
        "state_sha256": sha256_file(checkpoint / "state.json"),
    })


def validate_data(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    data = protocol["data"]
    manifest = Path(data["manifest"])
    if sha256_file(manifest) != data["manifest_sha256"]:
        raise SystemExit("manifest hash mismatch")
    for label in ("generation_receipt", "static_audit", "human_audit"):
        evidence = data["evidence"][label]
        if sha256_file(Path(evidence["path"])) != evidence["sha256"]:
            raise SystemExit(f"binding data evidence hash mismatch: {label}")
    generation = json.loads(Path(data["evidence"]["generation_receipt"]["path"]).read_text())
    static = json.loads(Path(data["evidence"]["static_audit"]["path"]).read_text())
    human = json.loads(Path(data["evidence"]["human_audit"]["path"]).read_text())
    if generation.get("manifest_sha256") != data["manifest_sha256"]:
        raise SystemExit("generation receipt does not bind manifest")
    if generation.get("final_test_generated") is not False:
        raise SystemExit("final test must remain absent")
    if static.get("status") != "passed_static_and_render_audit_pending_human_review":
        raise SystemExit("static audit did not pass")
    if not all(static.get("gates", {}).values()):
        raise SystemExit("static audit contains a failed gate")
    if human.get("status") != "passed_bounded_human_visual_audit_for_development_only":
        raise SystemExit("human audit did not admit development use")
    if human.get("formal_confirmation_use") is not False:
        raise SystemExit("human audit scope is broader than development")
    all_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if {row["split"] for row in all_rows} != {"train", "mechanism_dev", "selection_dev"}:
        raise SystemExit("binding partition set mismatch")
    rows = [row for row in all_rows if row["split"] == data["split"]]
    if len(rows) != data["scene_families"]:
        raise SystemExit("binding training-family count mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("statistical_cluster_id") != row.get("scene_pair_id"):
            raise SystemExit("binding statistical cluster mismatch")
        grouped.setdefault(row["scene_pair_id"], []).append(row)
    if len(grouped) != data["unique_scene_pairs"]:
        raise SystemExit("binding scene-pair count mismatch")
    if any(len(pair) != data["questions_per_scene_pair"] for pair in grouped.values()):
        raise SystemExit("binding questions-per-pair mismatch")
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
    if protocol.get("status") != "authorized_binding_residual_development_v1":
        raise SystemExit("binding residual protocol is not authorized")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("final test must remain sealed")
    prerequisite = protocol["prerequisite"]
    prerequisite_path = Path(prerequisite["plain_ce_summary"])
    if sha256_file(prerequisite_path) != prerequisite["plain_ce_summary_sha256"]:
        raise SystemExit("plain-CE prerequisite hash mismatch")
    prerequisite_result = json.loads(prerequisite_path.read_text())
    if prerequisite_result.get("plain_ce_decision") != prerequisite["required_decision"]:
        raise SystemExit("plain-CE prerequisite did not admit residual development")
    source_code = protocol["source_code"]
    if sha256_file(Path(__file__)) != source_code["trainer_sha256"]:
        raise SystemExit("binding residual trainer hash mismatch")
    if sha256_file(Path(__file__).with_name("modeling.py")) != source_code["modeling_sha256"]:
        raise SystemExit("binding residual modeling hash mismatch")
    if args.arm not in protocol["arms"]:
        raise SystemExit("arm is not declared in the protocol")
    arm = protocol["arms"][args.arm]
    architecture = arm["architecture"]
    if (
        architecture.get("compressor") != "spatial_query"
        or int(architecture.get("visual_tokens", 0)) != 196
        or int(architecture.get("output_visual_tokens", 0)) != 49
        or int(architecture.get("binding_residual_rank", 0)) != 16
    ):
        raise SystemExit("binding residual arm must be rank-16 on the frozen 196-to-49 path")
    weights = arm["loss_weights"]
    if weights["ordinary_answer"] != 1.0 or weights["interchange_answer"] not in {0.0, 1.0}:
        raise SystemExit("unsupported binding residual objective weights")
    if (args.arm == "residual_ce") != (weights["interchange_answer"] == 0.0):
        raise SystemExit("arm/objective mismatch")
    rows = validate_data(protocol)
    optimization = protocol["optimization"]
    batch_size = int(optimization["micro_batch_families"])
    if batch_size % int(protocol["data"]["questions_per_scene_pair"]):
        raise SystemExit("microbatch must preserve complete scene pairs")
    expected_steps = len(rows) // batch_size
    if expected_steps != optimization["max_optimizer_steps"]:
        raise SystemExit("one-epoch optimizer-step mismatch")
    max_steps = min(args.max_optimizer_steps or expected_steps, expected_steps)
    max_wall = min(args.max_wall_seconds or optimization["max_wall_seconds"], optimization["max_wall_seconds"])

    parents = protocol["parents"]
    base = Path(parents["language"])
    vision_path = Path(parents["vision"])
    parent_bridge = Path(parents["bridge"])
    parent_adapter = Path(parents["language_adapter"])
    if sha256_file(parent_bridge) != parents["bridge_sha256"]:
        raise SystemExit("parent bridge hash mismatch")
    for name, expected in parents["language_adapter_files"].items():
        if sha256_file(parent_adapter / name) != expected:
            raise SystemExit(f"parent adapter hash mismatch: {name}")
    parent_before = {"language": release_records(base), "vision": vision_records(vision_path)}

    torch.manual_seed(optimization["seed"])
    torch.cuda.manual_seed_all(optimization["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_path, local_files_only=True, use_fast=False)
    loader = DataLoader(
        FamilyDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=FamilyBatchBuilder(
            tokenizer,
            processor,
            optimization["text_tokens_max"],
            teacher=None,
            deduplicate_images=True,
        ),
        drop_last=True,
    )
    language = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    from peft import PeftModel

    language = PeftModel.from_pretrained(
        language, parent_adapter, is_trainable=False, local_files_only=True
    )
    language.config.use_cache = False
    freeze(language)
    vision = SiglipVisionModel.from_pretrained(
        vision_path, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    bridge = bridge_from_architecture(architecture).to(device="cuda", dtype=torch.bfloat16)
    missing = load_bridge_parent(bridge, load_file(parent_bridge, device="cpu"), allow_new_compressor=True)
    if not missing or not all(name.startswith("binding_residual.") for name in missing):
        raise RuntimeError(f"unexpected residual parent initialization: {missing}")
    freeze(bridge)
    assert bridge.binding_residual is not None
    bridge.binding_residual.requires_grad_(True)
    bridge.binding_residual.train()
    trainable = [parameter for parameter in bridge.binding_residual.parameters() if parameter.requires_grad]
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    if trainable_parameters != int(arm["trainable_parameters"]):
        raise RuntimeError("binding residual parameter count mismatch")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
        fused=True,
    )
    warmup = max(1, round(max_steps * optimization["warmup_ratio"]))

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
        "trainable_parameters": trainable_parameters,
        "input_visual_tokens": bridge.spec.input_tokens,
        "output_visual_tokens": bridge.spec.output_tokens,
        "binding_residual_rank": bridge.binding_residual.rank,
        "families": len(rows),
        "unique_scene_pairs": protocol["data"]["unique_scene_pairs"],
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
            pixels = batch["pixel_values"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention = batch["attention_mask"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            targets = batch["target_indices"].to("cuda", non_blocking=True)
            with torch.no_grad():
                unique_features = vision_features(
                    vision, pixels, int(architecture.get("vision_feature_layer", -1))
                )
                text = language.get_input_embeddings()(input_ids)
            indices = batch["image_feature_indices"].to("cuda", non_blocking=True)
            features = unique_features[indices.reshape(-1)].reshape(
                targets.shape[0], len(VARIANTS), unique_features.shape[1], unique_features.shape[2]
            )
            text = text.reshape(targets.shape[0], len(VARIANTS), 2, text.shape[1], text.shape[2])
            attention = attention.reshape(targets.shape[0], len(VARIANTS), 2, attention.shape[1])
            labels = labels.reshape(targets.shape[0], len(VARIANTS), 2, labels.shape[1])
            normal_scores = candidate_scores(
                language,
                bridge,
                text.flatten(0, 2),
                attention.flatten(0, 2),
                labels.flatten(0, 2),
                features.unsqueeze(2).expand(-1, -1, 2, -1, -1).flatten(0, 2),
            ).reshape(targets.shape[0], len(VARIANTS), 2)
            ordinary_loss = F.cross_entropy(normal_scores.flatten(0, 1), targets.flatten())
            intervention_loss = torch.zeros((), device="cuda")
            intervention_accuracy = torch.zeros((), device="cuda")
            if weights["interchange_answer"] > 0:
                core_features, residual_features = intervention_feature_pairs(features)
                intervention_targets = intervention_target_pairs(targets)
                intervention_scores = candidate_scores(
                    language,
                    bridge,
                    text[:, :2].flatten(0, 2),
                    attention[:, :2].flatten(0, 2),
                    labels[:, :2].flatten(0, 2),
                    core_features.unsqueeze(2).expand(-1, -1, 2, -1, -1).flatten(0, 2),
                    residual_features.unsqueeze(2).expand(-1, -1, 2, -1, -1).flatten(0, 2),
                ).reshape(targets.shape[0], 2, 2)
                intervention_loss = F.cross_entropy(
                    intervention_scores.flatten(0, 1), intervention_targets.flatten()
                )
                intervention_accuracy = (
                    intervention_scores.argmax(dim=-1) == intervention_targets
                ).float().mean()
            loss = (
                weights["ordinary_answer"] * ordinary_loss
                + weights["interchange_answer"] * intervention_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite binding residual loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, optimization["gradient_clip_norm"]
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite binding residual gradient")
            optimizer_step += 1
            scale = cosine_lr(optimizer_step, max_steps, warmup, optimization["minimum_learning_rate_ratio"])
            learning_rate = optimization["learning_rate"] * scale
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.step()
            elapsed = time.monotonic() - started
            event = {
                "optimizer_step": optimizer_step,
                "total_loss": float(loss.detach()),
                "ordinary_loss": float(ordinary_loss.detach()),
                "intervention_loss": float(intervention_loss.detach()),
                "ordinary_candidate_accuracy": float(
                    (normal_scores.argmax(dim=-1) == targets).float().mean().detach()
                ),
                "intervention_candidate_accuracy": float(intervention_accuracy.detach()),
                "gradient_norm": float(grad_norm),
                "learning_rate": learning_rate,
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
                save_checkpoint(args.output, bridge, optimizer, run)
            if optimizer_step >= max_steps:
                stop_reason = "optimizer_step_cap"
                break
        run["status"] = "complete"
        run["stop_reason"] = stop_reason
        run["completed_at"] = utc_now()
        run["wall_seconds"] = time.monotonic() - started
        run["total_wall_seconds"] = time.monotonic() - process_started
        if optimizer_step and optimizer_step % optimization["checkpoint_every_steps"]:
            save_checkpoint(args.output, bridge, optimizer, run)
        parent_after = {"language": release_records(base), "vision": vision_records(vision_path)}
        if parent_after != parent_before or sha256_file(parent_bridge) != parents["bridge_sha256"]:
            raise RuntimeError("immutable parent changed")
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
