#!/usr/bin/env python3
"""Recover full-token visual use with a language LoRA and trainable bridge."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from modeling import bridge_from_architecture, freeze, load_bridge_parent, vision_features
from counterfactual_losses import candidate_classification_loss
from train_stage_a_full_token import (
    BatchBuilder,
    ManifestDataset,
    cosine_lr,
    encode_prompt_response,
    release_records,
    sha256_file,
    utc_now,
    vision_records,
    write_json_atomic,
)
from train_stage_a_recovery import recovery_rows


ROOT = Path(__file__).resolve().parent


@dataclass
class CandidateBatchBuilder:
    """Collate one image and both audited answer candidates per example."""

    tokenizer: Any
    processor: Any
    max_tokens: int

    def __call__(self, rows: list[dict]) -> dict[str, Any]:
        images = []
        encoded = []
        target_indices = []
        for row in rows:
            candidates = row.get("candidate_answers", ["yes", "no"])
            if candidates != ["yes", "no"]:
                raise RuntimeError(f"unexpected candidate order: {candidates}")
            if row["response"] not in candidates:
                raise RuntimeError(f"response is not an audited candidate: {row['response']}")
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB").copy())
            encoded.extend(
                encode_prompt_response(self.tokenizer, row["prompt"], answer, self.max_tokens)
                for answer in candidates
            )
            target_indices.append(candidates.index(row["response"]))
        width = max(len(input_ids) for input_ids, _ in encoded)
        input_ids = torch.full(
            (len(encoded), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(encoded), width), -100, dtype=torch.long)
        attention = torch.zeros((len(encoded), width), dtype=torch.long)
        for index, (ids, targets) in enumerate(encoded):
            length = len(ids)
            input_ids[index, :length] = torch.tensor(ids)
            labels[index, :length] = torch.tensor(targets)
            attention[index, :length] = 1
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": pixels.contiguous(),
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
            "sources": [row["source"] for row in rows],
        }


def save_checkpoint(output: Path, bridge: MultimodalBridge, language, optimizer, state: dict) -> None:
    checkpoint = output / f"step-{state['optimizer_step']:06d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()},
        checkpoint / "bridge.safetensors",
    )
    adapter = checkpoint / "language_adapter"
    language.save_pretrained(adapter, safe_serialization=True)
    optimizer_partial = checkpoint / "optimizer.pt.partial"
    torch.save(optimizer.state_dict(), optimizer_partial)
    os.replace(optimizer_partial, checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    adapter_model = adapter / "adapter_model.safetensors"
    write_json_atomic(
        output / "latest.json",
        {
            "checkpoint": str(checkpoint),
            "optimizer_step": state["optimizer_step"],
            "bridge_sha256": sha256_file(checkpoint / "bridge.safetensors"),
            "adapter_model_sha256": sha256_file(adapter_model),
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "state_sha256": sha256_file(checkpoint / "state.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "stage_a_lora_recovery_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    allowed_statuses = {
        "authorized_stage_a_lora_recovery_after_frozen_language_failure",
        "authorized_stage_a_discriminative_recovery_after_generative_underfit",
        "authorized_stage_a_task_balanced_recovery_after_full_gate_failure",
        "authorized_stage_a_penultimate_feature_probe_after_dev_weakness",
        "authorized_stage_a_unique_scene_recovery_after_dev_overfit_diagnosis",
        "authorized_stage_b_answer_supervision_compression_gap_pilot",
    }
    if protocol.get("status") not in allowed_statuses:
        raise SystemExit("LoRA recovery protocol is not authorized")
    compression_experiment = bool(protocol.get("compression_experiment", False))
    if compression_experiment:
        if protocol.get("status") != "authorized_stage_b_answer_supervision_compression_gap_pilot":
            raise SystemExit("unsupported compression experiment status")
        if protocol.get("proposed_delta_loss_used"):
            raise SystemExit("Stage-B gap pilot cannot use the proposed delta loss")
    elif not protocol.get("not_a_compression_experiment") or protocol.get("proposed_delta_loss_used"):
        raise SystemExit("LoRA recovery must remain outside the compression experiment")
    if protocol.get("test_split_use_authorized") is not False:
        raise SystemExit("test split must remain sealed")
    failure = Path(protocol["failure_evidence"]["path"])
    if sha256_file(failure) != protocol["failure_evidence"]["sha256"]:
        raise SystemExit("frozen-language failure evidence hash mismatch")
    data = protocol["data"]
    manifest = Path(data["manifest"])
    if sha256_file(manifest) != data["manifest_sha256"]:
        raise SystemExit("counterfactual manifest hash mismatch")
    audit = json.loads(Path(data["human_audit"]).read_text())
    if audit.get("decision") != data["human_audit_decision"]:
        raise SystemExit("counterfactual human audit mismatch")
    optimization = protocol["optimization"]
    objective = protocol["architecture"].get("objective", "answer_only_causal_cross_entropy")
    vision_feature_layer = int(protocol["architecture"].get("vision_feature_layer", -1))
    candidate_mode = objective == "two_candidate_sequence_logprob_cross_entropy"
    if objective not in {
        "answer_only_causal_cross_entropy",
        "two_candidate_sequence_logprob_cross_entropy",
    }:
        raise SystemExit(f"unsupported recovery objective: {objective}")
    max_steps = min(args.max_optimizer_steps or optimization["max_optimizer_steps"], optimization["max_optimizer_steps"])
    max_wall_seconds = min(args.max_wall_seconds or optimization["max_wall_seconds"], optimization["max_wall_seconds"])
    if max_steps < 1 or max_wall_seconds < 1:
        raise SystemExit("invalid bounded run")

    base = Path(protocol["parents"]["language"])
    vision_path = Path(protocol["parents"]["vision"])
    initial_bridge = Path(protocol["parents"]["bridge"])
    if sha256_file(initial_bridge) != protocol["parents"]["bridge_sha256"]:
        raise SystemExit("selected bridge hash mismatch")
    parent_before = {"language": release_records(base), "vision": vision_records(vision_path)}
    rows = recovery_rows(
        manifest,
        data["split"],
        data["epochs"],
        optimization["seed"],
        data.get("task_repeats"),
        data["scene_families"],
    )
    if len(rows) != data["examples_per_epoch"] * data["epochs"]:
        raise RuntimeError("LoRA recovery example count mismatch")
    answers = {answer: sum(row["response"] == answer for row in rows) for answer in ("yes", "no")}
    if answers != data["answer_balance"]:
        raise RuntimeError(f"LoRA recovery answer imbalance: {answers}")

    seed = optimization["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_path, local_files_only=True)
    loader = DataLoader(
        ManifestDataset(rows),
        batch_size=optimization["micro_batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=(CandidateBatchBuilder if candidate_mode else BatchBuilder)(
            tokenizer, processor, optimization["text_tokens_max"]
        ),
        drop_last=True,
    )
    language = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).cuda()
    language.config.use_cache = False
    lora = protocol["architecture"]["lora"]
    initial_adapter = protocol["parents"].get("language_adapter")
    initial_adapter_hashes = None
    if initial_adapter is None:
        language = get_peft_model(
            language,
            LoraConfig(
                r=lora["rank"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                bias=lora["bias"],
                target_modules=lora["target_modules"],
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    else:
        initial_adapter = Path(initial_adapter)
        initial_adapter_hashes = protocol["parents"]["language_adapter_files"]
        for name, expected in initial_adapter_hashes.items():
            if sha256_file(initial_adapter / name) != expected:
                raise SystemExit(f"initial language adapter hash mismatch: {name}")
        language = PeftModel.from_pretrained(
            language, initial_adapter, is_trainable=True, local_files_only=True
        )
    language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    language.train()
    vision = SiglipVisionModel.from_pretrained(vision_path, dtype=torch.bfloat16, local_files_only=True).cuda()
    freeze(vision)
    bridge = bridge_from_architecture(protocol["architecture"]).to(device="cuda", dtype=torch.bfloat16)
    missing_bridge_parent_keys = load_bridge_parent(
        bridge,
        load_file(initial_bridge, device="cpu"),
        allow_new_compressor=bool(protocol["architecture"].get("initialize_new_compressor", False)),
    )
    bridge.train()
    lora_parameters = [(name, parameter) for name, parameter in language.named_parameters() if parameter.requires_grad]
    if not lora_parameters or any("lora_" not in name for name, _ in lora_parameters):
        raise RuntimeError("unexpected trainable language parameters")
    compressor_parameters = (
        [] if bridge.compressor is None else [parameter for parameter in bridge.compressor.parameters() if parameter.requires_grad]
    )
    compressor_ids = {id(parameter) for parameter in compressor_parameters}
    bridge_parameters = [parameter for parameter in bridge.parameters() if parameter.requires_grad]
    non_compressor_bridge_parameters = [parameter for parameter in bridge_parameters if id(parameter) not in compressor_ids]
    optimizer_groups = [
        {"params": [parameter for _, parameter in lora_parameters], "lr": optimization["adapter_learning_rate"]},
        {"params": non_compressor_bridge_parameters, "lr": optimization["bridge_learning_rate"]},
    ]
    if compressor_parameters:
        optimizer_groups.append({
            "params": compressor_parameters,
            "lr": optimization.get("compressor_learning_rate", optimization["bridge_learning_rate"]),
        })
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=optimization["weight_decay"],
        fused=True,
    )
    accumulation = optimization["gradient_accumulation_steps"]
    warmup = max(1, round(max_steps * optimization["warmup_ratio"]))
    args.output.mkdir(parents=True)
    log_path = args.output / "train.jsonl"
    run = {
        "schema_version": "2026-09-13-v1",
        "status": "running",
        "started_at": utc_now(),
        "device": torch.cuda.get_device_name(),
        "protocol_sha256": sha256_file(args.protocol),
        "runner_sha256": sha256_file(Path(__file__)),
        "modeling_sha256": sha256_file(ROOT / "modeling.py"),
        "manifest_sha256": sha256_file(manifest),
        "initial_bridge_sha256": sha256_file(initial_bridge),
        "initial_language_adapter": None if initial_adapter is None else str(initial_adapter),
        "initial_language_adapter_hashes": initial_adapter_hashes,
        "objective": objective,
        "vision_feature_layer": vision_feature_layer,
        "compression_experiment": compression_experiment,
        "input_visual_tokens": bridge.spec.input_tokens,
        "output_visual_tokens": bridge.spec.output_tokens,
        "achieved_compression_ratio": bridge.spec.achieved_ratio,
        "missing_bridge_parent_keys": missing_bridge_parent_keys,
        "parent_before": parent_before,
        "examples": len(rows),
        "answer_counts": answers,
        "lora_trainable_parameters": sum(parameter.numel() for _, parameter in lora_parameters),
        "bridge_trainable_parameters": sum(parameter.numel() for parameter in bridge_parameters),
        "max_optimizer_steps": max_steps,
        "max_wall_seconds": max_wall_seconds,
        "optimizer_step": 0,
        "micro_step": 0,
        "prediction_tokens": 0,
    }
    write_json_atomic(args.output / "run.json", run)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    accumulated_loss = 0.0
    accumulated_accuracy = 0.0
    optimizer_step = 0
    prediction_tokens = 0
    total_tokens = 0
    stop_reason = "data_exhausted"
    try:
        for micro_step, batch in enumerate(loader, 1):
            if time.monotonic() - started >= max_wall_seconds:
                stop_reason = "wall_time_cap"
                break
            pixels = batch["pixel_values"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention = batch["attention_mask"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            with torch.no_grad():
                visual_features = vision_features(vision, pixels, vision_feature_layer)
                text_embeddings = language.get_input_embeddings()(input_ids)
            if candidate_mode:
                visual_features = visual_features.repeat_interleave(2, dim=0)
            inputs, mask, targets = bridge.inject(text_embeddings.detach(), attention, labels, visual_features)
            if candidate_mode:
                logits = language(inputs_embeds=inputs, attention_mask=mask).logits
                target_indices = batch["target_indices"].to("cuda", non_blocking=True)
                loss, candidate_accuracy = candidate_classification_loss(
                    logits,
                    targets,
                    target_indices,
                    float(optimization.get("candidate_temperature", 1.0)),
                )
            else:
                loss = language(inputs_embeds=inputs, attention_mask=mask, labels=targets).loss
                candidate_accuracy = torch.tensor(float("nan"), device=loss.device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at micro step {micro_step}")
            (loss / accumulation).backward()
            accumulated_loss += float(loss.detach())
            if candidate_mode:
                accumulated_accuracy += float(candidate_accuracy.detach())
            prediction_tokens += int((labels != -100).sum())
            total_tokens += int(attention.sum()) + labels.shape[0] * (bridge.spec.output_tokens + 2)
            run["micro_step"] = micro_step
            if micro_step % accumulation:
                continue
            optimizer_step += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(bridge_parameters + [parameter for _, parameter in lora_parameters], optimization["gradient_clip_norm"])
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at step {optimizer_step}")
            lr_scale = cosine_lr(optimizer_step, max_steps, warmup, optimization["minimum_learning_rate_ratio"])
            learning_rates = [
                optimization["adapter_learning_rate"] * lr_scale,
                optimization["bridge_learning_rate"] * lr_scale,
            ]
            if compressor_parameters:
                learning_rates.append(
                    optimization.get("compressor_learning_rate", optimization["bridge_learning_rate"]) * lr_scale
                )
            for group, learning_rate in zip(optimizer.param_groups, learning_rates):
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            event = {
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "loss_mean": accumulated_loss / accumulation,
                "candidate_accuracy_mean": (
                    accumulated_accuracy / accumulation if candidate_mode else None
                ),
                "gradient_norm": float(grad_norm),
                "adapter_learning_rate": learning_rates[0],
                "bridge_learning_rate": learning_rates[1],
                "compressor_learning_rate": learning_rates[2] if compressor_parameters else None,
                "prediction_tokens": prediction_tokens,
                "prediction_tokens_per_second": prediction_tokens / elapsed,
                "total_input_and_visual_tokens_per_second": total_tokens / elapsed,
                "elapsed_seconds": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "timestamp": utc_now(),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            accumulated_loss = 0.0
            accumulated_accuracy = 0.0
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
        if optimizer_step and optimizer_step % optimization["checkpoint_every_steps"]:
            save_checkpoint(args.output, bridge, language, optimizer, run)
        parent_after = {"language": release_records(base), "vision": vision_records(vision_path)}
        if parent_after != parent_before:
            raise RuntimeError("immutable parent files changed during LoRA recovery")
        if sha256_file(initial_bridge) != protocol["parents"]["bridge_sha256"]:
            raise RuntimeError("initial bridge changed during LoRA recovery")
        if initial_adapter is not None:
            for name, expected in initial_adapter_hashes.items():
                if sha256_file(initial_adapter / name) != expected:
                    raise RuntimeError(f"initial language adapter changed: {name}")
        run["parent_after"] = parent_after
        write_json_atomic(args.output / "run.json", run)
    except Exception as error:
        run["status"] = "failed"
        run["failed_at"] = utc_now()
        run["error"] = f"{type(error).__name__}: {error}"
        run["wall_seconds"] = time.monotonic() - started
        write_json_atomic(args.output / "run.json", run)
        raise


if __name__ == "__main__":
    main()
