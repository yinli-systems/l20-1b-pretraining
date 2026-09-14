#!/usr/bin/env python3
"""Cache full-token teacher candidate scores for frozen train families."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from counterfactual_losses import masked_sequence_logprob
from evaluate_stage_a_visual_floor import collate_candidates, directory_records
from modeling import bridge_from_architecture, freeze, vision_features
from train_stage_a_full_token import release_records, sha256_file, utc_now, vision_records, write_json_atomic


VARIANTS = ("base", "edited", "invariant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-families", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("teacher cache output already exists")
    if args.batch_families < 1:
        raise SystemExit("batch-families must be positive")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("not_a_compression_experiment") is not True:
        raise SystemExit("teacher must be the admitted full-token reference")
    architecture = protocol["architecture"]
    if architecture.get("compressor") != "none":
        raise SystemExit("teacher protocol must use the full 196-token bridge")
    if sha256_file(args.manifest) != protocol["data"]["manifest_sha256"]:
        raise SystemExit("manifest hash mismatch")
    selection = protocol.get("selection_evidence")
    if selection and sha256_file(Path(selection["path"])) != selection["sha256"]:
        raise SystemExit("full-token selection evidence hash mismatch")
    manifest_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    rows = sorted(
        [row for row in manifest_rows if row["split"] == "train"],
        key=lambda row: row["scene_family_id"],
    )
    if len(rows) != protocol["data"]["scene_families"]:
        raise RuntimeError("unexpected teacher-cache family count")

    base = Path(protocol["parents"]["language"])
    vision_path = Path(protocol["parents"]["vision"])
    checkpoint = Path(protocol["parents"]["bridge"])
    adapter = Path(protocol["parents"]["language_adapter"])
    if sha256_file(checkpoint) != protocol["parents"]["bridge_sha256"]:
        raise SystemExit("teacher bridge hash mismatch")
    for name, expected in protocol["parents"]["language_adapter_files"].items():
        if sha256_file(adapter / name) != expected:
            raise SystemExit(f"teacher adapter hash mismatch: {name}")
    parent_before = {"language": release_records(base), "vision": vision_records(vision_path)}

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision_path, local_files_only=True)
    language = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda()
    language = PeftModel.from_pretrained(
        language, adapter, is_trainable=False, local_files_only=True
    )
    freeze(language)
    vision = SiglipVisionModel.from_pretrained(
        vision_path, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    freeze(vision)
    bridge = bridge_from_architecture(architecture).to(device="cuda", dtype=torch.bfloat16)
    bridge.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
    freeze(bridge)
    feature_layer = int(architecture.get("vision_feature_layer", -1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    started = time.monotonic()
    variant_correct = {variant: 0 for variant in VARIANTS}
    pair_correct = 0
    invariant_pair_correct = 0
    with partial.open("w") as handle, torch.inference_mode():
        for start in range(0, len(rows), args.batch_families):
            batch = rows[start : start + args.batch_families]
            images = []
            prompts = []
            answers = []
            expected_indices = []
            for row in batch:
                for variant in VARIANTS:
                    with Image.open(row[f"{variant}_image_path"]) as image:
                        images.append(image.convert("RGB").copy())
                    for answer in row["candidate_answers"]:
                        prompts.append(row["question"])
                        answers.append(answer)
                    expected_indices.append(row["candidate_answers"].index(row[f"{variant}_answer"]))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
                "cuda", dtype=torch.bfloat16
            )
            features = vision_features(vision, pixels, feature_layer).repeat_interleave(2, dim=0)
            input_ids, attention, labels = collate_candidates(
                tokenizer, prompts, answers, args.max_tokens
            )
            input_ids = input_ids.cuda(non_blocking=True)
            attention = attention.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            text = language.get_input_embeddings()(input_ids)
            inputs, mask, targets = bridge.inject(text, attention, labels, features)
            logits = language(inputs_embeds=inputs, attention_mask=mask).logits[:, :-1]
            shifted = targets[:, 1:]
            token_mask = shifted != -100
            scores = masked_sequence_logprob(
                logits, shifted.masked_fill(~token_mask, 0), token_mask
            ).view(len(batch), len(VARIANTS), 2).float().cpu()
            predicted = scores.argmax(dim=-1)
            for index, row in enumerate(batch):
                variants = {}
                correct_flags = []
                for variant_index, variant in enumerate(VARIANTS):
                    expected = expected_indices[index * len(VARIANTS) + variant_index]
                    is_correct = int(predicted[index, variant_index]) == expected
                    variant_correct[variant] += int(is_correct)
                    correct_flags.append(is_correct)
                    variants[variant] = {
                        "candidate_scores": [float(value) for value in scores[index, variant_index]],
                        "yes_minus_no_preference": float(scores[index, variant_index, 0] - scores[index, variant_index, 1]),
                        "expected_candidate_index": expected,
                        "correct": is_correct,
                    }
                pair_correct += int(correct_flags[0] and correct_flags[1])
                invariant_pair_correct += int(correct_flags[0] and correct_flags[2])
                handle.write(json.dumps({
                    "scene_family_id": row["scene_family_id"],
                    "task": row["task"],
                    "split": row["split"],
                    "candidate_answers": row["candidate_answers"],
                    "variants": variants,
                    "teacher_pair_correct": bool(correct_flags[0] and correct_flags[1]),
                    "teacher_invariant_pair_correct": bool(correct_flags[0] and correct_flags[2]),
                }, sort_keys=True) + "\n")
            if (start // args.batch_families + 1) % 100 == 0 or start + len(batch) == len(rows):
                print(json.dumps({
                    "families_cached": start + len(batch),
                    "families_total": len(rows),
                    "elapsed_seconds": time.monotonic() - started,
                }, sort_keys=True), flush=True)
    os.replace(partial, args.output)
    parent_after = {"language": release_records(base), "vision": vision_records(vision_path)}
    if parent_after != parent_before:
        raise RuntimeError("immutable teacher parents changed")
    receipt = {
        "schema_version": "2026-09-13-v1",
        "status": "complete",
        "completed_at": utc_now(),
        "families": len(rows),
        "variants_per_family": len(VARIANTS),
        "candidate_answers": ["yes", "no"],
        "cache": str(args.output),
        "cache_sha256": sha256_file(args.output),
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_sha256": sha256_file(args.protocol),
        "teacher_bridge_sha256": sha256_file(checkpoint),
        "teacher_adapter_records": directory_records(adapter),
        "teacher_variant_accuracy_percent": {
            variant: 100.0 * count / len(rows) for variant, count in variant_correct.items()
        },
        "teacher_pair_coverage_percent": 100.0 * pair_correct / len(rows),
        "teacher_invariant_pair_coverage_percent": 100.0 * invariant_pair_correct / len(rows),
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "parent_before": parent_before,
        "parent_after": parent_after,
        "test_split_rows": 0,
        "claim_boundary": "Train-split teacher scores are a compute cache, not an evaluation result.",
    }
    write_json_atomic(args.receipt, receipt)
    print(json.dumps({k: receipt[k] for k in (
        "families", "cache_sha256", "teacher_variant_accuracy_percent",
        "teacher_pair_coverage_percent", "teacher_invariant_pair_coverage_percent",
        "wall_seconds", "peak_allocated_gib",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
