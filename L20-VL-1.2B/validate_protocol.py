#!/usr/bin/env python3
"""Fail-closed checks for the bounded L20-VL-1.2B stage protocol."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).parent
    protocol = json.loads((root / "protocol.json").read_text())
    errors: list[str] = []
    if protocol.get("automatic_stage_advance") is not False:
        errors.append("automatic stage advance must be disabled")
    if protocol.get("publication_authorized") is not False:
        errors.append("publication must remain unauthorized")
    if protocol.get("formal_training_authorized") is not False:
        errors.append("formal training must remain unauthorized")
    if protocol.get("quality_gate_registry") != "quality_gates.json":
        errors.append("protocol must pin quality_gates.json")
    if protocol["parents"]["language"]["source"] != "released_20B_base_not_continuation_A_or_B":
        errors.append("language parent must be the released Base")
    vision = protocol["parents"]["vision_candidate"]
    if not vision["initially_frozen"]:
        errors.append("vision weights must remain frozen at Stage 0")
    if vision["weights_downloaded"]:
        for key in ("weights_receipt", "forward_smoke_receipt", "integrated_benchmark_receipt"):
            receipt = root / vision[key]
            if not receipt.is_file():
                errors.append(f"downloaded vision weights require {key}: {receipt}")
        smoke_path = root / vision["forward_smoke_receipt"]
        acquisition_path = root / vision["weights_receipt"]
        if acquisition_path.is_file():
            acquisition = json.loads(acquisition_path.read_text())
            if acquisition.get("status") != "verified_and_installed_remote_only":
                errors.append("vision acquisition receipt is not verified")
            if acquisition["files"]["model.safetensors"].get("sha256") != vision.get("weights_sha256"):
                errors.append("vision acquisition and protocol weight hashes differ")
        if smoke_path.is_file():
            smoke = json.loads(smoke_path.read_text())
            if smoke.get("status") != "pass" or smoke.get("training_prediction_tokens") != 0:
                errors.append("vision smoke must pass without training tokens")
            if smoke["model"].get("weights_sha256") != vision.get("weights_sha256"):
                errors.append("vision smoke and protocol weight hashes differ")
            if acquisition_path.is_file() and sha256(smoke_path) != acquisition["validation"].get("smoke_receipt_sha256"):
                errors.append("vision smoke receipt hash differs from acquisition receipt")
        integrated_path = root / vision["integrated_benchmark_receipt"]
        if integrated_path.is_file():
            integrated = json.loads(integrated_path.read_text())
            if integrated.get("formal_training") is not False or integrated.get("training_prediction_tokens") != 0:
                errors.append("integrated Stage-0 benchmark cannot be formal training")
            if integrated.get("real_multimodal_data") is not False:
                errors.append("integrated Stage-0 benchmark must declare synthetic inputs")
            if integrated["base_weights_before"] != integrated["base_weights_after"]:
                errors.append("Base changed during integrated Stage-0 benchmark")
            if integrated["vision_weights_before"] != integrated["vision_weights_after"]:
                errors.append("vision encoder changed during integrated Stage-0 benchmark")
        real_image_key = "real_image_forward_only_receipt"
        real_image_path = root / vision.get(real_image_key, "missing-real-image-receipt")
        if not real_image_path.is_file():
            errors.append(f"real-image forward-only receipt is missing: {real_image_path}")
        else:
            real_image = json.loads(real_image_path.read_text())
            if real_image.get("status") != "pass":
                errors.append("real-image forward-only systems test did not pass")
            if real_image.get("formal_training") is not False:
                errors.append("real-image Stage-0 systems test cannot be formal training")
            if real_image.get("optimizer_constructed") is not False:
                errors.append("real-image Stage-0 systems test cannot construct an optimizer")
            if real_image.get("backward_calls") != 0 or real_image.get("training_prediction_tokens") != 0:
                errors.append("real-image Stage-0 systems test must have zero updates and training tokens")
            if real_image.get("real_multimodal_data") is not True:
                errors.append("real-image Stage-0 systems test must declare real multimodal data")
            if real_image.get("quality_claim_authorized") is not False:
                errors.append("real-image Stage-0 systems test cannot authorize a quality claim")
            if real_image.get("parent_artifacts_before") != real_image.get("parent_artifacts_after"):
                errors.append("a parent artifact changed during the real-image systems test")
            if [arm.get("target_ratio") for arm in real_image.get("arms", [])] != [1, 4, 9, 16]:
                errors.append("real-image systems test must cover frozen ratios 1, 4, 9, and 16")
            for arm in real_image.get("arms", []):
                if arm.get("bridge_requires_grad_parameters") != 0:
                    errors.append("real-image systems test bridge cannot require gradients")
                if arm.get("bridge_state_sha256_before") != arm.get("bridge_state_sha256_after"):
                    errors.append("bridge changed during the real-image systems test")
    if protocol["data_policy"]["download_allowed_before_audit"] is not False:
        errors.append("data download must be blocked before audit")
    if protocol["data_policy"].get("admitted_source_count") != 0:
        errors.append("Stage 0 must have zero admitted sources")
    stages = {stage["id"]: stage for stage in protocol["stages"]}
    if stages["stage-0"]["training_prediction_tokens"] != 0:
        errors.append("Stage 0 cannot train")
    if stages["stage-3-multimodal-instruction-tuning"]["status"] != "not_authorized_requires_new_budget":
        errors.append("Stage 3 needs a new budget")
    if not all("blocked" in item["status"] for name, item in protocol["data_policy"]["candidates"].items()):
        errors.append("all candidate datasets must remain blocked pending audit")
    if errors:
        raise SystemExit("FAIL\n" + "\n".join(errors))
    print("PASS: Stage 0 only; no data download, formal training, promotion, or publication authorized")


if __name__ == "__main__":
    main()
