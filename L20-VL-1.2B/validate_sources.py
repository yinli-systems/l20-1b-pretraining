#!/usr/bin/env python3
"""Fail closed until every multimodal source has a reviewed admission receipt."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent
    registry = json.loads((root / "source_registry.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    gates = json.loads((root / "quality_gates.json").read_text())
    components = json.loads((root / "component_admission.json").read_text())
    pixmo_metadata = json.loads((root / "pixmo_cap_metadata_admission.json").read_text())
    pixmo_images = json.loads((root / "pixmo_image_audit_admission.json").read_text())
    errors: list[str] = []

    if registry.get("download_authorized") is not False:
        errors.append("registry must not authorize downloads at Stage 0")
    if registry.get("training_admission_authorized") is not False:
        errors.append("registry must not authorize training admission at Stage 0")
    if registry.get("admitted_sources"):
        errors.append("Stage 0 admitted_sources must be empty")
    expected_exceptions = {
        "nvidia/Llama-Nemotron-VLM-Dataset-v1:ocr_1",
        "nvidia/Llama-Nemotron-VLM-Dataset-v1:ocr_3",
    }
    if set(registry.get("acquisition_only_component_exceptions", [])) != expected_exceptions:
        errors.append("only the frozen ocr_1/ocr_3 acquisition exceptions are allowed")
    expected_metadata_exceptions = {
        "allenai/pixmo-cap@edce6390d9d5be6c8db0d863fbe62718c88988a4"
    }
    if set(registry.get("metadata_only_acquisition_exceptions", [])) != expected_metadata_exceptions:
        errors.append("only the frozen PixMo-Cap metadata acquisition exception is allowed")
    expected_image_audit_exceptions = {
        "allenai/pixmo-cap@edce6390d9d5be6c8db0d863fbe62718c88988a4:pixmo.s3:288"
    }
    if set(registry.get("image_audit_acquisition_exceptions", [])) != expected_image_audit_exceptions:
        errors.append("only the frozen 288-image PixMo-Cap audit exception is allowed")
    if pixmo_metadata.get("image_download_authorized") is not False:
        errors.append("PixMo-Cap image download must remain blocked")
    if pixmo_metadata.get("training_authorized") is not False:
        errors.append("PixMo-Cap training must remain blocked")
    if pixmo_metadata.get("repo") != "allenai/pixmo-cap":
        errors.append("unexpected PixMo-Cap metadata repository")
    if pixmo_metadata.get("revision") != "edce6390d9d5be6c8db0d863fbe62718c88988a4":
        errors.append("unexpected PixMo-Cap metadata revision")
    pixmo_total = 0
    for filename, metadata in pixmo_metadata.get("files", {}).items():
        if not filename.startswith("data/") or not filename.endswith(".parquet"):
            errors.append(f"PixMo-Cap metadata path is not a Parquet shard: {filename}")
        if ".." in Path(filename).parts or Path(filename).is_absolute():
            errors.append(f"PixMo-Cap unsafe metadata path: {filename}")
        if len(metadata.get("sha256", "")) != 64 or metadata.get("bytes", 0) <= 0:
            errors.append(f"PixMo-Cap incomplete metadata for {filename}")
        pixmo_total += metadata.get("bytes", 0)
    if pixmo_total != pixmo_metadata.get("max_bytes"):
        errors.append("PixMo-Cap metadata byte budget mismatch")
    if pixmo_images.get("image_audit_acquisition_authorized") is not True:
        errors.append("the bounded PixMo-Cap image audit must be explicitly authorized")
    if pixmo_images.get("formal_training_authorized") is not False:
        errors.append("PixMo-Cap image audit cannot authorize training")
    if pixmo_images.get("forward_only_system_test_authorized") is not True:
        errors.append("PixMo-Cap audited images are authorized only for forward-only systems testing")
    if pixmo_images.get("host_allowlist") != ["pixmo.s3.us-west-2.amazonaws.com"]:
        errors.append("PixMo-Cap image audit host allowlist changed")
    if sum(pixmo_images.get("path_families", {}).values()) != pixmo_images.get("max_files"):
        errors.append("PixMo-Cap image audit family quotas do not equal max_files")
    if pixmo_images.get("max_files") != 288:
        errors.append("PixMo-Cap image audit file cap changed")
    if not 0 < pixmo_images.get("max_total_bytes", 0) <= 1_500_000_000:
        errors.append("PixMo-Cap image audit byte cap is invalid")
    if components.get("formal_training_authorized") is not False:
        errors.append("component registry cannot authorize formal training")
    if components.get("collection_download_authorized") is not False:
        errors.append("whole-collection download must remain blocked")
    total_bytes = 0
    for component_name, component in components.get("components", {}).items():
        if component_name not in {"ocr_1", "ocr_3"}:
            errors.append(f"unexpected acquisition component: {component_name}")
        if component.get("acquisition_status") != "authorized_for_integrity_and_quality_audit_only":
            errors.append(f"{component_name}: acquisition is not audit-only")
        training_status = component.get("training_status", "")
        if not training_status.startswith(("blocked_", "rejected_")):
            errors.append(f"{component_name}: training must remain blocked or rejected")
        for filename, metadata in component.get("files", {}).items():
            if ".." in Path(filename).parts or Path(filename).is_absolute():
                errors.append(f"{component_name}: unsafe file path {filename}")
            if len(metadata.get("sha256", "")) != 64 or metadata.get("bytes", 0) <= 0:
                errors.append(f"{component_name}: incomplete file metadata for {filename}")
            total_bytes += metadata.get("bytes", 0)
    if total_bytes > components["acquisition_budget"]["max_bytes"]:
        errors.append("component acquisition exceeds its frozen byte budget")
    if protocol["data_policy"].get("source_registry") != "source_registry.json":
        errors.append("protocol must pin source_registry.json")
    if gates.get("formal_training_authorized") is not False:
        errors.append("quality gates cannot authorize formal training")
    if gates.get("evaluation_unit") != "image_or_source_family_not_individual_qa_row":
        errors.append("confidence intervals must cluster repeated QA rows by image/source family")
    if gates["confidence_interval"].get("resamples", 0) < 10000:
        errors.append("paired clustered bootstrap must use at least 10,000 resamples")
    stage_1 = gates["stage_1"]
    if stage_1["selection"].get("throughput_alone_can_select") is not False:
        errors.append("throughput alone cannot select a Stage-1 winner")
    if stage_1["selection"].get("no_winner_allowed") is not True:
        errors.append("the quality protocol must allow a no-winner outcome")

    sources = registry.get("sources", {})
    if not sources:
        errors.append("source registry cannot be empty")
    for name, source in sources.items():
        if len(source.get("revision", "")) != 40:
            errors.append(f"{name}: immutable 40-character revision is required")
        expected_download_status = (
            "blocked_except_pinned_metadata_and_bounded_288_image_audit"
            if name == "pixmo_cap"
            else "blocked"
        )
        if source.get("download_status") != expected_download_status:
            errors.append(f"{name}: downloads must remain blocked")
        if not source.get("training_status", "").startswith("blocked_"):
            errors.append(f"{name}: training status must remain blocked")
        if not source.get("unresolved"):
            errors.append(f"{name}: unresolved audit list cannot be empty")

    if errors:
        raise SystemExit("FAIL\n" + "\n".join(errors))
    print(
        f"PASS: {len(sources)} source candidates; only 2 audit-only component acquisitions, 1 "
        "metadata-only acquisition, 1 bounded 288-image audit, and zero training admissions authorized"
    )


if __name__ == "__main__":
    main()
