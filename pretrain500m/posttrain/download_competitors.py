#!/usr/bin/env python3
"""Download pinned public comparison checkpoints and write a hash manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1/baselines/models")
MODELS = {
    "qwen3-0.6b-base": {
        "repo_id": "Qwen/Qwen3-0.6B-Base",
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    },
    "smollm2-360m": {
        "repo_id": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    output: dict[str, object] = {"status": "PASS_PINNED_COMPETITOR_DOWNLOAD", "models": {}}
    for slug, spec in MODELS.items():
        target = ROOT / slug
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_dir=target,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
            max_workers=4,
        )
        files = []
        for path in sorted(target.iterdir()):
            if path.is_file():
                files.append(
                    {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                )
        if not any(item["path"].endswith(".safetensors") for item in files):
            raise RuntimeError(f"{slug} has no safetensors weights")
        output["models"][slug] = {**spec, "files": files}
    temporary = ROOT / "competitor-manifest.json.tmp"
    final = ROOT / "competitor-manifest.json"
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    with temporary.open() as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, final)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
