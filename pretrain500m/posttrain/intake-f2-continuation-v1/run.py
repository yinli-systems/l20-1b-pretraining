"""Run the bounded F2 continuation intake with the frozen expansion reader."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "intake-expansion-v1"
ROOT = Path("/ssd/scxi253/pretrain500m-20260912-v1")
EXPECTED = {
    "plan.json": "e916977e894b85baeb5fe7f2c3c5167a03e2cecdf309d80d5e323413284564d4",
    "intake.py": "b47a949cf6c96d827a38f878fc1732f933960d776f6a1ad9174d33254e63d0a2",
    "http_ranges.py": "c99ffe8765dd9a487ccf394a7a5262c9ed6a2b1ce655fe32a24c78cef114fffa",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_symlink(name: str, target_name: str) -> None:
    cache = Path("/dev/shm/p529m-intake-expansion-v1")
    target = cache / target_name
    link = cache / name
    if not target.is_file():
        raise FileNotFoundError("verified C++ cache is unavailable: " + target_name)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError("C++ cache symlink identity changed: " + name)
    elif link.exists():
        raise ValueError("unexpected C++ cache file: " + name)
    else:
        link.symlink_to(target)


def main() -> None:
    for name, expected in EXPECTED.items():
        path = HERE / name if name == "plan.json" else PRIOR / name
        if sha256(path) != expected:
            raise ValueError("frozen input changed: " + name)
    checked_symlink("code_cpp_f2cont.parquet", "code_cpp_extra.parquet")
    checked_symlink("code_cpp_f2cont.receipt.json", "code_cpp_extra.receipt.json")
    sys.path.insert(0, str(PRIOR))
    spec = importlib.util.spec_from_file_location("p529m_f2_frozen_intake", PRIOR / "intake.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen intake")
    intake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(intake)
    intake.SOURCE = HERE
    intake.OUTPUT = ROOT / "data/intake-f2-continuation-v1"
    intake.OUTER_WORKERS = 3
    intake.CODE_WORKERS = 32
    intake.main()


if __name__ == "__main__":
    main()
