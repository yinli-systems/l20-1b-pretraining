"""Pure helpers for the one-time Stage-D IID/OOD confirmation."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ARMS = ("A_spatial_49", "F_matched_196")
SPLITS = ("iid_test", "ood_binding")
CONDITIONS = ("true_image", "random_image", "no_image")
T_CRITICAL_975 = {
    1: 12.706204736432095,
    2: 4.302652729696142,
    3: 3.182446305284263,
    4: 2.7764451051977987,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def confirmation_execution_order(seeds: list[int]) -> list[dict[str, str | int]]:
    """Balance arm and split order while evaluating every cell exactly once."""
    order: list[dict[str, str | int]] = []
    even = (
        ("A_spatial_49", "iid_test"),
        ("F_matched_196", "iid_test"),
        ("F_matched_196", "ood_binding"),
        ("A_spatial_49", "ood_binding"),
    )
    odd = (
        ("F_matched_196", "ood_binding"),
        ("A_spatial_49", "ood_binding"),
        ("A_spatial_49", "iid_test"),
        ("F_matched_196", "iid_test"),
    )
    for index, seed in enumerate(seeds):
        for arm, split in even if index % 2 == 0 else odd:
            order.append({"arm": arm, "seed": seed, "split": split})
    return order


def family_joint(row: dict, condition: str = "true_image") -> float:
    expected = row["expected"]
    predicted = row["predictions"][condition]
    return float(
        predicted["base"] == expected["base"]
        and predicted["edited"] == expected["edited"]
    )


def ordinary_primary(row: dict, condition: str = "true_image") -> float:
    expected = row["expected"]
    predicted = row["predictions"][condition]
    return (
        float(predicted["base"] == expected["base"])
        + float(predicted["edited"] == expected["edited"])
    ) / 2.0


def paired_t_interval(values_pp: list[float]) -> dict[str, float | int]:
    if len(values_pp) < 2:
        raise ValueError("paired t interval requires at least two seeds")
    values = np.asarray(values_pp, dtype=np.float64)
    degrees = len(values_pp) - 1
    if degrees not in T_CRITICAL_975:
        raise ValueError("paired t interval is frozen only for two through five seeds")
    estimate = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    half_width = T_CRITICAL_975[degrees] * standard_deviation / math.sqrt(len(values_pp))
    return {
        "estimate_pp": estimate,
        "lower_95_ci_pp": estimate - half_width,
        "upper_95_ci_pp": estimate + half_width,
        "seed_pairs": len(values_pp),
        "sample_standard_deviation_pp": standard_deviation,
        "method": "paired_t_across_training_seeds",
    }


def crossed_bootstrap_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    """Resample paired seeds and aligned scene families independently."""
    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("crossed bootstrap requires a seed-by-family matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("crossed bootstrap input must be finite")
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    seed_count, family_count = matrix.shape
    for index in range(resamples):
        selected_seeds = rng.integers(0, seed_count, size=seed_count)
        selected_families = rng.integers(0, family_count, size=family_count)
        samples[index] = matrix[np.ix_(selected_seeds, selected_families)].mean()
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "estimate_pp": 100.0 * float(matrix.mean()),
        "lower_95_ci_pp": 100.0 * float(lower),
        "upper_95_ci_pp": 100.0 * float(upper),
        "seed_pairs": seed_count,
        "scene_families": family_count,
        "resamples": resamples,
        "bootstrap_seed": seed,
        "method": "crossed_bootstrap_paired_seeds_and_aligned_scene_families",
    }
