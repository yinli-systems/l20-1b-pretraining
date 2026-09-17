"""Deterministic, disjoint reader for the frozen 2,049-token NumPy pack."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch


class PackedReader:
    def __init__(
        self,
        directory: Path,
        seed: int,
        sequence_length: int = 2_048,
        repeat: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self.block_size = sequence_length + 1
        self.seed = seed
        self.repeat = repeat
        paths = sorted(directory.glob("*.npy"))
        if not paths:
            raise FileNotFoundError(directory)
        random.Random(seed).shuffle(paths)
        self.arrays: list[np.ndarray] = []
        self.counts: list[int] = []
        for path in paths:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.uint16 or array.ndim != 1:
                raise ValueError(f"invalid packed shard {path}: {array.dtype}/{array.shape}")
            if array.size % self.block_size:
                raise ValueError(
                    f"invalid packed shard {path}: {array.size} is not divisible by {self.block_size}"
                )
            count = array.size // self.block_size
            if count:
                self.arrays.append(array)
                self.counts.append(count)
        self.ends = np.cumsum(self.counts).tolist()
        self.total_sequences = sum(self.counts)
        self.unique_prediction_tokens = self.total_sequences * self.sequence_length

    def _physical_index(self, index: int) -> int:
        if index < 0:
            raise IndexError(index)
        epoch, position = divmod(index, self.total_sequences)
        if epoch and not self.repeat:
            raise IndexError(index)
        offset = (self.seed + epoch * 0x9E3779B1) % self.total_sequences
        return (position + offset) % self.total_sequences

    def sequence(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.total_sequences:
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        prior = 0 if shard == 0 else self.ends[shard - 1]
        offset = (index - prior) * self.block_size
        values = np.asarray(
            self.arrays[shard][offset : offset + self.block_size], dtype=np.int64
        )
        return torch.from_numpy(values.copy())

    def batch_for_step(
        self,
        step: int,
        rank: int,
        world_size: int,
        sequences_per_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = step * world_size * sequences_per_rank + rank * sequences_per_rank
        if not self.repeat and base + sequences_per_rank > self.total_sequences:
            raise StopIteration
        values = torch.stack(
            [self.sequence(self._physical_index(base + i)) for i in range(sequences_per_rank)]
        )
        return values[:, :-1], values[:, 1:]


class StageManifestPackedReader:
    """Deterministic reader for exact prediction-token slices in a stage manifest.

    A segment may start or end inside a 2,048-prediction physical block.  The
    full 2,049-token block remains available as context while targets outside
    the segment are set to ``ignore_index``.  Thus every admitted prediction
    target is exposed exactly once without rounding stage/source quotas.
    """

    def __init__(
        self,
        manifest_path: Path,
        seed: int,
        sequence_length: int = 2_048,
        pad_id: int = 1,
        ignore_index: int = -100,
    ) -> None:
        self.manifest_path = manifest_path
        self.seed = seed
        self.sequence_length = sequence_length
        self.block_size = sequence_length + 1
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") not in {
            "STAGE_MANIFEST_COMPLETE_NOT_ADMITTED",
            "FROZEN_VERIFIED_ADMITTED_STAGE_MANIFEST",
        }:
            raise ValueError(f"unexpected stage-manifest status: {manifest.get('status')!r}")
        if int(manifest.get("block_tokens", 0)) != self.block_size:
            raise ValueError("stage-manifest block size does not match the training sequence")
        if int(manifest.get("prediction_tokens_per_block", 0)) != sequence_length:
            raise ValueError("stage-manifest prediction block does not match the training sequence")

        self.arrays: dict[Path, np.memmap] = {}
        self.stages: list[dict] = []
        self.partial_sequence_indices: list[int] = []
        self.partial_sequence_deficits: list[int] = []
        stage_ends = []
        prediction_total = 0
        sequence_total = 0
        for stage_index, stage in enumerate(manifest["stages"]):
            segments = []
            segment_ends = []
            stage_predictions = 0
            stage_sequences = 0
            for source in stage["sources"]:
                source_predictions = 0
                for record in source["segments"]:
                    path = Path(record["path"])
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    prediction_start = int(record["prediction_start"])
                    prediction_count = int(record["prediction_count"])
                    if prediction_count <= 0:
                        raise ValueError("stage segment prediction count must be positive")
                    first_block = prediction_start // sequence_length
                    first_offset = prediction_start % sequence_length
                    if int(record["first_block"]) != first_block:
                        raise ValueError("stage segment first_block is inconsistent")
                    if int(record["first_block_prediction_offset"]) != first_offset:
                        raise ValueError("stage segment first-block offset is inconsistent")
                    byte_size = path.stat().st_size
                    if byte_size % (self.block_size * 2):
                        raise ValueError(f"invalid raw packed shard size: {path}")
                    physical_blocks = byte_size // (self.block_size * 2)
                    final_prediction = prediction_start + prediction_count
                    final_block_exclusive = math.ceil(final_prediction / sequence_length)
                    if final_block_exclusive > physical_blocks:
                        raise ValueError(f"stage segment exceeds raw packed shard: {path}")
                    segment_sequences = final_block_exclusive - first_block
                    stage_sequences += segment_sequences
                    segment_ends.append(stage_sequences)
                    segments.append(
                        {
                            "path": path,
                            "prediction_start": prediction_start,
                            "prediction_count": prediction_count,
                            "first_block": first_block,
                            "sequence_count": segment_sequences,
                        }
                    )
                    source_predictions += prediction_count
                if source_predictions != int(source["prediction_tokens"]):
                    raise ValueError("stage source segments do not sum to their quota")
                stage_predictions += source_predictions
            if stage_predictions != int(stage["prediction_tokens"]):
                raise ValueError("stage sources do not sum to the stage quota")
            multiplier, offset = self._affine_permutation(
                stage_sequences, seed, stage_index
            )
            inverse = 0 if stage_sequences == 1 else pow(multiplier, -1, stage_sequences)
            segment_start = 0
            for segment in segments:
                candidates = {0, segment["sequence_count"] - 1}
                for sequence_in_segment in candidates:
                    physical_block = segment["first_block"] + sequence_in_segment
                    block_prediction_start = physical_block * sequence_length
                    valid_start = max(
                        segment["prediction_start"], block_prediction_start
                    ) - block_prediction_start
                    valid_end = min(
                        segment["prediction_start"] + segment["prediction_count"],
                        block_prediction_start + sequence_length,
                    ) - block_prediction_start
                    valid = valid_end - valid_start
                    if valid < sequence_length:
                        shuffled = segment_start + sequence_in_segment
                        logical = (
                            0
                            if stage_sequences == 1
                            else (inverse * (shuffled - offset)) % stage_sequences
                        )
                        self.partial_sequence_indices.append(sequence_total + logical)
                        self.partial_sequence_deficits.append(sequence_length - valid)
                segment_start += segment["sequence_count"]
            sequence_total += stage_sequences
            stage_ends.append(sequence_total)
            prediction_total += stage_predictions
            self.stages.append(
                {
                    "segments": segments,
                    "segment_ends": segment_ends,
                    "sequence_count": stage_sequences,
                    "prediction_tokens": stage_predictions,
                    "permutation_multiplier": multiplier,
                    "permutation_offset": offset,
                }
            )
        if prediction_total != int(manifest["exact_prediction_tokens"]):
            raise ValueError("stage quotas do not equal exact_prediction_tokens")
        self.stage_ends = stage_ends
        self.total_sequences = sequence_total
        self.total_prediction_tokens = prediction_total
        self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ordered_partials = sorted(
            zip(self.partial_sequence_indices, self.partial_sequence_deficits)
        )
        self.partial_sequence_indices = [item[0] for item in ordered_partials]
        self.partial_sequence_deficits = [item[1] for item in ordered_partials]
        if (
            self.total_sequences * self.sequence_length
            - sum(self.partial_sequence_deficits)
            != self.total_prediction_tokens
        ):
            raise ValueError("stage-manifest partial-block accounting is inconsistent")

    @staticmethod
    def _affine_permutation(size: int, seed: int, stage_index: int) -> tuple[int, int]:
        if size < 1:
            raise ValueError("every stage must contain at least one sequence")
        digest = hashlib.sha256(f"{seed}:stage:{stage_index}".encode()).digest()
        multiplier = int.from_bytes(digest[:8], "big") % size
        if multiplier == 0:
            multiplier = 1
        while math.gcd(multiplier, size) != 1:
            multiplier = (multiplier + 1) % size or 1
        offset = int.from_bytes(digest[8:16], "big") % size
        return multiplier, offset

    def _array(self, path: Path) -> np.memmap:
        if path not in self.arrays:
            self.arrays[path] = np.memmap(path, dtype="<u2", mode="r")
        return self.arrays[path]

    def sequence(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if not 0 <= index < self.total_sequences:
            raise IndexError(index)
        stage_index = bisect.bisect_right(self.stage_ends, index)
        stage_start = 0 if stage_index == 0 else self.stage_ends[stage_index - 1]
        stage = self.stages[stage_index]
        logical = index - stage_start
        shuffled = (
            stage["permutation_multiplier"] * logical
            + stage["permutation_offset"]
        ) % stage["sequence_count"]
        segment_index = bisect.bisect_right(stage["segment_ends"], shuffled)
        segment_start = 0 if segment_index == 0 else stage["segment_ends"][segment_index - 1]
        segment = stage["segments"][segment_index]
        sequence_in_segment = shuffled - segment_start
        physical_block = segment["first_block"] + sequence_in_segment
        raw_start = physical_block * self.block_size
        raw = np.asarray(
            self._array(segment["path"])[raw_start : raw_start + self.block_size],
            dtype=np.int64,
        )
        inputs = torch.from_numpy(raw[:-1].copy())
        targets = torch.from_numpy(raw[1:].copy())
        block_prediction_start = physical_block * self.sequence_length
        valid_start = max(
            segment["prediction_start"], block_prediction_start
        ) - block_prediction_start
        valid_end = min(
            segment["prediction_start"] + segment["prediction_count"],
            block_prediction_start + self.sequence_length,
        ) - block_prediction_start
        if not 0 <= valid_start < valid_end <= self.sequence_length:
            raise RuntimeError("invalid stage segment mask interval")
        targets[:valid_start] = self.ignore_index
        targets[valid_end:] = self.ignore_index
        return inputs, targets, valid_end - valid_start

    def batch_for_step(
        self,
        step: int,
        rank: int,
        world_size: int,
        sequences_per_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        base = step * world_size * sequences_per_rank + rank * sequences_per_rank
        inputs = []
        targets = []
        valid_predictions = 0
        for offset in range(sequences_per_rank):
            index = base + offset
            if index < self.total_sequences:
                input_ids, target_ids, valid = self.sequence(index)
            else:
                input_ids = torch.full(
                    (self.sequence_length,), self.pad_id, dtype=torch.int64
                )
                target_ids = torch.full(
                    (self.sequence_length,), self.ignore_index, dtype=torch.int64
                )
                valid = 0
            inputs.append(input_ids)
            targets.append(target_ids)
            valid_predictions += valid
        return torch.stack(inputs), torch.stack(targets), valid_predictions

    def prediction_tokens_before_sequences(self, sequence_count: int) -> int:
        bounded = min(max(sequence_count, 0), self.total_sequences)
        partial_count = bisect.bisect_left(self.partial_sequence_indices, bounded)
        deficit = sum(self.partial_sequence_deficits[:partial_count])
        return bounded * self.sequence_length - deficit


def verify_frozen_manifest(data_directory: Path) -> dict:
    manifest_path = data_directory.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing data manifest: {manifest_path}")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "FROZEN_VERIFIED_PACK":
        raise ValueError(
            f"data pack is not admitted: status={manifest.get('status')!r}"
        )
    if manifest.get("block_size") != 2_049:
        raise ValueError(f"unexpected block size: {manifest.get('block_size')!r}")
    return manifest


def verify_stage_admission(stage_manifest_path: Path, admission_path: Path) -> dict:
    """Fail closed unless an admission receipt binds the exact stage manifest."""
    manifest_bytes = stage_manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    admission = json.loads(admission_path.read_text())
    expected_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if admission.get("status") != "TRAINING_ADMITTED" or admission.get("training_admitted") is not True:
        raise ValueError("formal stage data is not admitted")
    if admission.get("stage_manifest_sha256") != expected_hash:
        raise ValueError("data-admission receipt does not bind the stage manifest")
    if admission.get("open_gates") not in ([], None):
        raise ValueError("data-admission receipt retains open gates")
    if int(admission.get("exact_prediction_tokens", -1)) != int(
        manifest.get("exact_prediction_tokens", -2)
    ):
        raise ValueError("data-admission token count does not match the stage manifest")
    return admission
