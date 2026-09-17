import hashlib
import json

import numpy as np
import pytest

from cvcr_moe.data import (
    PackedReader,
    StageManifestPackedReader,
    verify_frozen_manifest,
    verify_stage_admission,
)


def test_reader_produces_shifted_disjoint_batches(tmp_path) -> None:
    data = tmp_path / "pack" / "train-npy"
    data.mkdir(parents=True)
    np.save(data / "train.npy", np.arange(30, dtype=np.uint16))
    reader = PackedReader(data, seed=0, sequence_length=4)
    inputs, targets = reader.batch_for_step(0, rank=0, world_size=2, sequences_per_rank=2)
    other_inputs, _ = reader.batch_for_step(
        0, rank=1, world_size=2, sequences_per_rank=2
    )
    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    assert np.array_equal(targets.numpy()[:, :-1], inputs.numpy()[:, 1:])
    assert set(map(tuple, inputs.tolist())).isdisjoint(set(map(tuple, other_inputs.tolist())))


def test_manifest_gate_fails_closed(tmp_path) -> None:
    data = tmp_path / "pack" / "train-npy"
    data.mkdir(parents=True)
    manifest = data.parent / "manifest.json"
    manifest.write_text(json.dumps({"status": "NOT_ADMITTED", "block_size": 2049}))
    with pytest.raises(ValueError, match="not admitted"):
        verify_frozen_manifest(data)
    manifest.write_text(
        json.dumps({"status": "FROZEN_VERIFIED_PACK", "block_size": 2049})
    )
    assert verify_frozen_manifest(data)["status"] == "FROZEN_VERIFIED_PACK"


def test_microbatch_accumulation_factorization_preserves_record_order(tmp_path) -> None:
    data = tmp_path / "pack" / "train-npy"
    data.mkdir(parents=True)
    np.save(data / "train.npy", np.arange(5 * 80, dtype=np.uint16))
    reader = PackedReader(data, seed=0, sequence_length=4, repeat=True)

    mb8 = np.concatenate(
        [
            reader.batch_for_step(step, 0, 1, 8)[0].numpy()
            for step in range(8)
        ],
        axis=0,
    )
    mb32 = np.concatenate(
        [
            reader.batch_for_step(step, 0, 1, 32)[0].numpy()
            for step in range(2)
        ],
        axis=0,
    )
    assert np.array_equal(mb8, mb32)


def test_stage_manifest_reader_preserves_exact_partial_block_targets(tmp_path) -> None:
    packed = tmp_path / "component.train.bin"
    np.arange(10, dtype="<u2").tofile(packed)
    manifest = {
        "status": "STAGE_MANIFEST_COMPLETE_NOT_ADMITTED",
        "block_tokens": 5,
        "prediction_tokens_per_block": 4,
        "exact_prediction_tokens": 7,
        "stages": [
            {
                "stage": "A",
                "prediction_tokens": 5,
                "sources": [
                    {
                        "source_id": "source",
                        "prediction_tokens": 5,
                        "segments": [
                            {
                                "path": str(packed),
                                "prediction_start": 1,
                                "prediction_count": 5,
                                "first_block": 0,
                                "first_block_prediction_offset": 1,
                            }
                        ],
                    }
                ],
            },
            {
                "stage": "B",
                "prediction_tokens": 2,
                "sources": [
                    {
                        "source_id": "source",
                        "prediction_tokens": 2,
                        "segments": [
                            {
                                "path": str(packed),
                                "prediction_start": 6,
                                "prediction_count": 2,
                                "first_block": 1,
                                "first_block_prediction_offset": 2,
                            }
                        ],
                    }
                ],
            },
        ],
    }
    manifest_path = tmp_path / "stage-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    reader = StageManifestPackedReader(manifest_path, seed=7, sequence_length=4)

    assert reader.total_sequences == 3
    assert reader.total_prediction_tokens == 7
    assert reader.prediction_tokens_before_sequences(0) == 0
    assert reader.prediction_tokens_before_sequences(3) == 7
    assert reader.prediction_tokens_before_sequences(4) == 7
    assert sum(reader.sequence(index)[2] for index in range(2)) == 5
    assert reader.sequence(2)[2] == 2
    exposed = []
    for index in range(reader.total_sequences):
        _, targets, _ = reader.sequence(index)
        exposed.extend(value for value in targets.tolist() if value != -100)
    assert sorted(exposed) == [2, 3, 4, 6, 7, 8, 9]

    _, rank0_targets, rank0_valid = reader.batch_for_step(0, 0, 2, 2)
    _, rank1_targets, rank1_valid = reader.batch_for_step(0, 1, 2, 2)
    assert rank0_valid + rank1_valid == 7
    assert (rank0_targets == -100).any()
    assert (rank1_targets[1] == -100).all()


def test_stage_admission_binds_exact_manifest_and_fails_closed(tmp_path) -> None:
    manifest_path = tmp_path / "stage.json"
    manifest_path.write_text(json.dumps({"exact_prediction_tokens": 7}))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    admission_path = tmp_path / "admission.json"
    admission = {
        "status": "TRAINING_ADMITTED",
        "training_admitted": True,
        "stage_manifest_sha256": digest,
        "exact_prediction_tokens": 7,
        "open_gates": [],
    }
    admission_path.write_text(json.dumps(admission))
    assert verify_stage_admission(manifest_path, admission_path)["training_admitted"]
    admission["stage_manifest_sha256"] = "0" * 64
    admission_path.write_text(json.dumps(admission))
    with pytest.raises(ValueError, match="does not bind"):
        verify_stage_admission(manifest_path, admission_path)
