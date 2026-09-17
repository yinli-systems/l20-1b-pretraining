#!/usr/bin/env python3
"""Differentially prove batched dedup lookups match scalar behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import tempfile

from build_moe7b_full_data import (
    existing_exact_hashes,
    near_hash_exists,
    near_hash_statuses,
    open_state,
    pack_signature,
    signature_bands,
)


def digest(label: str) -> bytes:
    return hashlib.sha256(label.encode()).digest()


def insert(connection, label: str, signature: tuple[int, ...]) -> bytes:
    value = digest(label)
    connection.execute(
        "INSERT INTO documents(hash, component, signature) VALUES (?, ?, ?)",
        (value, "test", pack_signature(signature)),
    )
    for band in signature_bands(signature):
        connection.execute(
            "INSERT INTO near_bands(band, hash) VALUES (?, ?)", (band, value)
        )
    return value


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        connection = open_state(Path(directory) / "state.sqlite")
        reference = tuple(range(16))
        exact = insert(connection, "exact", reference)
        connection.commit()
        near_signature = tuple([*range(15), 999])
        far_signature = tuple(range(100, 116))
        near_digest = digest("near")
        far_digest = digest("far")
        assert existing_exact_hashes(
            connection, [exact, near_digest, far_digest, exact]
        ) == {exact}
        statuses = near_hash_statuses(
            connection,
            [(near_digest, near_signature), (far_digest, far_signature)],
        )
        assert statuses[near_digest] == near_hash_exists(connection, near_signature)
        assert statuses[far_digest] == near_hash_exists(connection, far_signature)
        assert statuses[near_digest] == (True, False)
        assert statuses[far_digest] == (False, False)

        hot_prefix = (700, 701, 702, 703)
        for index in range(256):
            signature = hot_prefix + tuple(index * 100 + offset for offset in range(12))
            insert(connection, f"hot-{index}", signature)
        connection.commit()
        hot_signature = hot_prefix + tuple(range(900, 912))
        hot_digest = digest("hot-candidate")
        hot_status = near_hash_statuses(connection, [(hot_digest, hot_signature)])
        assert hot_status[hot_digest] == near_hash_exists(connection, hot_signature)
        assert hot_status[hot_digest] == (True, True)

        generator = random.Random(20260917)
        references = []
        reference_hashes = []
        for index in range(80):
            signature = tuple(generator.getrandbits(64) for _ in range(16))
            references.append(signature)
            reference_hashes.append(insert(connection, f"random-reference-{index}", signature))
        connection.commit()
        candidates = []
        for index in range(120):
            if index % 3 == 0:
                values = list(references[index % len(references)])
                for offset in range(index % 4):
                    values[-offset - 1] ^= 1
                signature = tuple(values)
            elif index % 3 == 1:
                signature = tuple(generator.getrandbits(64) for _ in range(16))
            else:
                signature = None
            candidates.append((digest(f"random-candidate-{index}"), signature))
        batched = near_hash_statuses(connection, candidates)
        for candidate_digest, signature in candidates:
            assert batched[candidate_digest] == near_hash_exists(connection, signature)
        assert existing_exact_hashes(
            connection,
            [*reference_hashes, *(value for value, _ in candidates)],
        ) == set(reference_hashes)
        connection.close()

    print(
        json.dumps(
            {
                "status": "PASS",
                "cases": [
                    "exact",
                    "near",
                    "far",
                    "hot_bucket",
                    "120_seeded_differential_cases",
                ],
                "claim": "batched indexed lookups match scalar reference semantics",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
