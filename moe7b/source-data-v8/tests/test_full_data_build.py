import importlib.util
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_moe7b_full_data", ROOT / "scripts" / "build_moe7b_full_data.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_contract_exactly_matches_frozen_source_totals():
    build = json.loads((ROOT / "data" / "moe7b_150b_build_v1.json").read_text())
    plan = json.loads((ROOT / "data" / "moe7b_150b_mixture_v1.json").read_text())
    MODULE.validate_config(build, plan)


def test_minhash_detects_identical_and_reordered_text_is_not_exact():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    identical = MODULE.minhash_signature(text)
    reordered = MODULE.minhash_signature("mu lambda kappa iota theta eta zeta epsilon delta gamma beta alpha")
    assert identical == MODULE.minhash_signature(text)
    assert identical != reordered
    assert MODULE.signature_similarity(identical, MODULE.unpack_signature(MODULE.pack_signature(identical))) == 1.0


def test_web_pii_redaction_only_replaces_valid_ipv4_and_email():
    text, changed = MODULE.redact_web_pii("mail a@example.com ip 192.168.1.1 version 999.999.999.999")
    assert changed
    assert "a@example.com" not in text
    assert "192.168.1.1" not in text
    assert "999.999.999.999" in text


def test_component_file_selection_is_deterministic_and_excludes_test():
    component = {"component_id": "lang", "path_prefix": "data/cmn_Hani/train/"}
    lock = {
        "files": [
            {"path": "data/cmn_Hani/train/b.parquet"},
            {"path": "data/cmn_Hani/test/a.parquet"},
            {"path": "data/cmn_Hani/train/a.parquet"},
        ]
    }
    first = MODULE.input_files(component, lock, 7)
    second = MODULE.input_files(component, lock, 7)
    assert first == second
    assert {item["path"] for item in first} == {
        "data/cmn_Hani/train/a.parquet",
        "data/cmn_Hani/train/b.parquet",
    }


def test_ordered_pipeline_analysis_and_tokenization_preserves_input_order():
    class CleanIndex:
        @staticmethod
        def contaminated(_text: str) -> bool:
            return False

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    MODULE.WORKER_CONTAMINATION = CleanIndex()
    MODULE.WORKER_TOKENIZER = tokenizer
    result = MODULE.analyze_and_tokenize_batch(["alpha beta", "beta alpha"])

    assert [item[3] for item in result] == [[1, 2], [2, 1]]
    assert [item[0] for item in result] == [
        MODULE.analyze_text("alpha beta")[0],
        MODULE.analyze_text("beta alpha")[0],
    ]


def test_batched_dedup_queries_match_scalar_reference(tmp_path):
    connection = MODULE.open_state(tmp_path / "state.sqlite")
    signature = tuple(range(16))
    stored_hash = b"s" * 32
    connection.execute(
        "INSERT INTO documents(hash, component, signature) VALUES (?, ?, ?)",
        (stored_hash, "test", MODULE.pack_signature(signature)),
    )
    for band in MODULE.signature_bands(signature):
        connection.execute(
            "INSERT INTO near_bands(band, hash) VALUES (?, ?)",
            (band, stored_hash),
        )
    connection.commit()
    new_hash = b"n" * 32
    near_signature = tuple([*range(15), 999])
    assert MODULE.existing_exact_hashes(connection, [stored_hash, new_hash]) == {
        stored_hash
    }
    observed = MODULE.near_hash_statuses(connection, [(new_hash, near_signature)])
    assert observed[new_hash] == MODULE.near_hash_exists(connection, near_signature)
    assert observed[new_hash] == (True, False)
    connection.close()


def test_receipt_reconciliation_bulk_inserts_and_is_idempotent(tmp_path):
    connection = MODULE.open_state(tmp_path / "state.sqlite")
    dedup_path = tmp_path / "records.bin"
    with dedup_path.open("wb") as handle:
        for index in range(5000):
            digest = index.to_bytes(32, "big")
            signature = tuple(index * 16 + offset for offset in range(16))
            handle.write(digest)
            handle.write(MODULE.pack_signature(signature))
    train_path = tmp_path / "train.bin"
    validation_path = tmp_path / "validation.bin"
    train_path.write_bytes(b"train")
    validation_path.write_bytes(b"validation")
    artifacts = {}
    for name, path in (
        ("train", train_path),
        ("validation", validation_path),
        ("dedup_records", dedup_path),
    ):
        artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": MODULE.sha256_file(path),
        }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "component_id": "test",
                "input": {"repository_path": "data/test", "sha256": "a" * 64},
                "artifacts": artifacts,
                "train_blocks": 7,
                "validation_blocks": 1,
            }
        )
    )
    MODULE.commit_receipt(connection, receipt_path)
    MODULE.commit_receipt(connection, receipt_path)
    assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 5000
    assert connection.execute("SELECT COUNT(*) FROM near_bands").fetchone()[0] == 20000
    assert connection.execute("SELECT COUNT(*) FROM completed_files").fetchone()[0] == 1
    connection.close()
