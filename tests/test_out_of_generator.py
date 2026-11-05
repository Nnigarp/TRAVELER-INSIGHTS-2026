"""Sealed synthetic challenge: no real text, no human labels, no new model tuning."""
import copy
import csv
import json

import pytest

from traveler_agent.corpus import CorpusError
from traveler_agent.out_of_generator_challenge import (
    FIELDS, audit_challenge, generate_challenge, save_challenge
)
from traveler_agent.evaluate_out_of_generator import (
    BASELINE_CSV_SHA, group_split, load_frozen_baseline, load_challenge,
    near_duplicate_audit,
)
from traveler_agent.synthetic_tourism import generate_records, audit_records, write_corpus


@pytest.fixture(scope="module")
def challenge():
    return generate_challenge(seed=913)


def test_challenge_is_reproducible_new_inventory(challenge):
    assert challenge == generate_challenge(seed=913)
    assert len(challenge) == 1260
    assert len({r["nucleus_id"] for r in challenge}) == 180
    assert all(r["label_origin"] == "AUTHORED_INTENT_SYNTHETIC_NOT_HUMAN_GOLD" for r in challenge)
    assert {r["domain"] for r in challenge} == {
        "hotel", "restaurante", "playa", "museo", "transporte", "experiencia"
    }


def test_challenge_balanced_and_groups_are_explicit(challenge):
    audit = audit_challenge(challenge)
    assert audit["rows"] == 1260
    assert audit["independent_nuclei"] == 180
    assert audit["labels"] == {"negativo": 420, "neutral": 420, "positivo": 420}
    assert audit["exact_overlap_count"] == 0
    assert len({r["text_sha256"] for r in challenge}) == 1260
    assert len({r["review_id"] for r in challenge}) == 1260


def test_all_nucleus_variants_are_grouped(challenge):
    groups = {}
    for r in challenge:
        groups.setdefault(r["nucleus_id"], []).append(r)
    assert len(groups) == 180
    assert all(len(rows) == 7 for rows in groups.values())
    assert all(len({x["intended_label"] for x in rows}) == 1 for rows in groups.values())


def test_challenge_not_human_gold_and_tamper_detected(challenge):
    altered = copy.deepcopy(challenge)
    altered[0]["label_origin"] = "HUMAN_GOLD"
    with pytest.raises(CorpusError,match="humanas"):
        audit_challenge(altered)
    altered = copy.deepcopy(challenge)
    altered[0]["text"] += " Cambio malicioso."
    with pytest.raises(CorpusError,match="alterado"):
        audit_challenge(altered)


def test_exact_contamination_is_blocked(challenge):
    with pytest.raises(CorpusError,match="Contaminación"):
        audit_challenge(challenge,original_texts=[challenge[0]["text"]])


def test_sealed_original_csv_hash_matches_original_seed(tmp_path):
    original = generate_records(seed=42, variants=2, generated_date="2026-11-05")
    root = tmp_path / "processed" / "frozen-original"
    manifest = write_corpus(original,root,audit_records(original),seed=42)
    assert manifest["rows_sha256"] == BASELINE_CSV_SHA
    # No need to rerun original test or fit models for this assertion.


def test_save_challenge_is_immutable_and_manifest_hashed(challenge,tmp_path):
    out=tmp_path/"processed"/"challenge"
    result=save_challenge(challenge,audit_challenge(challenge),out,seed=913)
    assert len(result["data_sha256"])==64
    with (out/"challenge_synthetic.csv").open(encoding="utf-8",newline="") as f:
        assert tuple(csv.DictReader(f).fieldnames)==FIELDS
    with pytest.raises(CorpusError,match="sobrescribir"):
        save_challenge(challenge,audit_challenge(challenge),out,seed=913)


def test_near_duplicate_screen_uses_only_text_and_is_explicit(challenge):
    original=generate_records(seed=42, variants=2)
    findings=near_duplicate_audit([x["text"] for x in original],
                                  [x["text"] for x in challenge])
    assert findings["method"]=="char_tfidf_cosine_3to5_cross_corpus"
    assert 0 <= findings["highest_similarity"] <= 1.00000001
    assert findings["candidate_count"] >= 0
    assert "cannot prove" in findings["note"]


def test_missing_original_report_blocks_evaluation(tmp_path):
    with pytest.raises(CorpusError,match="archivos originales"):
        load_frozen_baseline(tmp_path/"processed"/"unavailable")
