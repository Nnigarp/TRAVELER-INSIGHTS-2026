"""V4 synthetic corpus: generator intent labels are NOT human gold."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from traveler_agent.corpus import CorpusError
from traveler_agent.dataset_v4_training import prepare_beto_v4_plan, train_classical_v4
from traveler_agent.synthetic_tourism_v4 import (
    LABELS, PROTOCOL, ROLES, audit_records, generate_records, protocol, write_corpus,
)


@pytest.fixture(scope="module")
def rows():
    return generate_records()


def test_protocol_targets_ternary_balance():
    proto = protocol()
    assert proto["target_rows"] == 5001
    assert proto["family_count"] == 1667
    assert proto["rows_per_class"] == 1667
    assert sum(proto["roles"].values()) == 1667
    assert PROTOCOL.is_file()


def test_generator_reproducible_balanced_and_role_disjoint(rows):
    second = generate_records()
    assert rows == second
    assert len(rows) == 5001
    assert {r["intended_label"] for r in rows} == set(LABELS)
    counts = {label: sum(1 for r in rows if r["intended_label"] == label) for label in LABELS}
    assert counts == {label: 1667 for label in LABELS}
    families = {}
    for row in rows:
        families.setdefault(row["scenario_family"], set()).add(row["role"])
    assert len(families) == 1667
    assert all(len(roles) == 1 for roles in families.values())
    role_families = {
        role: {r["scenario_family"] for r in rows if r["role"] == role} for role in ROLES
    }
    assert {k: len(v) for k, v in role_families.items()} == protocol()["roles"]
    for a in ROLES:
        for b in ROLES:
            if a < b:
                assert not (role_families[a] & role_families[b])


def test_audit_accepts_full_corpus(rows):
    audit = audit_records(rows)
    assert audit["records"] == 5001
    assert audit["unique_text_ratio"] == 1.0
    assert audit["hard_holdout_negatives"] == 67
    assert audit["near_duplicate_candidates_cross_family"] == []
    assert audit["unique_token_count"] >= 400


def test_hard_holdout_encodes_failure_pattern(rows):
    hard_neg = [
        r for r in rows
        if r["role"] == "hard_holdout" and r["intended_label"] == "negativo"
    ]
    assert len(hard_neg) == 67
    for row in hard_neg:
        text = row["text"].casefold()
        assert row["template"] == "praise_then_failure"
        assert any(k in text for k in ("transporte", "traslado", "bus", "conductor"))
        assert any(k in text for k in ("reclam", "queja", "soporte", "llamamos"))
        assert any(k in text for k in ("no volver", "descarto", "no contratar"))


def test_rejects_human_gold_attribution(rows):
    bad = copy.deepcopy(rows[0])
    bad["label_origin"] = "HUMAN_GOLD"
    with pytest.raises(CorpusError, match="humano"):
        audit_records([bad] + rows[1:])


def test_write_splits_and_classical_smoke(rows, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed = tmp_path / "data" / "processed" / "synthetic-tourism-v4"
    artifacts = tmp_path / "artifacts" / "synthetic-tourism-v4"
    # recreate package-relative protocol resolution still uses installed package path
    audit = audit_records(rows)
    manifest = write_corpus(rows, processed, audit, seed=int(rows[0]["seed"]))
    assert (processed / "synthetic_reviews_v4.csv").is_file()
    assert (processed / "split_hard_holdout.csv").is_file()
    assert manifest["rows_sha256"]

    report = train_classical_v4(processed, artifacts, model_kind="lr")
    assert report["validation"]["metrics"]["macro_f1"] >= 0.5
    assert (artifacts / "lr.joblib").is_file()

    index = prepare_beto_v4_plan(processed, artifacts)
    assert index["status"] == "PREPARED_NOT_TRAINED"
    assert index["default_max_length"] == 256
    plan = json.loads((artifacts / "beto_v4_training_plan.json").read_text(encoding="utf-8"))
    assert plan["counts"]["train"] == 1120 * 3
    assert "hard_holdout" in plan["splits"]
