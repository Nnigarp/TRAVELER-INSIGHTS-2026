"""Corpus contracts and provenance; no fitted model sees the authored challenge."""
from collections import Counter
from copy import deepcopy
import json

import pytest

from traveler_agent.corpus import CorpusError, file_sha
from traveler_agent.synthetic_diversity import (
    CATALOG, PROTOCOL, assemble_strategies, lexical_audit, materialize,
    prepare, read_prepared, summarize,
)


def catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def test_frozen_catalog_roles_intentions_and_unlabeled_examples():
    rows = materialize(catalog())
    stats = summarize(rows)
    assert {k: stats[k]["rows"] for k in stats} == {
        "train": 90, "validation": 36, "challenge": 54, "ambiguity": 9, "illustrative": 4,
    }
    assert stats["challenge"]["labels"] == {"negativo": 18, "neutral": 18, "positivo": 18}
    assert all(r["intended_label"] is None for r in rows if r["role"] in {"ambiguity", "illustrative"})
    assert len({r["family"] for r in rows}) == 73
    assert len({r["text_sha256"] for r in rows}) == len(rows)
    assert rows == materialize(catalog())


def test_structural_and_linguistic_coverage_without_claiming_human_gold():
    stats = summarize(materialize(catalog()))
    required = {"negation", "double_negation", "contrast", "mixed_aspects",
                "positive_words_negative_context", "regionalism", "typo"}
    assert required <= set(stats["train"]["phenomena"])
    assert required <= set(stats["challenge"]["phenomena"])
    assert stats["train"]["length_bands"]["short"] > 0
    assert stats["train"]["length_bands"]["long"] > 0
    assert stats["challenge"]["length_bands"]["long"] > 0
    assert catalog()["provenance"]["human_authors"] is False


@pytest.mark.parametrize("mutation", ["human", "duplicate_text", "duplicate_family", "numeric_label"])
def test_provenance_duplicates_and_labels_fail_closed(mutation):
    data = deepcopy(catalog())
    if mutation == "human":
        data["provenance"]["human_authors"] = True
    elif mutation == "duplicate_text":
        data["families"][1]["cases"][0]["text"] = "  " + data["families"][0]["cases"][0]["text"].upper() + " "
    elif mutation == "duplicate_family":
        data["families"][1]["family"] = data["families"][0]["family"]
    else:
        data["families"][0]["cases"][0]["intended_label"] = "0"
    with pytest.raises(CorpusError):
        materialize(data)


def test_no_transform_of_user_examples_into_training_gold():
    data = catalog()
    data["illustrative_user_examples"][0]["intended_label"] = "positivo"
    with pytest.raises(CorpusError):
        materialize(data)


def test_lexical_screen_has_no_fitted_vocabulary_and_detects_near_copy():
    rows = [
        {"review_id": "a", "role": "train", "text": "Las explicaciones del recorrido permitieron comprender cada detalle de la historia del barrio."},
        {"review_id": "b", "role": "challenge", "text": "Las explicaciones del recorrido permitieron comprender cada detalle de la historia del pueblo."},
    ]
    assert lexical_audit(rows)["candidates"]
    assert lexical_audit(materialize(catalog()))["candidates"] == []


def test_prepare_and_tamper_detection_even_with_forged_record_hash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/processed/v2"
    manifest = prepare(root)
    rows, checked, _ = read_prepared(root)
    assert checked == manifest
    assert len(rows) == 193
    rows[0]["text"] = "Injected synthetic fixture"
    (root / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest["records_sha256"] = file_sha(root / "records.jsonl")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusError):
        read_prepared(root)


def test_refuse_overwrite_and_public_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "data/processed/old"
    old.mkdir(parents=True)
    (old / "evidence.json").write_text("old evidence", encoding="utf-8")
    with pytest.raises(CorpusError):
        prepare(old)
    assert (old / "evidence.json").read_text() == "old evidence"
    with pytest.raises(CorpusError):
        prepare(tmp_path / "published")


def test_matched_budgets_keep_sources_and_never_include_new_held_out_roles():
    rows = materialize(catalog())
    protocol = json.loads(PROTOCOL.read_text())
    strategies, diagnostic = assemble_strategies(rows, protocol)
    assert diagnostic["old_held_out_scores_recomputed"] is False
    assert len(strategies["A_historical_full_sensitivity"]) == 204
    for name, subset in strategies.items():
        if name != "A_historical_full_sensitivity":
            assert len(subset) == 90
            assert Counter(r["intended_label"] for r in subset) == {"negativo": 30, "neutral": 30, "positivo": 30}
        assert {r["role"] for r in subset} == {"train"}
        assert not {r["text_sha256"] for r in subset} & {r["text_sha256"] for r in rows if r["role"] != "train"}
    assert sorted(Counter(r["source"] for r in strategies["C_mixed_matched"]).values()) == [45, 45]
