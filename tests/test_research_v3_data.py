from collections import Counter
from copy import deepcopy
import json

import pytest

from traveler_agent.corpus import CorpusError, file_sha
from traveler_agent import research_v3_data as data


def catalog():
    return json.loads(data.CATALOG.read_text(encoding="utf-8"))


def test_factorial_preserves_budgets_class_balance_and_related_groups():
    rows = data.materialize(catalog())
    assert len(rows) == 335
    counts = data.coverage(rows)
    assert {k: v["rows"] for k, v in counts.items()} == {
        "train": 180, "tune": 30, "calibration": 30, "policy_dev": 30,
        "assessment": 45, "ambiguity_policy": 10, "ambiguity_assessment": 10}
    arms = data.training_arms(rows)
    for name, arm in arms.items():
        n = int(name.split("_")[1])
        assert len(arm) == n
        assert set(Counter(r["intended_label"] for r in arm).values()) == {n // 3}
        assert len({r["family"] for r in arm}) == (10 if name.startswith("narrow") else 20)
        for field in ("domain", "aspect"):
            assert len(set(Counter(r[field] for r in arm).values())) == 1
    for width in ("narrow", "broad"):
        assert {r["review_id"] for r in arms[f"{width}_60"]} < {r["review_id"] for r in arms[f"{width}_120"]}
    for component in {r["semantic_component"] for r in rows}:
        assert len({r["role"] for r in rows if r["semantic_component"] == component}) == 1
    assert len({r["text_sha256"] for r in rows}) == 335
    assert rows == data.materialize(catalog())


def test_ablations_keep_size_and_classes_without_held_out_rows():
    rows = data.materialize(catalog())
    for field, values in (("domain", data.DOMAINS), ("aspect", data.ASPECTS)):
        for value in values:
            arm = data.ablation_rows(rows, field, value)
            assert len(arm) == 120 and len({r["review_id"] for r in arm}) == 120
            assert all(r["role"] == "train" and r[field] != value for r in arm)
            assert set(Counter(r["intended_label"] for r in arm).values()) == {40}


@pytest.mark.parametrize("mutation", ["human", "duplicate", "component", "label", "variant", "ambiguous_gold"])
def test_rejects_invalid_provenance_leakage_or_annotations(mutation):
    c = catalog()
    if mutation == "human": c["provenance"]["human_authors"] = True
    elif mutation == "duplicate": c["families"][1]["cases"][0]["text"] = c["families"][0]["cases"][0]["text"].upper()
    elif mutation == "component":
        other = next(f for f in c["families"] if f["role"] == "assessment")
        other["semantic_component"] = c["families"][0]["semantic_component"]
    elif mutation == "label": c["families"][0]["cases"][0]["intended_label"] = 0
    elif mutation == "variant": c["families"][0]["cases"][0]["variant"] = 99
    else: c["ambiguity"][0]["intended_label"] = "neutral"
    with pytest.raises(CorpusError): data.materialize(c)


def test_v2_texts_are_denylisted_not_reused():
    c = catalog()
    old = json.loads(data.V2_CATALOG.read_text(encoding="utf-8"))
    family = next(f for f in old["families"] if f["role"] == "train")
    c["families"][0]["cases"][0]["text"] = family["cases"][0]["text"]
    with pytest.raises(CorpusError, match="V2"):
        data.materialize(c)


def test_pre_generation_coverage_ignores_reserved_cases(tmp_path, monkeypatch):
    path = tmp_path / "v2-fixture.json"
    path.write_text(json.dumps({"families": [
        {"role": "train", "cases": [{"intended_label": "neutral", "phenomena": ["negation"]}]},
        {"role": "validation", "cases": []}, {"role": "challenge", "cases": None}]}))
    monkeypatch.setattr(data, "V2_CATALOG", path)
    result = data.v2_coverage()
    assert result["roles_inspected"]["train"]["rows"] == 1
    assert "challenge" not in result["roles_inspected"]
    assert result["held_out_rows_scored_or_used_for_selection"] is False


def test_prepare_rejects_forged_hash_rows_counts_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/processed/v3"
    data.prepare(root)
    data.read_prepared(root)
    with pytest.raises(CorpusError): data.prepare(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coverage"]["assessment"]["rows"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CorpusError, match="Recuentos"):
        data.read_prepared(root)
    with pytest.raises(CorpusError): data.prepare(tmp_path / "public")


def test_near_duplicates_block_different_roles_and_train_audit_is_guarded():
    rows = [{"role": role, "text": text, "review_id": str(i)} for i, (role, text) in enumerate([
        ("train", "La entrada estaba cerrada y nadie pudo explicar cuándo abriría la sala de visitas."),
        ("policy_dev", "La entrada estaba cerrada y nadie pudo explicar cuándo abriría la sala de visitas hoy.")])]
    assert data.duplicate_screen(rows)["candidate_count"] == 1
    with pytest.raises(CorpusError, match="TRAIN"):
        data.shortcut_audit(rows)
