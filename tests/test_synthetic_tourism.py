"""Generator intent labels are NOT human gold. All fixtures are fictitious."""
import copy
import csv
import json

import pytest

from traveler_agent.corpus import CorpusError
from traveler_agent.synthetic_tourism import (
    ASPECTS, CATEGORIES, LABELS, audit_records, generate_records, write_corpus,
)
from traveler_agent.synthetic_evaluation import (
    _read_input, group_split, run_synthetic_experiment,
)


@pytest.fixture(scope="module")
def samples():
    return generate_records(seed=42, variants=2)


@pytest.fixture
def written(samples, tmp_path):
    destination = tmp_path / "processed" / "synthetic"
    write_corpus(samples, destination, audit_records(samples), seed=42)
    return destination


def test_generator_reproducible_and_balanced(samples):
    second = generate_records(seed=42, variants=2)
    assert samples == second
    assert len(samples) == len(CATEGORIES) * len(ASPECTS) * len(LABELS) * 2
    assert len({r["review_id"] for r in samples}) == len(samples)
    assert {r["target_label"] for r in samples} == set(LABELS)
    assert all(r["synthetic"] == "YES" for r in samples)
    assert all(r["label_origin"] == "GENERATOR_INTENT_NOT_HUMAN_GOLD" for r in samples)
    assert all(r["generator_model"] == "NO_LLM_LOCAL_RULE_BASED" for r in samples)


def test_quality_and_diverse_cases(samples):
    audit = audit_records(samples)
    assert audit["purpose"] == "INTERNAL_SYNTHETIC_QUALITY_ONLY"
    assert audit["families"] == 56
    assert audit["labels"] == {"negativo": 112, "neutral": 112, "positivo": 112}
    assert audit["unique_text_ratio"] == 1
    assert audit["unique_token_count"] >= 150
    assert "negacion" in audit["hard_cases"]
    assert "opinion_mixta" in audit["hard_cases"]
    assert "ironia_simulada" in audit["hard_cases"]
    assert audit["near_duplicate_candidates_cross_family"] == []


def test_generation_is_not_a_human_dataset(samples):
    row = copy.deepcopy(samples[0])
    row["synthetic"] = "NO"
    with pytest.raises(CorpusError, match="humanas"):
        audit_records([row] + samples[1:])
    row = copy.deepcopy(samples[0])
    row["label_origin"] = "HUMAN_GOLD"
    with pytest.raises(CorpusError, match="humanas"):
        audit_records([row] + samples[1:])


def test_duplicate_or_fingerprint_tamper_rejected(samples):
    with pytest.raises(CorpusError, match="ID repetido|Duplicado"):
        audit_records(samples + [copy.deepcopy(samples[0])])
    bad = copy.deepcopy(samples)
    bad[0]["text_sha256"] = "wrong"
    with pytest.raises(CorpusError, match="Huella"):
        audit_records(bad)


def test_family_split_is_disjoint_and_deterministic(samples):
    first = group_split(samples, seed=42)
    second = group_split(samples, seed=42)
    assert first == second
    assert set(first) == {"train", "validation", "test"}
    seen = set()
    for name, rows in first.items():
        groups = {row["scenario_family"] for row in rows}
        assert not seen & groups
        seen.update(groups)
        assert {row["target_label"] for row in rows} == set(LABELS)
    assert len(seen) == 56


def test_reject_untrusted_manifests_and_modified_text(written):
    with (written / "synthetic_reviews.csv").open("a", encoding="utf-8") as f:
        f.write("tampering\n")
    with pytest.raises(CorpusError, match="procedencia|hash"):
        _read_input(written)


def test_benchmark_end_to_end_exports_three_models_without_gold_claim(written, tmp_path):
    destination = tmp_path / "artifacts" / "synthetic-exp"
    report = run_synthetic_experiment(written, destination, seed=42)
    assert report["report_type"] == "SYNTHETIC_INTERNAL_EVALUATION_ONLY"
    assert report["research_claim"] == "NONE_NO_REAL_TOURIST_GENERALIZATION"
    assert report["not_human_testimonials"] is True
    assert report["label_provenance"] == "GENERATOR_INTENT_NOT_HUMAN_GOLD"
    assert set(report["models"]) == {"dummy", "lr", "svm"}
    assert sum(x["rows"] for x in report["counts"].values()) == 336
    for kind in ("dummy", "lr", "svm"):
        model_report = report["models"][kind]
        assert len(model_report["test"]["confusion_matrix"]) == 3
        assert model_report["test"]["n"] == report["counts"]["test"]["rows"]
        assert "macro_f1" in model_report["validation"]
        assert "accuracy" in model_report["test"]
        assert "test_error_audit" in model_report
        assert (destination / (kind + ".joblib")).exists()
        assert len(report["model_hashes"][kind]) == 64
    payload = json.loads(
        (destination / "synthetic_experiment_report.json").read_text(encoding="utf-8")
    )
    assert "SYNTHETIC_INTERNAL_EVALUATION_ONLY" in json.dumps(payload)
    assert "human" in " ".join(payload["warnings"]).lower()
    with pytest.raises(CorpusError, match="sobrescribir"):
        run_synthetic_experiment(written, destination, seed=42)


def test_benchmark_fails_on_seed_mismatch(written, tmp_path):
    with pytest.raises(CorpusError, match="Semilla"):
        run_synthetic_experiment(written, tmp_path / "artifacts" / "wrong", seed=43)


def test_no_artifacts_written_to_tracked_root(samples, tmp_path):
    with pytest.raises(CorpusError, match="Salida"):
        write_corpus(samples, tmp_path / "public", audit_records(samples), seed=42)
