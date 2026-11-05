"""Automated corpus import tests use fabricated data only."""
import csv
import json
from pathlib import Path

import pytest

from traveler_agent.corpus import (
    SOURCE_COLUMNS, REVIEW_COLUMNS, ANNOTATION_COLUMNS, ADJUDICATION_COLUMNS,
    CorpusError, _write_csv, validate_reviews
)
from traveler_agent.corpus_automation import (
    ingest_authorized_csv, privacy_flags, propose_demo_annotations,
    audit_near_duplicates
)


@pytest.fixture
def fixture_root(tmp_path):
    root = tmp_path / "private" / "tourism"
    root.mkdir(parents=True)
    _write_csv(root / "sources.csv", SOURCE_COLUMNS, [{
        "source_id": "approved-demo", "source_type": "open_corpus",
        "source_reference": "LOCAL_SYNTHETIC_TEST_ONLY", "reuse_basis": "license",
        "permission_reference": "FAKE_PERMISSION_FOR_TEST_ONLY",
        "authorization_reviewed": "yes",
    }])
    _write_csv(root / "reviews.csv", REVIEW_COLUMNS, [])
    _write_csv(root / "annotations.csv", ANNOTATION_COLUMNS, [])
    _write_csv(root / "adjudications.csv", ADJUDICATION_COLUMNS, [])
    return root


def _make_import(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["external", "review", "hotel"])
        writer.writeheader()
        writer.writerows(rows)


def test_synthetic_import_skips_pii_and_duplicates(fixture_root, tmp_path):
    source = tmp_path / "sample.csv"
    _make_import(source, [
        {"external": "a", "review": "El hotel fue muy bueno", "hotel": "h1"},
        {"external": "b", "review": "  el HOTEL fue muy bueno  ", "hotel": "h2"},
        {"external": "c", "review": "Mi correo es info@example.org y fue malo", "hotel": "h3"},
        {"external": "d", "review": "El desayuno fue terrible", "hotel": "h4"},
    ])
    outcome = ingest_authorized_csv(source, fixture_root, "approved-demo",
        text_column="review", group_column="hotel", id_column="external", language="es")
    assert outcome["imported"] == 2
    assert outcome["skipped_exact_duplicates_or_ids"] == 1
    assert outcome["skipped_possible_pii"] == 1
    reviews, summary = validate_reviews(fixture_root / "reviews.csv",
                                        fixture_root / "sources.csv")
    assert len(reviews) == 2
    assert not any("example.org" in r["text"] for r in reviews)
    assert all(r["source_id"] == "approved-demo" for r in reviews)
    # Idempotent import: repeated rows do not get inserted twice.
    again = ingest_authorized_csv(source, fixture_root, "approved-demo",
        text_column="review", group_column="hotel", id_column="external", language="es")
    assert again["imported"] == 0


def test_import_rejects_unapproved_source_and_missing_group(fixture_root, tmp_path):
    source = tmp_path / "reviews.csv"
    _make_import(source, [{"external":"1","review":"Muy bien", "hotel":"h1"}])
    with pytest.raises(CorpusError, match="Fuente no registrada"):
        ingest_authorized_csv(source, fixture_root, "not-a-source",
                              text_column="review", group_column="hotel")
    with pytest.raises(CorpusError, match="Faltan columnas"):
        ingest_authorized_csv(source, fixture_root, "approved-demo",
                              text_column="review", group_column="not_there")


def test_private_path_required_and_privacy_flags(tmp_path):
    assert privacy_flags("Contacta test@example.com") == ["possible_email"]
    assert "possible_phone" in privacy_flags("Llama al +57 300 555 1212")
    assert privacy_flags("Reseña sin datos personales evidentes") == []
    root = tmp_path / "public"
    with pytest.raises(CorpusError, match="privada"):
        ingest_authorized_csv(tmp_path / "absent.csv", root, "source",
                              text_column="review", group_column="hotel")


def test_preannotation_never_creates_gold_labels(fixture_root):
    _write_csv(fixture_root / "reviews.csv", REVIEW_COLUMNS, [{
        "review_id":"r01", "text":"El hotel fue excelente pero el precio era alto",
        "source_id":"approved-demo", "group_id":"g01", "language":"es"
    }])
    report = propose_demo_annotations(fixture_root)
    assert report["review_required"] == 1
    with (fixture_root / "preannotations_demo_lr.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["needs_human_review"] == "yes"
    assert "SYNTHETIC_DEMO" in rows[0]["model_provenance"]
    # Gold datasets remain empty; no model prediction may enter annotations.
    with (fixture_root / "annotations.csv").open(encoding="utf-8", newline="") as f:
        assert list(csv.DictReader(f)) == []


def test_near_duplicate_audit_only_reports_ids(fixture_root):
    _write_csv(fixture_root / "reviews.csv", REVIEW_COLUMNS, [
        {"review_id":"r01","text":"La habitación estaba limpia y bonita",
         "source_id":"approved-demo","group_id":"g1","language":"es"},
        {"review_id":"r02","text":"La habitación estaba limpia y bonitta",
         "source_id":"approved-demo","group_id":"g2","language":"es"},
    ])
    result = audit_near_duplicates(fixture_root, threshold=.85)
    assert result["candidates"] >= 1
    content = (fixture_root / "near_duplicate_candidates.json").read_text(encoding="utf-8")
    assert "habitación" not in content
    assert "revision_humana_pendiente" in content
