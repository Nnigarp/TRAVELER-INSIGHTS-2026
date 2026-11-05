"""Synthetic fixtures test plumbing ONLY: no tourism performance claims."""
import csv
import json
from pathlib import Path

import pytest
from traveler_agent.corpus import (
    CorpusError, REVIEW_COLUMNS, SOURCE_COLUMNS, ANNOTATION_COLUMNS,
    ADJUDICATION_COLUMNS, _write_csv, validate_sources, validate_reviews,
    validate_annotations, record_annotation, record_adjudication,
    validate_adjudications, text_sha, parse_aspects,
)
from traveler_agent.corpus_cli import run
from traveler_agent.corpus_splits import prepare_splits


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "private" / "corpus"
    root.mkdir(parents=True)
    sources = root / "sources.csv"
    reviews = root / "reviews.csv"
    annotations = root / "annotations.csv"
    adjudications = root / "adjudications.csv"
    _write_csv(sources, SOURCE_COLUMNS, [
        {"source_id": "voluntary2026", "source_type": "voluntary",
         "source_reference": "local_form_v1", "reuse_basis": "consent",
         "permission_reference": "private_consent_record", "authorization_reviewed": "yes"}
    ])
    _write_csv(reviews, REVIEW_COLUMNS, [])
    _write_csv(annotations, ANNOTATION_COLUMNS, [])
    _write_csv(adjudications, ADJUDICATION_COLUMNS, [])
    return root, sources, reviews, annotations, adjudications


def _synthetic_records(n=90):
    labels = ("positivo", "neutral", "negativo")
    records = []
    labels_by_id = {}
    for i in range(n):
        rid = f"r{i:03d}"
        records.append({
            "review_id": rid, "text": f"Texto artificial de prueba {i}",
            "source_id": "voluntary2026", "group_id": f"g{i:03d}", "language": "es"
        })
        labels_by_id[rid] = labels[i % 3]
    return records, labels_by_id


def test_empty_source_rejected(corpus):
    _, src, *_ = corpus
    _write_csv(src, SOURCE_COLUMNS, [])
    with pytest.raises(CorpusError, match="No hay fuentes"):
        validate_sources(src)


def test_source_requires_review_and_permission(corpus):
    _, src, *_ = corpus
    rows, _ = validate_sources(src)
    rows[0]["authorization_reviewed"] = "no"
    _write_csv(src, SOURCE_COLUMNS, rows)
    with pytest.raises(CorpusError, match="autorización"):
        validate_sources(src)


def test_reviews_require_source_and_pseudonym(corpus):
    _, src, reviews, *_ = corpus
    _write_csv(reviews, REVIEW_COLUMNS, [
        {"review_id": "r01", "text": "texto de ejemplo", "source_id": "unknown",
         "group_id": "g1", "language": "es"}
    ])
    with pytest.raises(CorpusError, match="fuente autorizada"):
        validate_reviews(reviews, src)


def test_duplicate_profile_uses_normalization_without_overwriting_text(corpus):
    _, src, reviews, *_ = corpus
    rows = [
        {"review_id": "r01", "text": "  HOLA   turismo 😃", "source_id": "voluntary2026",
         "group_id": "g1", "language": "es"},
        {"review_id": "r02", "text": "hola turismo 😃", "source_id": "voluntary2026",
         "group_id": "g2", "language": "es"},
    ]
    _write_csv(reviews, REVIEW_COLUMNS, rows)
    fetched, profile = validate_reviews(reviews, src)
    assert fetched[0]["text"] == "  HOLA   turismo 😃"
    assert profile["normalized_exact_duplicate_extra_rows"] == 1
    assert text_sha(rows[0]["text"]) == text_sha(rows[1]["text"])


def test_annotation_is_separate_from_adjudication(corpus):
    _, src, reviews, ann, adj = corpus
    _write_csv(reviews, REVIEW_COLUMNS, [{
        "review_id": "r01", "text": "Servicio bueno y espera larga",
        "source_id": "voluntary2026", "group_id": "g1", "language": "es",
    }])
    ids = {"r01"}
    record_annotation(ann, ids, review_id="r01", annotator_id="a1", sentiment="positivo",
                      aspects={"servicio": "positivo", "espera": "negativo"})
    record_annotation(ann, ids, review_id="r01", annotator_id="a2", sentiment="negativo",
                      aspects={"servicio": "positivo", "espera": "negativo"})
    annotations, stats = validate_annotations(ann, ids)
    assert stats["reviews_with_two_or_more_annotations"] == 1
    assert stats["label_counts"] == {"negativo": 1, "positivo": 1}
    with pytest.raises(CorpusError, match="existe anotación"):
        record_annotation(ann, ids, review_id="r01", annotator_id="a1",
                          sentiment="neutral", aspects={})
    reviews_data, _ = validate_reviews(reviews, src)
    record_adjudication(adj, "r01", "negativo", {"espera": "negativo"},
                        "lead", "Discusión de dos anotaciones", reviews_data, annotations)
    final, report = validate_adjudications(adj, ids, annotations)
    assert final[0]["sentiment"] == "negativo"
    assert report["adjudications"] == 1


@pytest.mark.parametrize("raw", ['{"servicio":"excelente"}', '{"banco":"positivo"}',
                                  '["positivo"]', 'not json'])
def test_unknown_aspects_or_polarity_rejected(raw):
    with pytest.raises(CorpusError):
        parse_aspects(raw)


def test_cli_init_consent_gate_and_annotation(corpus, tmp_path):
    root, src, reviews, ann, adj = corpus
    textfile = tmp_path / "contribution.txt"
    textfile.write_text("Reseña voluntaria de demostración", encoding="utf-8")
    args = ["--root", str(root), "contribute", "--text-file", str(textfile),
            "--source-id", "voluntary2026", "--group-id", "g1"]
    with pytest.raises(CorpusError, match="consentimiento"):
        run(args)
    response = run(args + ["--consent-confirmed"])
    rid = response["review_id"]
    assert rid.startswith("r_")
    assert run(["--root", str(root), "validate"])["reviews"]["rows"] == 1
    run(["--root", str(root), "annotate", "--review-id", rid, "--annotator", "a1",
         "--sentiment", "neutral", "--aspects", "{}"])
    run(["--root", str(root), "adjudicate", "--review-id", rid, "--adjudicator", "lead",
         "--sentiment", "neutral", "--reason", "Revisión humana", "--aspects", "{}"])
    assert run(["--root", str(root), "validate"])["adjudications"]["adjudications"] == 1
    assert "Reseña voluntaria" not in json.dumps(run(["--root", str(root), "validate"]),
                                                 ensure_ascii=False)


def test_leakage_safe_deterministic_group_split(corpus, tmp_path):
    root, src, reviews, ann, adj = corpus
    rows, labels = _synthetic_records(180)
    # Duplicate a text into an otherwise independent group; their split must agree.
    rows[179]["text"] = rows[0]["text"]
    labels["r179"] = labels["r000"]
    _write_csv(reviews, REVIEW_COLUMNS, rows)
    _write_csv(ann, ANNOTATION_COLUMNS, [
        {"review_id": rid, "annotator_id": "a1", "sentiment": label,
         "aspects_json": "{}"} for rid, label in labels.items()
    ])
    _write_csv(adj, ADJUDICATION_COLUMNS, [
        {"review_id": rid, "sentiment": label, "aspects_json": "{}",
         "adjudicator_id": "lead", "rationale": "Etiqueta de prueba sintética"}
        for rid, label in labels.items()
    ])
    out = tmp_path / "processed" / "corpus"
    manifest1 = prepare_splits(reviews, src, ann, adj, out, seed=42)
    manifest2 = prepare_splits(reviews, src, ann, adj, out, seed=42)
    assert manifest1 == manifest2
    assert sum(x["rows"] for x in manifest1["partitions"].values()) == 180
    assigned = {}
    texts = {}
    for name in ("train", "validation", "test"):
        with (out / (name + ".csv")).open(encoding="utf-8", newline="") as file:
            parsed = list(csv.DictReader(file))
        for row in parsed:
            assigned[row["review_id"]] = name
            key = text_sha(row["text"])
            if key in texts:
                assert texts[key] == name
            texts[key] = name
    assert assigned["r000"] == assigned["r179"]
    assert not any("Texto artificial" in json.dumps(manifest1) for _ in [0])


def test_refuses_split_if_not_all_three_labels(corpus, tmp_path):
    _, src, reviews, ann, adj = corpus
    records, _ = _synthetic_records(36)
    _write_csv(reviews, REVIEW_COLUMNS, records)
    _write_csv(ann, ANNOTATION_COLUMNS, [
        {"review_id": r["review_id"], "annotator_id": "a1",
         "sentiment": "positivo", "aspects_json": "{}"} for r in records
    ])
    _write_csv(adj, ADJUDICATION_COLUMNS, [
        {"review_id": r["review_id"], "sentiment": "positivo",
         "aspects_json": "{}", "adjudicator_id": "lead",
         "rationale": "Prueba sintética"} for r in records
    ])
    with pytest.raises(CorpusError, match="tres clases"):
        prepare_splits(reviews, src, ann, adj, tmp_path / "processed" / "out")


def test_split_refuses_to_write_to_public_directory(corpus, tmp_path):
    root, src, reviews, ann, adj = corpus
    records, labels = _synthetic_records(180)
    _write_csv(reviews, REVIEW_COLUMNS, records)
    _write_csv(ann, ANNOTATION_COLUMNS, [
        {"review_id": rid, "annotator_id": "a1",
         "sentiment": value, "aspects_json": "{}"} for rid, value in labels.items()
    ])
    _write_csv(adj, ADJUDICATION_COLUMNS, [
        {"review_id": rid, "sentiment": value, "aspects_json": "{}",
         "adjudicator_id": "lead", "rationale": "Prueba sintética"}
        for rid, value in labels.items()
    ])
    with pytest.raises(CorpusError, match="private o data/processed"):
        prepare_splits(reviews, src, ann, adj, tmp_path / "public" / "out")
