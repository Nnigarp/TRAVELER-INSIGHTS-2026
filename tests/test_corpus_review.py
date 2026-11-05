"""Annotation spans and quality signals: only synthetic local fixtures."""
import csv
import json

import pytest

from traveler_agent.corpus import CorpusError, ANNOTATION_COLUMNS, _write_csv
from traveler_agent.corpus_review import (
    SPAN_COLUMNS, validate_span_entries, record_evidence,
    validate_evidence, annotator_agreement, near_duplicate_candidates
)


def test_positional_evidence_is_literal_and_unicode_safe(tmp_path):
    review = "El hotel estaba sucio 😡, pero la atención fue excelente."
    start = review.index("sucio")
    evidence = validate_span_entries(review, [{
        "aspect": "higiene", "polarity": "negativo",
        "start": start, "end": start + len("sucio")
    }])
    assert evidence[0]["text"] == review[start:start + 5]
    assert evidence[0]["start"] == str(start)


@pytest.mark.parametrize("spans", [
    [{"aspect":"higiene","polarity":"negativo","start":-1,"end":2}],
    [{"aspect":"higiene","polarity":"negativo","start":1,"end":999}],
    [{"aspect":"higiene","polarity":"negativo","start":True,"end":3}],
    [{"aspect":"invalid","polarity":"positivo","start":0,"end":2}],
    [{"aspect":"higiene","polarity":"negativo","start":0,"end":0}],
])
def test_rejects_invalid_literal_spans(spans):
    with pytest.raises(CorpusError):
        validate_span_entries("texto original", spans)


def test_evidence_validation_and_annotation_consistency(tmp_path):
    text = "El servicio fue bueno, pero hubo espera."
    start = text.index("servicio")
    spans = [{"aspect":"servicio","polarity":"positivo",
              "start":start,"end":start+len("servicio")}]
    target = tmp_path / "evidence_spans.csv"
    record_evidence(target, "r01", "a01", text, spans, {"servicio":"positivo"})
    review_rows = [{"review_id":"r01","text":text}]
    annotations = [{
        "review_id":"r01","annotator_id":"a01","aspects_json":'{"servicio":"positivo"}'
    }]
    result = validate_evidence(target, review_rows, annotations)
    assert result["evidence_spans"] == 1
    with target.open("r",encoding="utf-8",newline="") as f:
        data = list(csv.DictReader(f))
    data[0]["text"] = "alterado"
    _write_csv(target, SPAN_COLUMNS, data)
    with pytest.raises(CorpusError, match="no coincide"):
        validate_evidence(target, review_rows, annotations)


def test_agreement_requires_overlapping_human_reviews():
    rows = [
        {"review_id":"r01","annotator_id":"a1","sentiment":"positivo",
         "aspects_json":'{"servicio":"positivo"}'},
        {"review_id":"r01","annotator_id":"a2","sentiment":"positivo",
         "aspects_json":'{"servicio":"positivo"}'},
        {"review_id":"r02","annotator_id":"a1","sentiment":"negativo",
         "aspects_json":'{"higiene":"negativo"}'},
        {"review_id":"r02","annotator_id":"a2","sentiment":"positivo",
         "aspects_json":'{"higiene":"negativo"}'},
    ]
    outcome = annotator_agreement(rows)
    assert len(outcome["comparisons"]) == 1
    c = outcome["comparisons"][0]
    assert c["shared_reviews"] == 2
    assert c["global_kappa"] is not None
    assert c["aspect_kappa"]["higiene"] == 1.0


def test_near_duplicate_candidates_do_not_delete_or_expose_text():
    reviews = [
        {"review_id":"r01","text":"El hotel estuvo muy limpio y bonito"},
        {"review_id":"r02","text":"El hotel estuvo mui limpio y bonito"},
        {"review_id":"r03","text":"Había lluvias torrenciales"},
    ]
    result = near_duplicate_candidates(reviews, threshold=.85)
    assert result["candidates"]
    assert result["candidates"][0]["review_ids"] == ["r01","r02"]
    assert result["candidates"][0]["decision"] == "revision_humana_pendiente"
    assert "mui limpio" not in json.dumps(result)
    with pytest.raises(CorpusError, match="limitada"):
        near_duplicate_candidates(reviews, max_rows=2)


def test_annotation_ui_callbacks_allow_labeled_evidence(monkeypatch, tmp_path):
    import corpus_annotation_app as ui
    from traveler_agent.corpus import REVIEW_COLUMNS, SOURCE_COLUMNS, _write_csv
    root = tmp_path / "private" / "corpus"
    root.mkdir(parents=True)
    _write_csv(root / "sources.csv", SOURCE_COLUMNS, [{
        "source_id":"s1","source_type":"voluntary","source_reference":"synthetic_test",
        "reuse_basis":"consent","permission_reference":"simulated",
        "authorization_reviewed":"yes"
    }])
    text = "El hotel estaba sucio."
    _write_csv(root / "reviews.csv", REVIEW_COLUMNS, [{
        "review_id":"r01","text":text,"source_id":"s1","group_id":"g1","language":"es"
    }])
    _write_csv(root / "annotations.csv", ANNOTATION_COLUMNS, [])
    monkeypatch.setattr(ui, "ROOT", root)
    assert ui.available_reviews() == ["r01"]
    assert ui.read_review("r01") == text
    start = text.index("sucio")
    spans = json.dumps([{
        "aspect":"higiene","polarity":"negativo",
        "start":start,"end":start+5
    }])
    assert '"text": "sucio"' in ui.preview_evidence("r01", spans)
    msg = ui.submit_annotation("r01","a01","negativo",'{"higiene":"negativo"}', spans)
    assert "guardada" in msg
    assert (root / "evidence_spans.csv").is_file()
    assert "Error" in ui.submit_annotation("r01","a01","positivo","{}", "[]")
