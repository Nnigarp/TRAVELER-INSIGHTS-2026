import csv
import io
import json
import pytest
from traveler_agent.batch import (
    parse_review_lines, parse_review_csv, analyze_batch,
    apply_human_correction, demo_trends, export_json, export_csv,
)
from traveler_agent.model import train
from traveler_agent.demo_data import DEMO

@pytest.fixture(scope="module")
def demo_model():
    return train([t for t, _ in DEMO], [y for _, y in DEMO])


def test_parse_lines_and_limits():
    assert parse_review_lines("  Bien 😍\n\n Mal 😡  ") == ["Bien 😍", "Mal 😡"]
    with pytest.raises(ValueError): parse_review_lines("\n")
    with pytest.raises(ValueError): parse_review_lines("x\n" * 201)
    with pytest.raises(ValueError): parse_review_lines("a" * 5001)


def test_csv_utf8_and_quoted_newlines():
    data = 'reseña,origen\n"Bien, excelente",A\n"Servicio\nregular",B\n'.encode("utf-8-sig")
    assert parse_review_csv(data) == ["Bien, excelente", "Servicio\nregular"]


def test_csv_rejects_invalid_and_large():
    with pytest.raises(ValueError): parse_review_csv(b"texto\nHola\n")
    with pytest.raises(ValueError): parse_review_csv(b"\xff")
    with pytest.raises(ValueError): parse_review_csv(b"x" * (1024 * 1024 + 1))


def test_batch_output_and_trends(demo_model):
    rows = analyze_batch(["El guía fue amable", "El precio era excesivo"], demo_model)
    assert len(rows) == 2
    assert rows[0]["id"] == "demo-001"
    assert rows[1]["id"] == "demo-002"
    assert all("puntuaciones_modelo_no_calibradas" in r for r in rows)
    assert sum(t["cantidad"] for t in demo_trends(rows)) == 2


def test_correction_preserves_model_prediction_and_original_state(demo_model):
    rows = analyze_batch(["La visita estuvo bien"], demo_model)
    original_label = rows[0]["sentimiento"]
    edited = apply_human_correction(rows, "demo-001", "negativo", "Revisión de muestra")
    assert edited[0]["sentimiento"] == original_label
    assert edited[0]["etiqueta_corregida_humano"] == "negativo"
    assert edited[0]["estado_revision"] == "corregida"
    assert rows[0]["etiqueta_corregida_humano"] is None
    assert "no automatizar" in edited[0]["recomendacion"]


def test_correction_rejects_invalid_requests(demo_model):
    rows = analyze_batch(["Buenas vistas"], demo_model)
    for id_, label in [("demo-999", "positivo"), ("demo-001", "desconocido")]:
        with pytest.raises(ValueError): apply_human_correction(rows, id_, label)


def test_json_csv_exports_keep_predictions_and_review(demo_model):
    rows = analyze_batch(["Bien 😍"], demo_model)
    rows = apply_human_correction(rows, "demo-001", "positivo")
    payload = json.loads(export_json(rows))
    assert payload["tipo"] == "DEMO_NO_VALIDADO"
    assert payload["registros"][0]["etiqueta_corregida_humano"] == "positivo"
    parsed = list(csv.DictReader(io.StringIO(export_csv(rows))))
    assert len(parsed) == 1
    assert parsed[0]["etiqueta_corregida_humano"] == "positivo"
