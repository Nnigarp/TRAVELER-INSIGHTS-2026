"""Independently specified content expectations; no real LLM or prediction metrics."""
import copy
import html
import json
from pathlib import Path

import pytest

import app
from traveler_agent.batch import export_csv, export_json
from traveler_agent.ollama_report import (
    OllamaConfig, OllamaReporter, ReportRejected, evidence_units, report_schema,
    validate_report, validate_generation, render_generation_html,
)
from traveler_agent.report_content import content_options, content_provenance, PROPOSALS
from test_ollama_report import TEXT, fixture_report, response_for, simulated_ollama
from test_qwen_integration import from_request

BATTERY_PATH = Path(__file__).parent / "fixtures/qwen_report_quality_v2.json"
CASES = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_battery_has_grounded_distinct_summaries_and_contextual_proposals(case):
    source, sentiment, aspects, review = case["text"], case["sentiment"], case["aspects"], case["needs_human"]
    options = content_options(source, evidence_units(source))
    assert options["mode"] == case["mode"]
    for summary in options["summaries"]:
        report = fixture_report(source, sentiment, review, aspects)
        report["resumen"] = summary
        validated = validate_report(report, source, sentiment, aspects, review)
        assert all(term.casefold() in summary.casefold() for term in case["summary_terms"])
        assert report["sentimiento_analizado"] == sentiment
        assert report["aspectos_identificados"] == aspects
        assert report["requiere_revision_humana"] is review
        assert all(quote and quote in source for quote in report["evidencia_textual"])
        if case["mode"] == "PARAFRASIS_CONTROLADA":
            assert summary != source
        assert case["problem_term"] in report["problema_identificado"]
        assert report["recomendacion"] in PROPOSALS[case["category"]][:2]
        assert report["accion_propuesta"] == PROPOSALS[case["category"]][2]
        assert validated == report


@pytest.mark.parametrize("field,value,code", [
    ("resumen", "El hotel de Cartagena incumplió el contrato el lunes.", "HECHOS_NO_SUSTENTADOS"),
    ("resumen", "El transporte llegó y se realizó la excursión.", "HECHOS_NO_SUSTENTADOS"),
    ("problema_identificado", "El hotel debe pagar la excursión.", "HECHOS_NO_SUSTENTADOS"),
    ("recomendacion", "Exigir al hotel el reembolso de 200 euros.", "PROPUESTA_NO_ADMITIDA"),
    ("accion_propuesta", "Ya enviamos una reclamación al hotel.", "PROPUESTA_NO_ADMITIDA"),
    ("evidencia_textual", ["el transporte llegó"], "EVIDENCIA_NO_VERIFICABLE"),
    ("evidencia_textual", [], "EVIDENCIA_NO_VERIFICABLE"),
    ("evidencia_textual", "el transporte nunca llegó", "CAMPOS_INVALIDOS"),
    ("sentimiento_analizado", "positivo", "ANALISIS_ALTERADO"),
    ("aspectos_identificados", ["precio"], "ANALISIS_ALTERADO"),
    ("requiere_revision_humana", False, "ANALISIS_ALTERADO"),
])
def test_free_inventions_omissions_and_overrides_are_rejected(field, value, code):
    report = fixture_report(TEXT, "negativo", True, ["acceso"])
    report[field] = value
    with pytest.raises(ReportRejected, match=code):
        validate_report(report, TEXT, "negativo", ["acceso"], True)


@pytest.mark.parametrize("source", [
    "El transporte nunca llegó?", "¿La habitación estaba sucia?", "El hotel era cómodo...",
    "No fue cierto. La habitación estaba sucia.", "Dicen que el transporte nunca llegó.",
    "No estaba sucia la habitación.", "El hotel era cómodo, pero no sé si el transporte llegó.",
])
def test_uncertain_scope_never_becomes_a_confirmed_incident(source):
    report = fixture_report(source)
    options = content_options(source, evidence_units(source))
    assert options["mode"] == "CITA_CONSERVADORA"
    assert report["problema_identificado"] == ""
    assert all(quote in source for quote in report["evidencia_textual"])
    assert report["resumen"] in evidence_units(source)


def test_main_problem_and_proposal_must_match_even_when_both_are_individually_allowed():
    source = "La habitación estaba sucia. Nos cobraron más de lo anunciado."
    report = fixture_report(source)
    report["recomendacion"] = PROPOSALS["precio"][0]
    with pytest.raises(ReportRejected, match="PROPUESTA_DESCONTEXTUALIZADA"):
        validate_report(report, source, "negativo", [], True)


def test_action_agent_recomputes_provenance_and_preserves_original_control():
    report = fixture_report(TEXT, "positivo", True, ["acceso"])
    # A positive classifier label cannot erase negative textual evidence, nor can
    # the generated content change that label to make a nicer report.
    generation = {"estado": "GENERADO", "informe": report,
                  "modelo_solicitado": "qwen3:1.7b", "modelo_ejecutado": "qwen3:1.7b",
                  "validacion_contenido": {"resumen": "UNTRUSTED"}}
    validated = validate_generation(generation, TEXT, "positivo", ["acceso"], True)
    assert validated["validacion_contenido"]["resumen"] == "PARAFRASIS_CONTROLADA"
    assert validated["informe"]["sentimiento_analizado"] == "positivo"
    assert validated["informe"]["requiere_revision_humana"] is True
    assert "pérdida" in validated["informe"]["resumen"]
    assert validated["acciones_externas_ejecutadas"] is False


def test_card_is_ordered_readable_inert_and_survives_legacy_reports(monkeypatch):
    with simulated_ollama(from_request) as (url, _):
        monkeypatch.setenv("TRAVELER_OLLAMA_ENDPOINT", url)
        summary, raw, view = app.analyze_with_agents(TEXT, "lr", True)
    record = json.loads(raw)
    card = render_generation_html(record)
    headings = ["Resumen ejecutivo", "Sentimiento del clasificador", "Aspectos del análisis", "Problema identificado",
                "Evidencia textual", "Recomendación contextual", "Acción propuesta (sin ejecutar)",
                "Revisión humana", "Advertencias", "Información técnica de generación"]
    assert [card.index(h) for h in headings] == sorted(card.index(h) for h in headings)
    assert '<details class="ti-report-tech">' in card
    assert "Recomendación exploratoria" not in summary
    assert len(record["multiagente"]["traza"]) == 10 and len(record["multiagente"]["traza_generacion"]) == 3
    legacy = copy.deepcopy(record)
    legacy["generacion_qwen"]["informe"].pop("evidencia_textual")
    assert "Protocolo anterior sin evidencia separada" in render_generation_html(legacy)
    for source in ('<script>evil()</script>\n\n![img](https://example.com)', '=HYPERLINK("bad")'):
        report = fixture_report(source)
        generation = {**record["generacion_qwen"], "informe": report}
        safe = render_generation_html({"generacion_qwen": generation})
        assert "<script>" not in safe and "<img" not in safe and "<a " not in safe


def test_literal_evidence_and_modes_survive_audio_batch_exports_and_human_correction(tmp_path, monkeypatch):
    import csv
    import io
    with simulated_ollama(from_request) as (url, _):
        monkeypatch.setenv("TRAVELER_OLLAMA_ENDPOINT", url)
        records, *_ = app.run_batch_lines(TEXT, "svm", True)
        summary, raw = app.analyze_audio_ui(TEXT, {"modalidad": "audio", "transcripcion_original": "El hotel era cómodo."}, "lr", True)
    audio = json.loads(raw)
    assert audio["generacion_qwen"]["informe"]["evidencia_textual"] == records[0]["generacion_qwen"]["informe"]["evidencia_textual"]
    edited, *_ = app.revise(records, "demo-001", "neutral", "Revisión humana")
    rows = list(csv.DictReader(io.StringIO(export_csv(edited))))
    assert json.loads(rows[0]["qwen_evidencia_textual"]) == records[0]["generacion_qwen"]["informe"]["evidencia_textual"]
    assert json.loads(rows[0]["qwen_validacion_contenido"])["resumen"] == "PARAFRASIS_CONTROLADA"
    assert rows[0]["qwen_vigencia"] == "ANTERIOR_A_CORRECCION_HUMANA"
    assert json.loads(export_json(edited))["registros"][0]["generacion_qwen"] == records[0]["generacion_qwen"]


def test_reproducible_battery_reports_contract_scope_and_observable_limits(tmp_path, monkeypatch):
    from scripts import check_qwen_report_quality as battery
    output = tmp_path / "contracts.json"
    monkeypatch.setattr(OllamaReporter, "generate", lambda *a: pytest.fail("Contract mode must not call the LLM"))
    assert battery.main(["--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["scope"] == "CONTRACT_ONLY_NO_LLM_OR_CLASSIFIER_EXECUTED"
    assert len(result["cases"]) == len(CASES) == 12
    assert result["classifier_requested"] == "NOT_EXECUTED"
    fallback = [row for row in result["cases"] if row["observed_limitations"]]
    assert {row["id"] for row in fallback} == {"negated_problem", "uncertain_attribution", "regional", "irony"}
    with pytest.raises(SystemExit):
        battery.main(["--output", str(output)])


def test_actual_battery_requires_effective_classifiers_and_qwen(tmp_path, monkeypatch):
    from scripts import check_qwen_report_quality as battery
    with simulated_ollama(from_request) as (url, _):
        monkeypatch.setenv("TRAVELER_OLLAMA_ENDPOINT", url)
        result = battery.run(actual=True, classifier="svm")
    # This test server is simulated; the production command has no fixture mode
    # that fabricates an API response. Do not cite this as real Qwen inference.
    assert result["status"] == "PASSED"
    assert result["ollama_identity"]["ollama_version"] == "SIMULATED_TEST_ONLY"
    assert all(row["observed"]["modelo_utilizado"] == "svm" for row in result["cases"])
    monkeypatch.setenv("TRAVELER_OLLAMA_ENDPOINT", "http://127.0.0.1:1")
    result = battery.run(actual=True, classifier="lr")
    assert result["status"] == "FAILED"
    assert "OLLAMA_IDENTITY_NOT_VERIFIED" in result["preflight_violations"]
    assert all("QWEN_NOT_VALIDATED" in row["violations"] for row in result["cases"])
