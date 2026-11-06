"""Regression checks for a presentation-only agent trace in Midnight Indigo."""
import copy
import html
import json

import app
from traveler_agent.multiagent_view import render_multiagent_view


def _record(state="COMPLETADO"):
    return {
        "sentimiento": "negativo",
        "recomendacion": "Revisar la atención.",
        "revision_humana": state == "REVISION_HUMANA",
        "multiagente": {
            "orquestador": "TravelerOrchestrator",
            "estado": state,
            "razones_revision": ["SCORE_DEMO_BAJO"] if state == "REVISION_HUMANA" else [],
            "traza": [
                {"sequence": 1, "agent": "orquestador", "task": "plan", "status": "PLANNED",
                 "note_code": "PLAN_FIJO_CON_GATES"},
                *[
                    item
                    for n, name in enumerate(("percepcion", "semantica", "control", "accion"))
                    for item in (
                        {"sequence": 2 + n*2, "agent": "orquestador", "task": name,
                         "status": "DELEGATED", "note_code": "HANDOFF"},
                        {"sequence": 3 + n*2, "agent": name, "task": name,
                         "status": "OK", "note_code": "VALIDATED_CONTRACT"},
                    )
                ],
                {"sequence": 10, "agent": "orquestador", "task": "cierre",
                 "status": state, "note_code": "NO_EXTERNAL_ACTION"},
            ],
        },
    }


def test_completed_renders_only_actual_events_and_card_statuses():
    record = _record()
    before = copy.deepcopy(record)
    markup = render_multiagent_view(record)
    assert "TravelerOrchestrator" in markup
    assert all(name in markup for name in ("Percepción", "Semántica", "Control", "Acción"))
    assert "Ejecución completada" in markup
    assert "Revisar la atención." in markup
    assert markup.count('class="ti-flow-event"') == 10
    assert record == before


def test_human_review_renders_reasons_from_record_only():
    record = _record("REVISION_HUMANA")
    markup = render_multiagent_view(record)
    assert "Revisión humana" in markup
    assert "Puntuación DEMO inferior al umbral interno" in markup
    assert "Ejecución completada" not in markup


def test_failed_partial_trace_does_not_invent_following_specialists():
    record = _record("DETENIDO_SEGURO")
    record["multiagente"]["traza"] = [
        {"sequence": 1, "agent": "orquestador", "task": "plan",
         "status": "PLANNED", "note_code": "PLAN_FIJO_CON_GATES"},
        {"sequence": 2, "agent": "orquestador", "task": "percepcion",
         "status": "DELEGATED", "note_code": "HANDOFF"},
        {"sequence": 3, "agent": "percepcion", "task": "percepcion",
         "status": "FAILED", "note_code": "SAFE_ABORT"},
    ]
    markup = render_multiagent_view(record)
    assert markup.count('class="ti-flow-event"') == 3
    assert "Detenido con seguridad" in markup
    assert "Fallido" in markup
    assert markup.count("Sin ejecutar") == 3
    assert "Revisar la atención." not in markup
    assert "Ejecución completada" not in markup


def test_no_trace_does_not_fabricate_failure_details():
    markup = render_multiagent_view({})
    assert "Sin traza de ejecución disponible" in markup
    assert "Fallido" not in markup
    assert "Ejecución completada" not in markup


def test_ui_wrapper_does_not_reclassify_or_modify_json():
    review = "El guía fue amable, pero esperamos demasiado para entrar."
    expected_summary, expected_raw = app.analyze(review, "lr")
    summary, raw, visual = app.analyze_with_agents(review, "lr")
    assert summary == expected_summary
    assert json.loads(raw) == json.loads(expected_raw)
    assert "TravelerOrchestrator" in visual
    assert "Revisión humana" in visual
    assert visual.count('class="ti-flow-event"') == len(
        json.loads(raw)["multiagente"]["traza"]
    )


def test_callback_invokes_original_analyze_once(monkeypatch):
    calls = []
    original = _record()
    def fake_analyze(text, kind):
        calls.append((text, kind))
        return "Resumen antiguo", json.dumps(original, ensure_ascii=False)
    monkeypatch.setattr(app, "analyze", fake_analyze)
    summary, raw, view = app.analyze_with_agents("ejemplo", "svm")
    assert calls == [("ejemplo", "svm")]
    assert summary == "Resumen antiguo"
    assert json.loads(raw) == original
    assert "TravelerOrchestrator" in view


def test_errors_are_not_misrepresented_as_traces():
    summary, raw, html_view = app.analyze_with_agents("", "lr")
    assert "Error" in summary
    assert raw == "{}"
    assert "Sin traza de ejecución disponible" in html_view
    assert "Detenido con seguridad" not in html_view


def test_html_escaping_and_strict_allowlist_for_trace():
    record = _record()
    payload = '<script>alert("secret")</script> /private/audio.wav'
    record["recomendacion"] = payload
    record["multiagente"]["traza"][2]["agent"] = payload
    record["multiagente"]["traza"][2]["note_code"] = payload
    markup = render_multiagent_view(record)
    assert "<script>" not in markup
    assert html.escape(payload, quote=True) in markup  # only in permitted recommendation
    assert markup.count(payload) == 0
    assert "Agente no identificado" in markup
    # The coordinator delegated perception before its response was modified:
    # a recorded handoff is DELEGATED, never silently marked OK or not started.
    assert "Delegado" in markup


def test_three_tabs_and_all_original_batch_audio_callbacks_remain():
    config = app.app.get_config_file()
    tabs = {component.get("props", {}).get("label") for component in config["components"]
            if component.get("type") in ("tabitem", "tab")}
    assert {"Reseña individual", "Audio y transcripción local", "Lotes y revisión humana"} <= tabs
    assert len(config["dependencies"]) >= 9
    assert app.analyze_audio_ui and app.run_batch_lines and app.run_batch_csv
    assert app.revise and app.export_files and app.reset_audio_trace
    assert app.THEME.body_background_fill_dark == "#0F172A"
