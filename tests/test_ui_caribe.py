"""Presentation, session adapters and accessibility contracts; no optional weights."""
import copy
import html
import json
import re

import pytest

import app
from traveler_agent.ui_presenter import EMPTY_REPORT_HTML, render_result
from traveler_agent.multiagent_view import render_agents_overview
from traveler_agent.ui_style import UI_CSS, THEME_JS
from test_ollama_report import TEXT, simulated_ollama
from test_qwen_integration import from_request


def test_idle_ui_has_no_fabricated_result_and_examples_do_not_infer(monkeypatch):
    monkeypatch.setattr(app, "run_selected", lambda *a: pytest.fail("Presentation controls must not infer"))
    config = app.app.config
    models = [c["props"] for c in config["components"] if c["props"].get("elem_id") == "ti-model"]
    assert models[0]["value"] == "lr"
    text = [c["props"] for c in config["components"] if c["props"].get("elem_id") == "ti-review"]
    assert text[0]["value"] == ""
    assert "ti-indicators" not in EMPTY_REPORT_HTML
    assert app.clear_analysis()[0] == "" and app.clear_analysis()[2] == EMPTY_REPORT_HTML
    for name in app.DEMO_EXAMPLES:
        loaded = app.load_example(name)
        assert loaded[0] == app.DEMO_EXAMPLES[name]
        assert loaded[2] == EMPTY_REPORT_HTML and loaded[3] == "{}" and loaded[5] is None


@pytest.mark.parametrize("text,length", [("", 0), ("😃", 1), ("a" * 5000, 5000), ("b" * 5001, 5001)])
def test_counter_matches_backend_character_semantics(text, length):
    assert f"{length:,}" in app.character_count(text)
    assert ("límite excedido" in app.character_count(text)) is (length > 5000)


def test_session_adapters_use_records_without_changing_or_reclassifying(monkeypatch):
    _, encoded, _ = app.analyze_with_agents(TEXT, "lr")
    record = json.loads(encoded)
    original = copy.deepcopy(record)
    monkeypatch.setattr(app, "run_selected", lambda *a: pytest.fail("Agent view must not reclassify"))
    captured, view, raw, source = app.capture_execution(encoded)
    assert captured == original and raw == encoded and source == "Análisis individual"
    assert "TravelerOrchestrator" in view and view.count('class="ti-flow-event"') == 10
    assert app.capture_execution("{}")[0] is None
    assert app.capture_execution("invalid")[0] is None
    rows = [{**record, "id": "demo-001"}, {**record, "id": "demo-002"}]
    selected, _, _, source = app.capture_batch_execution(rows, "demo-002")
    assert selected["id"] == "demo-002" and source == "Lote · demo-002"
    assert app.capture_batch_execution(rows, "missing")[0] is None
    assert record == original


@pytest.mark.parametrize("enabled,fail", [(False, False), (True, False), (True, True)])
def test_report_layout_represents_actual_generation_state_and_four_detail_cards(monkeypatch, enabled, fail):
    with simulated_ollama(from_request, status=500 if fail else 200) as (url, calls):
        monkeypatch.setenv("TRAVELER_OLLAMA_ENDPOINT", url)
        summary, encoded, _ = app.analyze_with_agents(TEXT, "svm", enabled)
    record = json.loads(encoded)
    assert record["modelo_utilizado"] == "svm"
    assert summary.count('class="ti-report-block ') == 4
    assert summary.index('ti-report-summary') < summary.index('ti-indicators') < summary.index('ti-report-grid') < summary.index('ti-warnings') < summary.index('ti-report-tech')
    assert "Puntuaciones no calibradas" in summary and "probabilidad verificada" not in summary
    assert len(record["multiagente"]["traza"]) == 10
    if enabled:
        assert len(record["multiagente"]["traza_generacion"]) == 3
        assert len(calls) == 1
    if enabled and not fail:
        assert "Resumen ejecutivo" in summary and "Generado con Qwen" in summary
        assert record["generacion_qwen"]["informe"]["resumen"] in html.unescape(summary)
    else:
        assert "Salida tradicional basada en reglas" in summary
        assert "Resumen ejecutivo" not in summary
        assert record["recomendacion"] in html.unescape(summary)
    before = copy.deepcopy(record)
    render_result(record)
    assert record == before


def test_agent_service_is_subordinate_and_idle_states_are_honest():
    idle = render_agents_overview()
    assert idle.count('class="ti-flow-agent"') == 4
    assert "Sin ejecución registrada" in idle or "sin ejecución registrada" in idle
    assert "Completado" not in idle
    assert 'class="ti-qwen-service"' in idle and "servicio generativo subordinado" in idle
    assert all(name in idle for name in ("PerceptionAgent", "SemanticsAgent", "ControlAgent", "ActionAgent"))


def _contrast(a, b):
    def light(color):
        if len(color) == 4:
            color = "#" + "".join(c * 2 for c in color[1:])
        rgb = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
        rgb = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
        return sum(v * k for v, k in zip(rgb, (.2126, .7152, .0722)))
    x, y = sorted((light(a), light(b)))
    return (y + .05) / (x + .05)


def test_theme_contrast_focus_and_responsive_grids_have_explicit_contracts():
    for block in UI_CSS.split(':root[data-ti-theme="utb"]')[:2]:
        # Both palettes are defined before the component rules.
        palette = dict(re.findall(r"--ti-([\w-]+):\s*(#[0-9a-f]{3}(?:[0-9a-f]{3})?)\s*[;}]", block.split('}')[0]))
        for foreground, background in (("text", "panel"), ("muted", "panel"), ("gold", "gold-bg"), ("red", "panel3")):
            assert _contrast(palette[foreground], palette[background]) >= 4.5
        assert _contrast("#ffffff", palette["primary"]) >= 4.5
    assert ':focus-visible' in UI_CSS and 'prefers-reduced-motion' in UI_CSS
    assert 'grid-template-columns:minmax(0,1fr) minmax(0,1fr)' in UI_CSS
    assert '@media(max-width:600px)' in UI_CSS
    assert 'fetch(' not in THEME_JS and 'location' not in THEME_JS
