"""Visual-only regression tests; no microphone or external dependencies."""
import json
import app


def test_caribe_and_utb_have_explicit_light_and_dark_tokens():
    assert app._PRIMARY.c500 == "#14b8a6"
    assert app.THEME.body_background_fill == "#f3f7fa"
    assert app.THEME.body_background_fill_dark == "#061217"
    assert app.THEME.block_background_fill == "#ffffff"
    assert app.THEME.block_background_fill_dark == "#0c1f27"


def test_four_tabs_and_event_wiring_remain_present():
    config = app.app.get_config_file()
    tabs = [c.get("props", {}).get("label") for c in config["components"]
            if c.get("type") in ("tabitem", "tab")]
    assert tabs == ["Análisis", "Audio", "Lotes", "Agentes"]
    dependencies = config["dependencies"]
    # 7 button clicks + 2 change/reset handlers, as in original contract.
    assert len(dependencies) >= 9
    assert len([d for d in dependencies if d.get("trigger_mode")]) >= 7


def test_new_visual_summary_preserves_json_contract():
    summary, raw = app.analyze(
        "El guía fue amable, pero esperamos demasiado para entrar.", "lr"
    )
    record = json.loads(raw)
    for expected in ("Sentimiento DEMO", "Puntuaciones no calibradas",
                     "Aspectos exploratorios", "Revisión humana",
                     "Recomendación exploratoria"):
        assert expected in summary
    for field in ("sentimiento", "puntuaciones_modelo_no_calibradas",
                  "aspectos_exploratorios", "fragmentos_literales_no_explicativos",
                  "revision_humana", "recomendacion"):
        assert field in record
    svm_summary, svm_raw = app.analyze("El servicio fue excelente.", "svm")
    assert "SVM" in svm_summary
    assert json.loads(svm_raw)["modelo_utilizado"] == "svm"
