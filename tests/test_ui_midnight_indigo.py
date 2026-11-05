"""Visual-only regression tests; no microphone or external dependencies."""
import json
import app


def test_midnight_indigo_has_light_and_dark_tokens():
    assert app._PRIMARY.c500 == "#4F46E5"
    assert app._PRIMARY.c300 == "#818CF8"
    assert "#F8FAFC" in str(app.THEME.get_theme_dict())
    assert "#0F172A" in str(app.THEME.get_theme_dict())
    assert "#FFFFFF" in str(app.THEME.get_theme_dict())
    assert "#1E293B" in str(app.THEME.get_theme_dict())


def test_three_tabs_and_event_wiring_remain_present():
    config = app.app.get_config_file()
    tabs = [c.get("props", {}).get("label") for c in config["components"]
            if c.get("type") in ("tabitem", "tab")]
    assert "Reseña individual" in tabs
    assert "Audio y transcripción local" in tabs
    assert "Lotes y revisión humana" in tabs
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
