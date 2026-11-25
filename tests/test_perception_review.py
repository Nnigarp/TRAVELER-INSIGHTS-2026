"""Functional perception checks. Synthetic ASR metadata, no trained weights."""
import json
from types import SimpleNamespace

import app
from traveler_agent.audio_asr import perception_signals
from traveler_agent.multiagent_view import render_agents_overview


def test_misheard_italian_stops_before_classifier(monkeypatch):
    heard = "y el servicio era excelente, maí, transporto, no me maí, arriba a todo y habíamos perso y el escorsión."
    info = perception_signals([], SimpleNamespace(language="es"), heard, "auto")
    assert info["advertencias_percepcion"] and info["probabilidad_idioma_auto"] is None
    origin = {**info, "transcripcion_original": heard, "idioma_detectado_asr": "es"}
    monkeypatch.setattr(app, "run_selected", lambda *args: (_ for _ in ()).throw(AssertionError("Clasificación prematura")))
    card, raw, _ = app.unified(heard, "beto", False, origin)
    assert raw == "{}" and "retranscribe" in card


def test_segment_data_preserves_observations_and_decoder_options():
    segment = SimpleNamespace(avg_logprob=-1.3, compression_ratio=3, no_speech_prob=.61)
    options = SimpleNamespace(log_prob_threshold=-1, compression_ratio_threshold=2.4)
    info = SimpleNamespace(language="es", language_probability=.4, transcription_options=options)
    signals = perception_signals([segment], info, "El hotel fue excelente.", "auto")
    assert signals["probabilidad_idioma_auto"] == .4 and len(signals["advertencias_percepcion"]) == 3
    assert signals["segmentos_asr"][0]["no_speech_prob"] == .61
    assert perception_signals([], info, "El hotel fue excelente.", "es")["probabilidad_idioma_auto"] is None


def test_reviewed_audio_correction_retains_history(monkeypatch):
    original = "El transporte nunca llegó y perdimos la exclusión."
    fixed = original.replace("exclusión", "excursión")
    origin = {"audio_original_sha256": "a"*64, "modalidad": "audio", "transcripcion_original": original,
              "idioma_detectado_asr": "es", "advertencias_percepcion": ["Revisar el segmento."]}
    monkeypatch.setenv("TRAVELER_LANGUAGE_QWEN", "0")
    _, raw, _ = app.unified(fixed, "lr", False, origin, asr_reviewed=True)
    record = json.loads(raw)
    assert record["multiagente"]["transcripcion_editada"] is True
    trace = record["procedencia_entrada"]
    assert trace["transcripcion_original"] == original and trace["transcripcion_editada"] == fixed
    assert trace["audio_original_sha256"] == origin["audio_original_sha256"] and trace["modalidad"] == "audio"
    assert trace["fue_corregida"] and trace["revision_percepcion_confirmada"]
    assert trace["texto_analizado"] == fixed
    html = render_agents_overview(record)
    assert "10 principales · 0 generativos" in html and "Sin consulta" in html and "Audio · Whisper" in html


def test_language_disagreement_preserves_input_for_retranscription():
    signals = perception_signals([], SimpleNamespace(language="es"), "Il servizio non era buono.", "auto")
    assert any("distintos" in warning for warning in signals["advertencias_percepcion"])



def test_specific_translation_warnings_require_review(monkeypatch):
    from traveler_agent import language_input as li
    for lang, source, proposal in (
        ("fr", "Nous avons manqué l'excursion.", "Nos faltó la excursión."),
        ("zh", "接驳车没有来。", "El vehículo de albergamiento no llegó."),
        ("zh", "接驳车从未来。", "El vehículo llegó."),
    ):
        monkeypatch.setattr(li, "ask_qwen", lambda *args: proposal)
        result = li.prepare(source, manual=lang)
        assert result["advertencias"] and result["texto_analizado"] is None
        assert result["traduccion_automatica"] == proposal
