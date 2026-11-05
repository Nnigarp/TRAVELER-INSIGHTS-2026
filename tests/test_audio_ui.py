"""Gradio audio callback smoke tests with an injected ASR stub; no weights."""
from __future__ import annotations
import json
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import app


def make_wav(path: Path):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack("<h", 2000) * 16000)
    return path


def test_ui_audio_trace_and_manual_correction(tmp_path, monkeypatch):
    from traveler_agent.audio_asr import recognize_audio

    class Mock:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(text="Fue malo el hotel.")]), SimpleNamespace(language="es")

    fake = Mock()
    monkeypatch.setattr(app, "recognize_audio",
                        lambda filepath, language: recognize_audio(filepath, language, backend=fake))
    wav = make_wav(tmp_path / "sample.wav")
    original, trace, status = app.transcribe_audio_ui(str(wav), "es")
    assert original == "Fue malo el hotel."
    assert trace["audio_sha256"]
    assert "terminada" in status
    summary, data = app.analyze_audio_ui("No fue malo el hotel.", trace, "lr")
    result = json.loads(data)
    assert result["procedencia_entrada"]["transcripcion_original"] == original
    assert result["procedencia_entrada"]["fue_corregida"] is True
    assert result["modelo_utilizado"] == "lr"
    summary2, data2 = app.analyze_audio_ui("No fue malo el hotel.", trace, "svm")
    assert json.loads(data2)["modelo_utilizado"] == "svm"
    assert "Sentimiento DEMO" in summary2


def test_ui_audio_empty_input_reset_and_guard():
    text, trace, status = app.transcribe_audio_ui(None, "es")
    assert text == "" and trace == {}
    assert "Error de audio" in status
    summary, payload = app.analyze_audio_ui("Reseña", {}, "lr")
    assert "Error" in summary and payload == "{}"
    assert app.reset_audio_trace() == ("", {}, "", "", "{}")
