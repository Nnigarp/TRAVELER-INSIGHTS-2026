"""No real Whisper downloads are necessary to exercise ASR integration."""
from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from traveler_agent.audio_asr import (
    AudioInputError, analyze_edited_transcript, recognize_audio, validate_wav
)
from traveler_agent.demo_data import DEMO
from traveler_agent.model import train


def write_wav(path: Path, *, duration=.5, value=900, rate=16000, nchannels=1):
    n = int(rate * duration)
    samples = struct.pack("<h", value) * n * nchannels
    with wave.open(str(path), "wb") as out:
        out.setnchannels(nchannels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples)
    return path


class FakeASR:
    def __init__(self, text="No fue malo. El servicio estuvo bien.", language="es"):
        self.text, self.language = text, language
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter([SimpleNamespace(text=self.text)]), SimpleNamespace(language=self.language)


@pytest.fixture
def wav(tmp_path):
    return write_wav(tmp_path / "voice.wav")


@pytest.fixture(scope="module")
def classifier():
    return train([x for x, _ in DEMO], [y for _, y in DEMO])


def test_audio_validation_success_and_sha(wav):
    info = validate_wav(wav)
    assert info.duration_seconds == .5
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert len(info.sha256) == 64


@pytest.mark.parametrize("variant", ["none", "missing", "wrong_extension", "corrupt", "silent", "short", "long", "oversize"])
def test_audio_validation_rejects_invalid(tmp_path, variant):
    path = tmp_path / "voice.wav"
    if variant == "none":
        with pytest.raises(AudioInputError, match="Graba audio"):
            validate_wav(None)
        return
    if variant == "missing":
        with pytest.raises(AudioInputError, match="no está disponible"):
            validate_wav(path)
        return
    if variant == "wrong_extension":
        path = tmp_path / "voice.mp3"
        path.write_bytes(b"f" * 50)
    elif variant == "corrupt":
        path.write_bytes(b"no_wav" * 20)
    elif variant == "silent":
        write_wav(path, value=0)
    elif variant == "short":
        write_wav(path, duration=.1)
    elif variant == "long":
        write_wav(path, duration=121)
    elif variant == "oversize":
        path.write_bytes(b"x" * (12 * 1024 * 1024 + 1))
    with pytest.raises(AudioInputError):
        validate_wav(path)


@pytest.mark.parametrize("language", ["es", "en", "it", "auto"])
def test_mock_asr_languages_without_download(wav, language):
    asr = FakeASR()
    text, trace = recognize_audio(wav, language, backend=asr)
    assert text == "No fue malo. El servicio estuvo bien."
    assert trace["transcripcion_original"] == text
    assert trace["modalidad"] == "audio"
    assert trace["motor_asr"] == "ASR_SIMULADO_TEST"
    assert trace["idioma_solicitado"] == language
    assert asr.calls[0][1]["language"] == (None if language == "auto" else language)
    assert asr.calls[0][1]["vad_filter"] is True
    assert "voice.wav" not in json.dumps(trace)


def test_empty_asr_rejected(wav):
    with pytest.raises(AudioInputError, match="No se reconoció"):
        recognize_audio(wav, "es", backend=FakeASR("   "))


def test_asr_runtime_error_sanitized(wav):
    class FailingASR:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("private file path or audio content")
    with pytest.raises(AudioInputError, match="Falló el reconocimiento") as err:
        recognize_audio(wav, backend=FailingASR())
    assert "private file path" not in str(err.value)


def test_invalid_language_before_transcription(wav):
    with pytest.raises(AudioInputError, match="Idioma no admitido"):
        recognize_audio(wav, "xx", backend=FakeASR())


def test_edited_transcript_preserves_asr_negation_and_original(wav, classifier):
    asr = FakeASR("Fue malo y no volvería.")
    original, trace = recognize_audio(wav, backend=asr)
    corrected = "No fue malo y volvería 😊."
    record = analyze_edited_transcript(corrected, trace, classifier, model_kind="svm")
    assert record["sentimiento"] in ("negativo", "neutral", "positivo")
    assert record["modelo_utilizado"] == "svm"
    assert record["procedencia_entrada"]["transcripcion_original"] == original
    assert record["procedencia_entrada"]["transcripcion_editada"] == corrected
    assert record["procedencia_entrada"]["fue_corregida"] is True
    assert "no" in [x["texto"].lower() for x in record["perfil_linguistico_exploratorio"]["indicios_contextuales"]["negaciones"]]
    assert "audio_sha256" in json.dumps(record)


def test_no_analyze_without_provenance(classifier):
    with pytest.raises(AudioInputError, match="Transcribe"):
        analyze_edited_transcript("Hola", None, classifier)


def test_audio_provenance_does_not_modify_input(wav, classifier):
    _, metadata = recognize_audio(wav, backend=FakeASR("Buen servicio"))
    record = analyze_edited_transcript("Buen servicio", metadata, classifier)
    assert record["procedencia_entrada"]["fue_corregida"] is False
    assert "transcripcion_editada" not in metadata


def test_no_dependency_download_or_internet_for_mock(monkeypatch, wav):
    monkeypatch.setenv("TRAVELER_ASR_MODEL", "not-a-model")
    text, trace = recognize_audio(wav, backend=FakeASR("Hola"))
    assert text == "Hola"
    assert trace["modelo_asr"] == "backend_simulado"


def test_truncated_declared_wav_frames_rejected(tmp_path):
    wav = write_wav(tmp_path / "truncated.wav")
    original = wav.read_bytes()
    wav.write_bytes(original[:-3000])
    with pytest.raises(AudioInputError, match="truncado"):
        validate_wav(wav)
