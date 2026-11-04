"""Local-only ASR adapter for Gradio-converted PCM WAV audio.

Only the optional backend knows about faster-whisper. Importing this module,
running pytest, and using text analysis never downloads or loads ASR weights.
"""
from __future__ import annotations

import hashlib
import os
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_AUDIO_SECONDS = 120.0
MIN_AUDIO_SECONDS = 0.25
LANGUAGES = ("es", "en", "it", "auto")
SUPPORTED_MODELS = ("tiny", "base", "small")


class AudioInputError(ValueError):
    """Audio input or recognition cannot be used safely."""


@dataclass(frozen=True)
class AudioInfo:
    sha256: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int


def validate_wav(filepath: str | Path | None) -> AudioInfo:
    """Validate actual PCM header, duration, channels, file size and silence.

    Gradio Audio(format='wav', type='filepath') converts microphone/uploads
    to WAV before reaching this function. Other formats MUST first be
    converted by Gradio; extension checks alone are never sufficient.
    """
    if filepath is None or not isinstance(filepath, (str, Path)):
        raise AudioInputError("Graba audio o carga un archivo antes de transcribir.")
    path = Path(filepath)
    if path.suffix.casefold() != ".wav":
        raise AudioInputError("Solo se admite WAV PCM tras la conversión de Gradio.")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AudioInputError("El archivo de audio no está disponible.") from exc
    if not 44 <= size <= MAX_AUDIO_BYTES:
        raise AudioInputError("Audio inválido o mayor que 12 MiB.")
    try:
        with wave.open(str(path), "rb") as stream:
            rate, channels, width = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
            frames = stream.getnframes()
            if stream.getcomptype() != "NONE" or channels not in (1, 2) or width != 2:
                raise AudioInputError("Se requiere WAV PCM de 16 bits, mono o estéreo.")
            if not 8000 <= rate <= 48000:
                raise AudioInputError("Frecuencia de audio fuera del rango 8–48 kHz.")
            duration = frames / rate
            if not MIN_AUDIO_SECONDS <= duration <= MAX_AUDIO_SECONDS:
                raise AudioInputError("La grabación debe durar entre 0,25 y 120 segundos.")
            # Inspect samples, not only headers; reject completely digital-silent audio.
            nonzero = False
            while True:
                chunk = stream.readframes(8192)
                if not chunk:
                    break
                samples = np.frombuffer(chunk, dtype="<i2")
                if np.any(samples != 0):
                    nonzero = True
                    break
            if not nonzero:
                raise AudioInputError("El audio es silencio digital; graba una frase audible.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return AudioInfo(digest.hexdigest(), size, round(duration, 3), rate, channels)
    except (EOFError, wave.Error, OSError) as exc:
        raise AudioInputError("Archivo WAV ilegible, truncado o corrupto.") from exc


def _select_asr_model() -> str:
    local = os.environ.get("TRAVELER_ASR_MODEL_PATH", "").strip()
    if local:
        path = Path(local).expanduser()
        if not path.is_dir():
            raise AudioInputError("TRAVELER_ASR_MODEL_PATH debe apuntar a una carpeta local válida.")
        return str(path)
    name = os.environ.get("TRAVELER_ASR_MODEL", "tiny")
    if name not in SUPPORTED_MODELS:
        raise AudioInputError("Modelo ASR no permitido. Usa tiny, base o small.")
    return name


@lru_cache(maxsize=1)
def _load_local_backend(model_id: str, allow_download: bool):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AudioInputError(
            'ASR opcional no instalado. Ejecuta: python -m pip install -e ".[asr]"'
        ) from exc
    try:
        return WhisperModel(
            model_id, device="cpu", compute_type="int8", cpu_threads=2,
            num_workers=1, local_files_only=not allow_download,
        )
    except Exception as exc:
        raise AudioInputError(
            "No se pudo cargar el modelo ASR local. Descarga inicial solo con "
            "TRAVELER_ASR_ALLOW_DOWNLOAD=1 y conexión autorizada, o define "
            "TRAVELER_ASR_MODEL_PATH. Verifica memoria y compatibilidad CPU."
        ) from exc


def recognize_audio(
    filepath: str | Path | None, language: str = "es", *, backend: Any = None
) -> tuple[str, dict]:
    """Return transcription and provenance. Backend injection enables offline tests."""
    if language not in LANGUAGES:
        raise AudioInputError("Idioma no admitido: selecciona es, en, it o auto.")
    info = validate_wav(filepath)
    model_id = _select_asr_model() if backend is None else "backend_simulado"
    allow_download = os.environ.get("TRAVELER_ASR_ALLOW_DOWNLOAD") == "1"
    engine = backend if backend is not None else _load_local_backend(model_id, allow_download)
    try:
        segments, metadata = engine.transcribe(
            str(filepath), language=None if language == "auto" else language,
            beam_size=1, vad_filter=True,
        )
        # faster-whisper returns a generator; iterate to execute recognition.
        recognized = " ".join(str(part.text).strip() for part in segments).strip()
    except Exception as exc:
        raise AudioInputError(
            "Falló el reconocimiento local. Revisa calidad de la voz, formato y recursos."
        ) from exc
    if not recognized:
        raise AudioInputError(
            "No se reconoció habla. Prueba con voz más clara, menos ruido o revisa el idioma."
        )
    if len(recognized) > 5000:
        raise AudioInputError("Transcripción superior al límite de 5000 caracteres.")
    detected = getattr(metadata, "language", None)
    if detected not in ("es", "en", "it"):
        detected = "desconocido"
    record = {
        "modalidad": "audio",
        "formato_procesado": "wav_pcm16",
        "audio_sha256": info.sha256,
        "tamano_bytes": info.size_bytes,
        "duracion_segundos": info.duration_seconds,
        "frecuencia_hz": info.sample_rate,
        "canales": info.channels,
        "idioma_solicitado": language,
        "idioma_detectado_asr": detected,
        "motor_asr": "faster-whisper" if backend is None else "ASR_SIMULADO_TEST",
        "modelo_asr": model_id,
        "transcripcion_original": recognized,
        "advertencia": "ASR puede omitir negaciones o confundir habla con ruido; corregir antes de analizar.",
    }
    # Never persist raw audio or local file paths in the trace.
    return recognized, record


def analyze_edited_transcript(
    edited_text: str, provenance: dict | None, classifier: Any, *, model_kind: str = "lr"
) -> dict:
    """Pass corrected transcript to the existing agent, preserving ASR trace."""
    if not provenance or provenance.get("modalidad") != "audio":
        raise AudioInputError("Transcribe un audio antes de analizarlo.")
    if not isinstance(edited_text, str) or not edited_text.strip():
        raise AudioInputError("La transcripción editada no puede estar vacía.")
    if len(edited_text) > 5000:
        raise AudioInputError("Máximo 5000 caracteres en la transcripción.")
    original = provenance.get("transcripcion_original")
    if not isinstance(original, str) or not original.strip():
        raise AudioInputError("No existe transcripción ASR original verificable.")
    from .agent import inspect_review
    record = inspect_review(edited_text, classifier)
    record["modelo_utilizado"] = model_kind
    record["procedencia_entrada"] = {
        **provenance,
        "transcripcion_editada": edited_text,
        "fue_corregida": edited_text != original,
        "texto_analizado": edited_text,
    }
    return record
