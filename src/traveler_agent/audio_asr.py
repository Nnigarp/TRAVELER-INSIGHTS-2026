"""Local-only ASR adapter for Gradio-converted PCM WAV audio.

Only the optional backend knows about faster-whisper. Importing this module,
running pytest, and using text analysis never downloads or loads ASR weights.
"""
from __future__ import annotations

import hashlib
import os
import wave
import logging
import traceback
import sys
import importlib.metadata as pkg_metadata
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

_LOG = logging.getLogger("traveler_agent.audio_asr")
_LOG.setLevel(logging.ERROR)
if not _LOG.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    _LOG.addHandler(_handler)
_LOG.propagate = False


def _safe_trace(exc: BaseException) -> str:
    """Stack locations and exception types; never paths, source lines or args."""
    lines = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        lines.append("exception_class=" + type(exc).__module__ + "." + type(exc).__name__)
        if exc.__traceback__:
            for frame in traceback.extract_tb(exc.__traceback__):
                # Do not log absolute paths, filenames supplied by users or source code.
                lines.append("  function=%s line=%d" % (frame.name, frame.lineno))
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return "\n".join(lines)


def _diagnose(stage: str, exc: BaseException) -> None:
    # Only OS error codes and module identifiers are safe to disclose from args.
    winerror = getattr(exc, "winerror", None)
    errno = getattr(exc, "errno", None)
    module = getattr(exc, "name", None) if isinstance(exc, ImportError) else None
    module = module if isinstance(module, str) and module.replace("_", "").replace(".", "").isalnum() else None
    _LOG.error("ASR_STAGE=%s ERROR_TYPE=%s WINERROR=%s ERRNO=%s MODULE=%s\n%s",
               stage, type(exc).__name__, winerror, errno, module,
               _safe_trace(exc))


def _inference_failure_stage(exc: BaseException) -> str:
    # Classification by exception origin, never by the contents of private text.
    current = exc
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        module = type(current).__module__.lower()
        if module.startswith("ctranslate2"):
            return "ctranslate2"
        if module.startswith("onnxruntime"):
            return "vad_inference"
        current = current.__cause__ if current.__cause__ else current.__context__
    return "inference"


def dependency_versions() -> dict[str, str]:
    names = ("faster-whisper", "ctranslate2", "av", "onnxruntime",
             "tokenizers", "huggingface-hub", "numpy", "gradio")
    found = {}
    for name in names:
        try:
            found[name] = pkg_metadata.version(name)
        except pkg_metadata.PackageNotFoundError:
            found[name] = "NO_INSTALADO"
    return found


def _decode_local_audio(filepath):
    """Separate PyAV decoding from deferred model inference."""
    try:
        from faster_whisper.audio import decode_audio
    except (ImportError, OSError) as exc:
        _diagnose("model_import", exc)
        raise AudioInputError("Fallo al importar PyAV/faster-whisper (model_import).") from None
    try:
        samples = decode_audio(str(filepath), sampling_rate=16000)
        if samples.size == 0:
            raise ValueError("Empty decoded PCM")
        return samples
    except Exception as exc:
        _diagnose("decode", exc)
        raise AudioInputError("Error en decodificación PyAV (decode). Consulta terminal.") from None



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
            if size < 44 + frames * channels * width:
                raise AudioInputError("El WAV está truncado: no contiene todas las muestras declaradas.")
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
    except (ImportError, OSError) as exc:
        _diagnose("model_import", exc)
        raise AudioInputError("No se importó el backend ASR (model_import). Verifica instalación.") from None
    try:
        return WhisperModel(
            model_id, device="cpu", compute_type="int8", cpu_threads=2,
            num_workers=1, local_files_only=not allow_download,
        )
    except Exception as exc:
        _diagnose("model_load", exc)
        raise AudioInputError(
            "Falló la carga del modelo ASR (model_load). Consulta clase y traza "
            "en terminal; verifica caché, modelo descargado y CPU."
        ) from None


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
    # PyAV decoding is separate only for the real backend. Test mocks
    # keep their original filepath contract and require no optional packages.
    audio_input = _decode_local_audio(filepath) if backend is None else str(filepath)
    try:
        segments, metadata = engine.transcribe(
            audio_input, language=None if language == "auto" else language,
            beam_size=1, vad_filter=True,
        )
        # Inference may fail during the call OR the lazy generator iteration.
        recognized = " ".join(str(part.text).strip() for part in segments).strip()
    except Exception as exc:
        stage = _inference_failure_stage(exc)
        _diagnose(stage, exc)
        raise AudioInputError(
            "Falló la inferencia ASR (" + stage + "). Consulta terminal; "
            "no se registra audio ni transcripción."
        ) from None
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
    record = inspect_review(
        edited_text, classifier, modality="audio_transcrito",
        edited_by_human=(edited_text != original),
    )
    record["modelo_utilizado"] = model_kind
    record["procedencia_entrada"] = {
        **provenance,
        "transcripcion_editada": edited_text,
        "fue_corregida": edited_text != original,
        "texto_analizado": edited_text,
    }
    return record


def main(argv=None):
    """Metadata-only Windows diagnostics; no weights, audio or network."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Diagnóstico local ASR sin audio")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return
    report = {"python": sys.version.split()[0], "platform": sys.platform,
              "dependencies": dependency_versions()}
    try:
        import ctranslate2
        report["cpu_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cpu"))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        _diagnose("model_import", exc)
        report["cpu_compute_types"] = "NO_DISPONIBLE"
    try:
        import av
        report["pyav_import"] = "OK"
    except (ImportError, OSError) as exc:
        _diagnose("model_import", exc)
        report["pyav_import"] = "ERROR"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
