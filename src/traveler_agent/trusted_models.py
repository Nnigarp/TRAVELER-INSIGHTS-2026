"""Explicitly trusted, checksum-verified loading of own benchmark artifacts.

Checksum detects file modifications, NOT malicious artifacts: both model and
manifest can be replaced by an adversary. NEVER load untrusted joblib files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .corpus import CorpusError, file_sha
from .labels import LABELS

_ALLOWED_KINDS = frozenset({"dummy", "lr", "svm"})


def load_own_benchmark_model(
    output_dir: Path, model_kind: str, *, trust_local_artifacts: bool = False
) -> Any:
    """Trust must be explicitly granted; otherwise refuse pickle execution."""
    if not trust_local_artifacts:
        raise CorpusError(
            "La deserialización joblib puede ejecutar código. "
            "Solo cargar archivos creados por ti en un directorio confiable."
        )
    if model_kind not in _ALLOWED_KINDS:
        raise CorpusError("Modelo no admitido.")
    directory = Path(output_dir)
    manifest_path = directory / "model_manifest.json"
    model_path = directory / f"{model_kind}.joblib"
    if not manifest_path.is_file() or not model_path.is_file():
        raise CorpusError("Faltan archivos del modelo o manifiesto.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise CorpusError("Manifiesto inválido.") from exc
    metadata = manifest.get("model_metadata", {}).get(model_kind)
    if not isinstance(metadata, dict):
        raise CorpusError("Modelo no registrado en manifiesto.")
    if file_sha(model_path) != metadata.get("sha256"):
        raise CorpusError("SHA-256 del artefacto no coincide con el manifiesto.")
    if set(metadata.get("classes", [])) != set(LABELS):
        raise CorpusError("El modelo no declara las tres clases requeridas.")
    model = joblib.load(model_path)
    if not all(hasattr(model, name) for name in ("predict", "predict_proba", "classes_")):
        raise CorpusError("Interfaz del estimador no compatible con el agente.")
    if set(model.classes_) != set(LABELS):
        raise CorpusError("Las etiquetas del estimador no coinciden con las esperadas.")
    return model
