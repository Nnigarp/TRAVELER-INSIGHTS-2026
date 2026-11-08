"""Explicit optional supervised model selection with auditable classical fallback."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from .corpus import file_sha


def configured_artifact():
    value = os.environ.get("TRAVELER_BETO_ARTIFACT", "")
    if not value:
        return None
    path = Path(value).resolve()
    if not any(path.is_relative_to((Path.cwd()/part).resolve()) for part in ("artifacts", "data/private")):
        return None
    return path if (path/"model_manifest.json").is_file() else None


@lru_cache(maxsize=1)
def _load(path, manifest_sha256):
    from .beto_classifier import load_classifier
    return load_classifier(Path(path))


def run_selected(kind, classical_factory, operation):
    """Run actual inference. All rows use the same effective model after retry.

    No optional imports/downloads for the default path. A BETO runtime failure
    retries the operation with LR; returned metadata explicitly names that fact.
    """
    if kind in ("lr", "svm"):
        return operation(classical_factory(kind)), kind
    if kind != "beto":
        raise ValueError("Modelo no disponible")
    attempted = configured_artifact()
    metadata = {"requested": "beto", "supervised_experimental": True,
                "human_validated": False, "review_policy_validated": False}
    try:
        if attempted is None:
            raise FileNotFoundError("No configured supervised artifact")
        digest = file_sha(attempted/"model_manifest.json")
        model = _load(str(attempted), digest)
        result = operation(model)
        effective = "beto"
        metadata.update(effective="beto", fallback=False, manifest_sha256=digest,
                        checkpoint_revision=model.metadata["revision"],
                        training_mode=model.metadata["training"]["mode"])
    except Exception as exc:
        # Optional backend isolation. Never expose input, paths or exception args.
        result = operation(classical_factory("lr"))
        effective = "lr"
        metadata.update(effective="lr", fallback=True, failure_type=type(exc).__name__)
    records = result if isinstance(result,list) else [result]
    for record in records:
        record["modelo_utilizado"] = effective
        record["seleccion_modelo"] = dict(metadata)
    return result, effective
