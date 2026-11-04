"""Modelo clásico reproducible: TF-IDF de palabras+caracteres y regresión logística."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from typing import Protocol, runtime_checkable
from .labels import LABELS

MODEL_KINDS = ("lr", "svm")

@runtime_checkable
class SentimentClassifier(Protocol):
    """Interface required by the existing agent (fitted classifiers only)."""
    classes_: object
    def predict(self, texts): ...
    def predict_proba(self, texts): ...


def _validate_kind(model_kind: str) -> str:
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"Modelo desconocido: {model_kind!r}. Opciones: {MODEL_KINDS}")
    return model_kind


def _text_features():
    return FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode",
                                   sublinear_tf=True, min_df=1)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                   sublinear_tf=True, max_features=12000)),
    ])


def build_pipeline(model_kind: str = "lr"):
    model_kind = _validate_kind(model_kind)
    if model_kind == "lr":
        return Pipeline([
            ("tfidf", _text_features()),
            ("classifier", LogisticRegression(max_iter=1500,
                                               class_weight="balanced",
                                               random_state=42)),
        ])
    # IMPORTANT: TF-IDF belongs INSIDE calibration folds. Wrapping only
    # LinearSVC would fit a shared vocabulary on all calibration data.
    return CalibratedClassifierCV(
        estimator=Pipeline([
            ("tfidf", _text_features()),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ]),
        method="sigmoid",
        cv=3,
    )

def train(texts, labels, *, model_kind: str = "lr"):
    if len(texts) != len(labels) or not texts:
        raise ValueError("Se requieren textos y etiquetas no vacíos con idéntica longitud.")
    if set(labels) != set(LABELS):
        raise ValueError(f"Se requieren las tres clases {LABELS} presentes.")
    _validate_kind(model_kind)
    if model_kind == "svm" and min(Counter(labels)[label] for label in LABELS) < 3:
        raise ValueError("SVM requiere al menos 3 ejemplos por clase para calibración interna CV=3.")
    model = build_pipeline(model_kind)
    model.fit(texts, labels)
    return model

def save_model(model, path: Path, *, provenance: str, dataset_sha256: str | None = None, model_kind: str = "lr"):
    _validate_kind(model_kind)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    metadata = {"provenance":provenance,"dataset_sha256": dataset_sha256,
                "labels":list(map(str, model.classes_)),"method": f"tfidf_words_chars+{model_kind}",
                "model_kind": model_kind,
                "score_note": "Estimaciones internas no verificadas con datos de turismo reales; no asumir fiabilidad.",
                "warning":"Artefacto joblib local: solo cargar si es propio y de origen confiable."}
    path.with_suffix(".meta.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")

def load_model(path: Path):
    path=Path(path)
    if not path.is_file() or not path.with_suffix('.meta.json').is_file():
        raise FileNotFoundError("Falta modelo o manifiesto; entrene primero.")
    # IMPORTANTE: joblib puede ejecutar código al deserializar. No cargar archivos ajenos.
    return joblib.load(path)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()
