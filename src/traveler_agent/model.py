"""Modelo clásico reproducible: TF-IDF de palabras+caracteres y regresión logística."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .labels import LABELS

def build_pipeline():
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1,2), strip_accents="unicode", sublinear_tf=True, min_df=1)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True, max_features=12000)),
        ])),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ])

def train(texts, labels):
    if len(texts) != len(labels) or not texts:
        raise ValueError("Se requieren textos y etiquetas no vacíos con idéntica longitud.")
    if set(labels) != set(LABELS):
        raise ValueError(f"Se requieren las tres clases {LABELS} presentes.")
    model = build_pipeline()
    model.fit(texts, labels)
    return model

def save_model(model, path: Path, *, provenance: str, dataset_sha256: str | None = None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    metadata = {"provenance":provenance,"dataset_sha256": dataset_sha256,
                "labels":list(map(str, model.classes_)),"method":"tfidf_words_chars+logistic_regression",
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
