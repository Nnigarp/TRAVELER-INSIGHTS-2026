"""Metrics on saved research predictions, with explicit abstention denominators."""
from __future__ import annotations

import re

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .corpus import CorpusError
from .labels import LABELS


def score_matrix(model, texts: list[str]) -> np.ndarray:
    classes = [str(c) for c in model.classes_]
    if set(classes) != set(LABELS) or len(classes) != 3:
        raise CorpusError("El estimador debe tener las tres clases canónicas.")
    values = np.asarray(model.predict_proba(texts), dtype=float)
    if (values.shape != (len(texts), 3) or not np.isfinite(values).all()
            or (values < 0).any() or not np.allclose(values.sum(axis=1), 1., atol=1e-6)):
        raise CorpusError("Puntuaciones del modelo inválidas.")
    return values[:, [classes.index(label) for label in LABELS]]


def review_mask(scores: np.ndarray, texts: list[str]) -> np.ndarray:
    confidence = scores.max(axis=1)
    contrast = np.asarray([bool(re.search(r"\b(pero|aunque|sin embargo)\b", t, re.I)) for t in texts])
    return (confidence < .56) | (contrast & (confidence < .80))


def summarize_predictions(truth, prediction, *, reviewed=None) -> dict:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if (truth.ndim != 1 or prediction.shape != truth.shape or not len(truth)
            or not set(truth) <= set(LABELS) or not set(prediction) <= set(LABELS)):
        raise CorpusError("Vectores de verdad prevista/predicción inválidos o vacíos.")
    negatives = truth == "negativo"
    neg_n = int(negatives.sum())
    neg_pos = int(((prediction == "positivo") & negatives).sum())
    neg_neutral = int(((prediction == "neutral") & negatives).sum())
    counts = {label: int((truth == label).sum()) for label in LABELS}
    result = {
        "n": len(truth), "support": counts,
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
        "three_class_macro_comparable": all(counts.values()),
        "accuracy": float(accuracy_score(truth, prediction)),
        "per_class": classification_report(truth, prediction, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=LABELS).tolist(),
        "negative_to_positive_count": neg_pos,
        "negative_to_positive_rate": neg_pos / neg_n if neg_n else None,
        "negative_to_neutral_count": neg_neutral,
        "negative_to_neutral_rate": neg_neutral / neg_n if neg_n else None,
    }
    if reviewed is not None:
        reviewed = np.asarray(reviewed, dtype=bool)
        if reviewed.shape != truth.shape:
            raise CorpusError("Máscara de revisión de longitud incorrecta.")
        accepted = ~reviewed
        accepted_n = int(accepted.sum())
        dangerous = int((accepted & negatives & (prediction == "positivo")).sum())
        result["abstention"] = {
            "accepted": accepted_n, "referred": int(reviewed.sum()),
            "coverage": accepted_n / len(truth), "review_rate": float(reviewed.mean()),
            "error_risk_among_accepted": float((truth[accepted] != prediction[accepted]).mean()) if accepted_n else None,
            "automatic_negative_to_positive_count": dangerous,
            "automatic_negative_to_positive_rate_over_all_negatives": dangerous / neg_n if neg_n else None,
        }
    return result


def slice_metrics(rows, truth, prediction, reviewed) -> dict:
    tags = sorted({tag for row in rows for tag in row.get("phenomena", [])})
    selections = {tag: [i for i, row in enumerate(rows) if tag in row.get("phenomena", [])] for tag in tags}
    for field in ("region", "length_band"):
        for value in sorted({r[field] for r in rows}):
            selections[f"{field}:{value}"] = [i for i, row in enumerate(rows) if row[field] == value]
    truth, prediction, reviewed = np.asarray(truth), np.asarray(prediction), np.asarray(reviewed)
    return {name: summarize_predictions(truth[ids], prediction[ids], reviewed=reviewed[ids])
            for name, ids in selections.items() if ids}


def family_confusions(rows, truth, prediction) -> tuple[list[str], np.ndarray]:
    if not len(rows) or len(rows) != len(truth) or len(rows) != len(prediction):
        raise CorpusError("Predicciones y familias no están alineadas.")
    groups = sorted({r["family"] for r in rows})
    positions = {g: i for i, g in enumerate(groups)}
    labels = {name: i for i, name in enumerate(LABELS)}
    cubes = np.zeros((len(groups), 3, 3), dtype=int)
    for row, target, guessed in zip(rows, truth, prediction):
        if target not in labels or guessed not in labels:
            raise CorpusError("Clase desconocida en remuestreo.")
        cubes[positions[row["family"]], labels[target], labels[guessed]] += 1
    return groups, cubes


def _macro_from_cm(matrix: np.ndarray) -> np.ndarray:
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-2) + matrix.sum(axis=-1)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0).mean(axis=-1)


def bootstrap_indices(groups: int, resamples: int, seed: int) -> np.ndarray:
    if groups < 2 or not 1 <= resamples <= 10000:
        raise CorpusError("Remuestreo requiere al menos dos familias y un presupuesto acotado.")
    return np.random.default_rng(seed).integers(0, groups, size=(resamples, groups))


def bootstrap_macro(cubes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return _macro_from_cm(cubes[indices].sum(axis=1))


def interval(values: np.ndarray) -> dict:
    low, high = np.percentile(values, [2.5, 97.5])
    return {"low_95": float(low), "high_95": float(high),
            "scope": "paired synthetic family bootstrap, not a human population interval"}
