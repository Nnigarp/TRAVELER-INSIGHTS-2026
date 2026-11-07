"""Offline calibration and selective-risk research; never changes DEMO policy."""
from __future__ import annotations

import numpy as np

from .corpus import CorpusError
from .labels import LABELS
from .research_metrics import summarize_predictions


def valid_probabilities(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1., atol=1e-6)):
        raise CorpusError("Probabilidades finitas de tres clases requeridas.")
    return values


def temperature_transform(probabilities, temperature: float) -> np.ndarray:
    values = valid_probabilities(probabilities)
    if not np.isfinite(temperature) or temperature <= 0:
        raise CorpusError("Temperatura positiva requerida.")
    logits = np.log(np.clip(values, 1e-15, 1.)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _targets(rows) -> np.ndarray:
    try:
        return np.asarray([LABELS.index(row["intended_label"]) for row in rows], dtype=int)
    except (ValueError, KeyError) as exc:
        raise CorpusError("Se requieren intenciones de autor canónicas, no ambiguos.") from exc


def probability_metrics(rows, probabilities, bins=5) -> dict:
    scores = valid_probabilities(probabilities)
    targets = _targets(rows)
    if len(rows) != len(scores) or not 2 <= bins <= 20:
        raise CorpusError("Longitudes o número de bins incorrectos.")
    onehot = np.eye(3)[targets]
    confidence = scores.max(axis=1)
    correct = (scores.argmax(axis=1) == targets).astype(float)

    def reliability(predicted, observed):
        indices = np.minimum((predicted * bins).astype(int), bins - 1)
        result = []
        for i in range(bins):
            selected = indices == i
            n = int(selected.sum())
            result.append({"lower": i / bins, "upper": (i + 1) / bins, "n": n,
                           "mean_probability": float(predicted[selected].mean()) if n else None,
                           "observed_frequency": float(observed[selected].mean()) if n else None})
        return result

    top = reliability(confidence, correct)
    ece = sum(b["n"] / len(rows) * abs(b["mean_probability"] - b["observed_frequency"]) for b in top if b["n"])
    return {"n": len(rows), "nll": float(-np.log(np.clip(scores[np.arange(len(rows)), targets], 1e-15, 1.)).mean()),
            "multiclass_brier_sum": float(np.square(scores - onehot).sum(axis=1).mean()),
            "top_confidence_ece": float(ece), "top_confidence_bins": top,
            "per_class_reliability": {label: reliability(scores[:, i], onehot[:, i]) for i, label in enumerate(LABELS)},
            "limits": "Small synthetic sample; NLL/Brier combine reliability, discrimination and uncertainty; ECE depends on fixed bins."}


def fit_temperature(rows, probabilities, protocol) -> dict:
    if not rows or any(r["role"] != "calibration" for r in rows):
        raise CorpusError("Temperatura solo se ajusta en calibration independiente.")
    attempts = []
    for temperature in protocol["calibration"]["temperatures"]:
        metrics = probability_metrics(rows, temperature_transform(probabilities, temperature), protocol["calibration"]["ece_bins"])
        attempts.append({"temperature": temperature, "fit_nll": metrics["nll"]})
    selected = min(range(len(attempts)), key=lambda i: (attempts[i]["fit_nll"], i))
    return {"temperature": attempts[selected]["temperature"], "candidates": attempts,
            "fitted_on": "calibration only; fit NLL is not an unbiased evaluation"}


def policy_review(probabilities, specification: dict) -> np.ndarray:
    scores = valid_probabilities(probabilities)
    if specification["kind"] == "REVIEW_ALL":
        return np.ones(len(scores), dtype=bool)
    if specification["kind"] != "ASYMMETRIC_CONFIDENCE_MARGIN":
        raise CorpusError("Política de investigación desconocida.")
    threshold = specification["global_threshold"]
    positive = specification["positive_threshold"]
    margin = specification["margin"]
    if not all(np.isfinite(x) for x in (threshold, positive, margin)) or min(threshold, positive, margin) < 0:
        raise CorpusError("Parámetros de política inválidos.")
    predicted = scores.argmax(axis=1)
    required = np.where(predicted == LABELS.index("positivo"), max(threshold, positive), threshold)
    ordered = np.sort(scores, axis=1)
    return (scores.max(axis=1) < required) | ((ordered[:, -1] - ordered[:, -2]) < margin)


def policy_summary(rows, probabilities, specification) -> dict:
    scores = valid_probabilities(probabilities)
    reviewed = policy_review(scores, specification)
    return summarize_predictions([r["intended_label"] for r in rows],
                                 np.asarray(LABELS)[scores.argmax(axis=1)], reviewed=reviewed)


def select_policy(rows, probabilities, ambiguity_rows, ambiguity_probabilities, protocol) -> dict:
    if (not rows or any(r["role"] != "policy_dev" for r in rows)
            or not ambiguity_rows or any(r["role"] != "ambiguity_policy" for r in ambiguity_rows)):
        raise CorpusError("Selección de revisión solo con policy_dev y ambiguity_policy.")
    scores, ambiguous = valid_probabilities(probabilities), valid_probabilities(ambiguity_probabilities)
    if len(scores) != len(rows) or len(ambiguous) != len(ambiguity_rows):
        raise CorpusError("Cohortes y probabilidades desalineadas.")
    config = protocol["review_policy"]
    candidates = []
    for global_threshold in config["global_thresholds"]:
        for positive_threshold in config["positive_thresholds"]:
            specification = {"kind": "ASYMMETRIC_CONFIDENCE_MARGIN", "global_threshold": global_threshold,
                             "positive_threshold": positive_threshold, "margin": config["margin"]}
            metrics = policy_summary(rows, scores, specification)["abstention"]
            ambiguity_rate = float(policy_review(ambiguous, specification).mean())
            eligible = (metrics["accepted"] >= config["min_accepted"]
                        and metrics["coverage"] >= config["min_coverage"]
                        and metrics["error_risk_among_accepted"] is not None
                        and metrics["error_risk_among_accepted"] <= config["max_empirical_risk"]
                        and metrics["automatic_negative_to_positive_count"] <= config["max_automatic_negative_to_positive"]
                        and ambiguity_rate >= config["min_ambiguity_review_rate"])
            candidates.append({"specification": specification, "development": metrics,
                               "ambiguity_review_rate": ambiguity_rate, "eligible": bool(eligible)})
    eligible = [i for i, row in enumerate(candidates) if row["eligible"]]
    if eligible:
        best = max(eligible, key=lambda i: (candidates[i]["development"]["coverage"],
                                          -candidates[i]["development"]["error_risk_among_accepted"], -i))
        selected, status = candidates[best]["specification"], "FEASIBLE_ON_SMALL_SYNTHETIC_DEVELOPMENT_ONLY"
    else:
        selected, status = {"kind": "REVIEW_ALL"}, "NO_FEASIBLE_DEVELOPMENT_POLICY_REVIEW_ALL"
    return {"selected": selected, "status": status, "candidates": candidates,
            "development": policy_summary(rows, scores, selected)["abstention"],
            "ambiguity_review_rate": float(policy_review(ambiguous, selected).mean()),
            "guarantee": "NONE: empirical constraints on author-intent development, not a population guarantee"}


def fixed_threshold_curve(rows, probabilities, thresholds) -> list[dict]:
    """Descriptive predeclared curve; never returns a winning assessment threshold."""
    result = []
    for threshold in thresholds:
        specification = {"kind": "ASYMMETRIC_CONFIDENCE_MARGIN", "global_threshold": threshold,
                         "positive_threshold": threshold, "margin": 0.}
        result.append({"threshold": threshold, **policy_summary(rows, probabilities, specification)["abstention"]})
    return result
