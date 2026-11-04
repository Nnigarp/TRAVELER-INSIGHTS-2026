"""Execution and backward-compatibility checks; not benchmark results."""
from pathlib import Path
import json
import numpy as np
import pytest

from traveler_agent.agent import inspect_review
from traveler_agent.batch import analyze_batch
from traveler_agent.demo_data import DEMO
from traveler_agent.model import (
    MODEL_KINDS, SentimentClassifier, build_pipeline,
    load_model, save_model, train,
)
from traveler_agent.labels import LABELS


@pytest.fixture(scope="module")
def training():
    return [x for x, _ in DEMO], [y for _, y in DEMO]


@pytest.mark.parametrize("kind", ["lr", "svm"])
def test_train_inference_share_model_interface(kind, training):
    xs, ys = training
    model = train(xs, ys, model_kind=kind)
    assert isinstance(model, SentimentClassifier)
    assert set(model.classes_) == set(LABELS)
    p = model.predict_proba(["La visita duró dos horas.", "Me encantó el hotel."])
    assert p.shape == (2, 3)
    assert np.all(np.isfinite(p)) and np.all(p >= 0) and np.all(p <= 1)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
    record = inspect_review("El precio fue excesivo.", model)
    assert record["sentimiento"] in LABELS
    batch = analyze_batch(["El tour estuvo muy bien."], model, model_kind=kind)
    assert batch[0]["modelo_utilizado"] == kind


def test_default_is_still_logistic_regression(training):
    xs, ys = training
    model = train(xs, ys)
    assert model.named_steps["classifier"].__class__.__name__ == "LogisticRegression"
    assert "tfidf" in build_pipeline().named_steps


def test_svm_rejects_too_few_per_class():
    with pytest.raises(ValueError, match="al menos 3"):
        train(["a", "b", "c"], list(LABELS), model_kind="svm")


def test_unknown_model_rejected(training):
    xs, ys = training
    with pytest.raises(ValueError, match="desconocido"):
        train(xs, ys, model_kind="transformer")


def test_svm_artifact_roundtrip_records_method(tmp_path: Path, training):
    xs, ys = training
    model = train(xs, ys, model_kind="svm")
    path = tmp_path / "svm.joblib"
    save_model(model, path, provenance="SINTETICO_DEMO_NO_PUBLICABLE", model_kind="svm")
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["model_kind"] == "svm"
    loaded = load_model(path)
    assert list(loaded.predict(["La visita fue interesante."])) == list(model.predict(["La visita fue interesante."]))


def test_svm_calibration_refits_full_vectorizer_inside_each_fold():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    estimator = build_pipeline("svm")
    assert isinstance(estimator, CalibratedClassifierCV)
    assert estimator.cv == 3
    assert isinstance(estimator.estimator, Pipeline)
    assert "tfidf" in estimator.estimator.named_steps
    assert estimator.estimator.named_steps["classifier"].__class__.__name__ == "LinearSVC"
