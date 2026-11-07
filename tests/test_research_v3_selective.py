import json
from copy import deepcopy

import numpy as np
import pytest

from traveler_agent.corpus import CorpusError
from traveler_agent.labels import LABELS
from traveler_agent.research_v3_data import PROTOCOL
from traveler_agent.research_v3_selective import (temperature_transform, probability_metrics,
    fit_temperature, select_policy, policy_review, fixed_threshold_curve)


def protocol(): return json.loads(PROTOCOL.read_text())


def test_probability_scores_have_known_nll_brier_ece_and_empty_bins():
    rows = [{"intended_label": "negativo"}, {"intended_label": "neutral"}]
    metrics = probability_metrics(rows, [[1., 0, 0], [.5, .5, 0]])
    assert metrics["nll"] == pytest.approx(np.log(2) / 2)
    assert metrics["multiclass_brier_sum"] == pytest.approx(.25)
    assert metrics["top_confidence_ece"] == pytest.approx(.25)
    assert any(b["n"] == 0 and b["observed_frequency"] is None for b in metrics["top_confidence_bins"])
    assert sum(b["n"] for b in metrics["per_class_reliability"]["positivo"]) == 2


def test_temperature_preserves_predictions_normalizes_and_fits_only_calibration():
    scores = np.asarray([[.6, .3, .1], [.05, .7, .25], [.1, .2, .7]])
    rows = [{"role": "calibration", "intended_label": label} for label in LABELS]
    for t in (.5, 1., 3.):
        transformed = temperature_transform(scores, t)
        np.testing.assert_array_equal(scores.argmax(axis=1), transformed.argmax(axis=1))
        np.testing.assert_allclose(transformed.sum(axis=1), 1)
    assert fit_temperature(rows, scores, protocol())["temperature"] == .5
    with pytest.raises(CorpusError, match="calibration"):
        fit_temperature([{**r, "role": "assessment"} for r in rows], scores, protocol())


@pytest.mark.parametrize("scores,t", [([[float('nan'),0,1]],1), ([[.2,.2,.2]],1), ([[1,0,0]],0), ([[1,0,0]],-1)])
def test_rejects_invalid_probabilities_and_temperature(scores, t):
    with pytest.raises(CorpusError): temperature_transform(scores, t)


def test_positive_threshold_is_stricter_without_forcing_negative_or_neutral_review():
    spec = {"kind": "ASYMMETRIC_CONFIDENCE_MARGIN", "global_threshold": .56, "positive_threshold": .9, "margin": .1}
    assert policy_review([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]], spec).tolist() == [False, False, True]


def test_development_selection_protects_negative_to_positive_and_reviews_ambiguity():
    rows = [{"role": "policy_dev", "intended_label": LABELS[i % 3]} for i in range(30)]
    scores = np.full((30,3), .01)
    for i in range(30): scores[i,i%3] = .98
    scores[0] = [.1,.05,.85]  # high-confidence negative -> positive; dev, not assessment
    ambiguity = [{"role": "ambiguity_policy", "intended_label": None}] * 10
    result = select_policy(rows, scores, ambiguity, np.full((10,3),1/3), protocol())
    assert result["status"].startswith("FEASIBLE")
    assert result["development"]["automatic_negative_to_positive_count"] == 0
    assert result["development"]["accepted"] == 29
    assert result["ambiguity_review_rate"] == 1.
    with pytest.raises(CorpusError, match="policy_dev"):
        select_policy([{**r,"role":"assessment"} for r in rows], scores, ambiguity, np.full((10,3),1/3), protocol())


def test_no_feasible_policy_falls_back_to_all_review_with_undefined_risk():
    rows = [{"role": "policy_dev", "intended_label": LABELS[i%3]} for i in range(30)]
    result = select_policy(rows, np.full((30,3),1/3), [{"role":"ambiguity_policy"}]*10, np.full((10,3),1/3), protocol())
    assert result["selected"] == {"kind": "REVIEW_ALL"}
    assert result["development"]["accepted"] == 0
    assert result["development"]["error_risk_among_accepted"] is None


def test_fixed_curve_is_descriptive_and_coverage_is_monotone():
    rows = [{"intended_label": label} for label in LABELS]
    curve = fixed_threshold_curve(rows, [[.9,.05,.05],[.2,.6,.2],[.1,.1,.8]], [.5,.7,.95,1.01])
    assert [r["accepted"] for r in curve] == [3,2,0,0]
    assert curve[-1]["error_risk_among_accepted"] is None
    assert all("selected" not in point for point in curve)
