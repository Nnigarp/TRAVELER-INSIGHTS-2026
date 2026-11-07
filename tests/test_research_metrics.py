import numpy as np
import pytest

from traveler_agent.corpus import CorpusError
from traveler_agent.multiagent import ControlAgent, Message
from traveler_agent.research_metrics import (
    bootstrap_indices, bootstrap_macro, family_confusions, interval,
    review_mask, score_matrix, summarize_predictions,
)


def test_negative_errors_and_zero_coverage_are_not_reported_as_zero_risk():
    metrics = summarize_predictions(["negativo", "negativo", "positivo"],
                                    ["positivo", "negativo", "positivo"], reviewed=[True] * 3)
    assert metrics["negative_to_positive_count"] == 1
    assert metrics["negative_to_positive_rate"] == .5
    assert metrics["three_class_macro_comparable"] is False
    assert metrics["abstention"]["coverage"] == 0
    assert metrics["abstention"]["error_risk_among_accepted"] is None


def test_absent_negative_support_produces_null_rate():
    assert summarize_predictions(["positivo"], ["positivo"])["negative_to_positive_rate"] is None


def test_score_columns_are_aligned_by_class_not_column_position():
    class Model:
        classes_ = ["positivo", "negativo", "neutral"]
        def predict_proba(self, texts):
            return [[.7, .1, .2] for _ in texts]
    assert score_matrix(Model(), ["fixture"]).tolist() == [[.1, .2, .7]]


@pytest.mark.parametrize("values", [[[float("nan"), 0, 1]], [[.2, .2, .2]], [[-.1, .2, .9]]])
def test_invalid_scores_fail_closed(values):
    class Model:
        classes_ = ["negativo", "neutral", "positivo"]
        def predict_proba(self, texts):
            return values
    with pytest.raises(CorpusError):
        score_matrix(Model(), ["fixture"])


def test_review_policy_matches_unchanged_control_agent_at_boundaries():
    texts = ["Una descripción", "Fue breve, pero útil", "Aunque breve, sirve", "Sin embargo, funciona"]
    matrix = np.asarray([[.56, .3, .14], [.8, .1, .1], [.7, .2, .1], [.55, .25, .2]])
    observed = review_mask(matrix, texts)
    control = ControlAgent(.56)
    for i, text in enumerate(texts):
        import re
        result = control.handle(Message("orquestador", "control", "control", {
            "semantic": {"confidence_raw": float(matrix[i].max()),
                         "contrast": bool(re.search(r"\b(pero|aunque|sin embargo)\b", text, re.I))}}))
        assert bool(observed[i]) == result["needs_human"]


def test_group_bootstrap_is_paired_and_uses_families():
    rows = [{"family": family} for family in ("a", "a", "a", "b", "b", "b")]
    truth = ["negativo", "neutral", "positivo"] * 2
    groups, perfect = family_confusions(rows, truth, truth)
    _, worse = family_confusions(rows, truth, ["positivo", "neutral", "positivo"] * 2)
    sample = bootstrap_indices(len(groups), 100, 42)
    p, q = bootstrap_macro(perfect, sample), bootstrap_macro(worse, sample)
    assert np.allclose(p, 1)
    assert interval(p - p)["low_95"] == interval(p - p)["high_95"] == 0
    assert interval(p - q)["low_95"] > 0
    assert sample.shape == (100, 2)


def test_invalid_empty_metrics_and_single_family_bootstrap_rejected():
    with pytest.raises(CorpusError):
        summarize_predictions([], [])
    with pytest.raises(CorpusError):
        bootstrap_indices(1, 100, 42)
