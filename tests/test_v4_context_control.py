import pytest
from traveler_agent.multiagent import ControlAgent, Message


@pytest.mark.parametrize("balance,expected", [
    ("positivo", False), ("negativo", True), ("incierto", True), ("no_disponible", True),
])
def test_auxiliary_verification_only_escalates(balance, expected):
    semantic = dict(sentiment="positivo", confidence_raw=.99, contrast=True)
    result = ControlAgent(.56).handle(Message("orquestador", "control", "control",
                                              {"semantic": semantic, "balance_qwen": balance}))
    assert result["needs_human"] is expected
    assert semantic["sentiment"] == "positivo"


def test_truncation_requires_review_even_with_unanimous_positive_scores():
    semantic = dict(sentiment="positivo", confidence_raw=1., contrast=False, truncated=True)
    result = ControlAgent(.56).handle(Message("orquestador", "control", "control", {"semantic": semantic}))
    assert result["needs_human"]
    assert "TEXTO_TRUNCADO_POR_ENCODER" in result["reason_codes"]
