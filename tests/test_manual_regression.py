"""Casos cualitativos de 04-11-2026; no son evaluación de rendimiento."""
import pytest
from traveler_agent.aspects import find_aspects, evidence_spans
from traveler_agent.model import train
from traveler_agent.demo_data import DEMO
from traveler_agent.agent import inspect_review
from traveler_agent.policy import recommend

@pytest.fixture(scope="module")
def model():
    return train([t for t, _ in DEMO], [label for _, label in DEMO])

MANUAL = [
    ("Visitamos el centro histórico de Cartagena. El guía fue excelente, el recorrido estuvo muy bien organizado y la atención fue maravillosa.", "positivo"),
    ("El servicio fue terrible, esperamos demasiado tiempo y nos cobraron un precio excesivo. No recomiendo esta experiencia turística.", "negativo"),
    ("El recorrido comenzó a las nueve de la mañana, duró aproximadamente dos horas y terminó cerca de la Torre del Reloj.", "neutral"),
    ("El hotel estaba limpio y los empleados fueron amables, pero la habitación tenía mucho ruido y el desayuno fue decepcionante.", "negativo"),
]

@pytest.mark.parametrize("text,expected", MANUAL)
def test_manual_cases_remain_valid_outputs(text, expected, model):
    output = inspect_review(text, model)
    assert output["sentimiento"] in {"positivo", "neutral", "negativo"}
    assert output["revision_humana"] in (True, False)
    assert all(span in text for span in output["fragmentos_literales_no_explicativos"])
    assert "puntuaciones_modelo_no_calibradas" in output


def test_multiaspect_coverage_not_aspect_sentiment():
    text = "El hotel estaba limpio y los empleados fueron amables, pero la habitación tenía mucho ruido y el desayuno fue decepcionante."
    found = find_aspects(text)
    assert {"higiene", "servicio", "ruido", "alimentacion"} <= set(found)
    assert all(isinstance(v, list) for v in found.values())


def test_multiple_recommendations_are_not_assumed_aspect_polarity():
    status, action = recommend("negativo", {"precio": ["precio"], "espera": ["esperamos"]}, False)
    assert status == "sugerencia"
    assert "precio:" in action and "espera:" in action
    assert "polaridad por aspecto no determinada" in action


def test_reviewer_gate_preserved():
    status, action = recommend("negativo", {"precio": ["precio"], "espera": ["esperamos"]}, True)
    assert status == "revisión_humana"
    assert "Revisar manualmente" in action


@pytest.mark.xfail(strict=True, reason="Error de negación reportado en modelo DEMO; pendiente entrenamiento/evaluación con datos reales")
def test_negation_expected_positive_is_known_limitation(model):
    text = "No fue una mala experiencia. El servicio no estuvo nada mal y volvería a visitar el lugar."
    assert inspect_review(text, model)["sentimiento"] == "positivo"
