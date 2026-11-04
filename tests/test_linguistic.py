import pytest
from traveler_agent.linguistic import (
    normalize_for_comparison, detect_language_heuristic,
    aspect_spans, contextual_signals, linguistic_profile
)
from traveler_agent.agent import inspect_review
from traveler_agent.model import train
from traveler_agent.demo_data import DEMO


def test_normalization_is_derived_without_destroying_emoji():
    text = "  NO  fue MALO 😍  "
    assert normalize_for_comparison(text) == "no fue malo 😍"
    assert text == "  NO  fue MALO 😍  "


@pytest.mark.parametrize("text,expected", [
    ("El recorrido fue muy bueno y la guía estuvo amable.", "es"),
    ("The hotel was very good but the room was terrible.", "en"),
    ("Il servizio era molto bello ma la camera non era pulita.", "it"),
    ("😍", "desconocido"),
    ("hotel", "desconocido"),
])
def test_language_heuristic_abstains_when_uncertain(text, expected):
    result = detect_language_heuristic(text)
    assert result["idioma_indicativo"] == expected
    assert result["metodo"].endswith("no_validado")


def test_evidence_offsets_exact_literal_unicode():
    text = "¡Servicio increíble 😍, pero el desayuno fue malo!"
    spans = aspect_spans(text)
    assert spans
    for span in spans:
        assert text[span["inicio"]:span["fin"]] == span["texto"]
    assert any(span["texto"].lower() == "servicio" for span in spans)
    assert any(span["texto"].lower() == "desayuno" for span in spans)


def test_negation_and_contrast_are_signals_not_sentiment():
    result = contextual_signals("No fue malo, pero tampoco perfecto 🙃")
    assert [x["texto"].lower() for x in result["negaciones"]] == ["no"]
    assert result["posible_opinion_mixta"] is True
    assert "polaridad_por_aspecto" not in result


def test_no_contrast_does_not_claim_mixed_opinion():
    assert contextual_signals("El hotel estaba limpio.")[ "posible_opinion_mixta"] is False


def test_agent_integration_preserves_existing_fields_and_spans():
    model = train([x for x, _ in DEMO], [y for _, y in DEMO])
    text = "El servicio fue bueno, pero el desayuno malo 😡"
    result = inspect_review(text, model)
    assert result["sentimiento"] in {"positivo", "neutral", "negativo"}
    assert "recomendacion" in result
    profile = result["perfil_linguistico_exploratorio"]
    assert profile["indicios_contextuales"]["posible_opinion_mixta"] is True
    assert all(text[s["inicio"]:s["fin"]] == s["texto"] for s in profile["evidencias_posicionales"])


def test_misspelling_does_not_fabricate_evidence():
    text = "El serbicio estuvo masomenos"
    assert aspect_spans(text) == []


def test_wrong_type_rejected():
    with pytest.raises(TypeError):
        linguistic_profile(None)
