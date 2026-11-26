"""Unit checks for exploratory contextual semantics (no training, no Ollama)."""
from traveler_agent.policy import recommend
from traveler_agent.semantic_context import analyze_context


def test_mixed_transport_marks_unresolved_incident_and_evidence():
    text = "El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión."
    ctx = analyze_context(text)
    assert ctx["aspects"]
    assert any(i["status"] == "no_resuelto" for i in ctx["incidents"])
    assert any("transporte" in e.casefold() or "excursión" in e.casefold() for e in ctx["evidence"])
    state, rec = recommend("negativo", ctx["aspects"], False, context=ctx)
    assert state == "sugerencia"
    assert "traslado" in rec.casefold() or "acceso" in rec.casefold() or "incidente" in rec.casefold()


def test_resolved_incident_under_positive_label():
    text = ("Al inicio la llave no funcionaba. El personal lo resolvió de inmediato "
            "y el resto de la estancia fue excelente.")
    ctx = analyze_context(text)
    assert any(i["status"] == "resuelto" for i in ctx["incidents"])
    state, rec = recommend("positivo", ctx["aspects"], False, context=ctx)
    assert state == "informativo"
    assert "resuelto" in rec.casefold() or "favorable" in rec.casefold()


def test_doubt_forces_human_recommendation_path():
    text = "No sé si volvería. El lugar estaba bien en general."
    ctx = analyze_context(text)
    assert ctx["ambiguity"]["doubt"] or ctx["ambiguity"]["ambiguous"]
    state, rec = recommend("neutral", ctx["aspects"], False, context=ctx)
    assert state == "revisión_humana"


def test_open_incident_with_positive_label_escalates_when_needs_review():
    text = "Todo amable, pero el transporte nunca llegó y perdimos la visita."
    ctx = analyze_context(text)
    state, rec = recommend("positivo", ctx["aspects"], True, context=ctx)
    assert state == "revisión_humana"
    assert "manual" in rec.casefold() or "incidente" in rec.casefold()
