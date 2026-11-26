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

def test_negated_resolution_and_absent_delays():
    ctx = analyze_context("Nunca nos reembolsaron el cobro. El transporte llegó sin demoras.")
    assert [i["status"] for i in ctx["incidents"]] == ["no_resuelto"]
    assert not ctx["incidents"][0]["resolved_marker"]


def test_recommendation_follows_problem_aspect_only():
    ctx = analyze_context("La habitación estaba limpia pero el transporte nunca llegó.")
    _, rec = recommend("negativo", ctx["aspects"], False, context=ctx)
    assert "traslado" in rec and "higiene" not in rec


def test_no_unsupported_price_transport_or_resolution_facts():
    from traveler_agent.report_content import content_options
    for text in ("No nos cobraron dos veces.", "El transporte llegó y perdimos la visita.",
                 "Nadie resolvió el problema.", "No se perdió la excursión."):
        result = content_options(text, [text])
        assert result["mode"] == "CITA_CONSERVADORA" and result["problems"] == [""]

def test_duration_only_does_not_authorize_favorable_action():
    from traveler_agent.report_content import is_factual_description
    from traveler_agent.multiagent import ControlAgent, Message
    assert is_factual_description("El recorrido duró 45 minutos.")
    assert not is_factual_description("El recorrido duró 45 minutos. Fue excelente.")
    semantic = dict(confidence_raw=.99, sentiment="positivo", contrast=False, factual_description=True)
    result = ControlAgent(.56).handle(Message("orquestador", "control", "control", {"semantic": semantic}))
    assert result["needs_human"] and "DESCRIPCION_FACTUAL_VS_ETIQUETA" in result["reason_codes"]
