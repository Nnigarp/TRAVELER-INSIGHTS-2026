"""Regression: SemanticsAgent facts must ground Qwen report enums when coverage is partial."""
from traveler_agent.ollama_report import evidence_units, failed_generation, report_schema, validate_generation
from traveler_agent.report_content import content_options, content_provenance
from traveler_agent.semantic_context import analyze_context


CARTAGENA = (
    "El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión. "
    "Presenté una reclamación y nadie resolvió el problema."
)


def test_cartagena_partial_coverage_grounds_transport_problem():
    ctx = analyze_context(CARTAGENA)
    assert any(i["status"] == "no_resuelto" for i in ctx["incidents"])
    options = content_options(CARTAGENA, evidence_units(CARTAGENA), context=ctx)
    assert options["mode"] == "PARAFRASIS_CONTROLADA"
    assert options["complete"] is False
    assert options["grounded_partial"] is True
    assert any("transporte" in p.casefold() for p in options["problems"] if p)
    assert "transporte" in options["categories"]
    assert any("traslado" in r.casefold() or "transporte" in r.casefold() for r in options["recommendations"])
    provenance = content_provenance(
        CARTAGENA, evidence_units(CARTAGENA),
        {"problema_identificado": options["problems"][0]},
        context=ctx,
    )
    assert provenance["resumen"] == "PARAFRASIS_CONTROLADA"
    assert provenance["problema"] == "PARAFRASIS_CONTROLADA"
    assert provenance["cobertura_parcial_sustentada"] is True
    assert "transporte_ausente" in provenance["reglas_aplicadas"] or any(
        "transporte" in r for r in provenance["reglas_aplicadas"]
    )


def test_schema_exposes_transport_problem_with_semantic_context():
    ctx = analyze_context(CARTAGENA)
    schema = report_schema(CARTAGENA, "negativo", ["traslado", "alojamiento"], True, context=ctx)
    problems = schema["properties"]["problema_identificado"]["enum"]
    assert any("transporte" in p.casefold() for p in problems if p)
    assert all(s.startswith("Según el relato") or s.startswith("En la experiencia")
               for s in schema["properties"]["resumen"]["enum"])


def test_resolved_incident_stays_favorable_not_aclaracion():
    text = ("Al inicio la llave no funcionaba. El personal lo resolvió de inmediato "
            "y el resto de la estancia fue excelente.")
    ctx = analyze_context(text)
    options = content_options(text, evidence_units(text), context=ctx)
    assert "aclaracion" not in options["categories"] or options["mode"] == "PARAFRASIS_CONTROLADA"
    assert any(f in options["rules"] for f in
               ("incidente_resuelto", "incidente_resuelto_semantico", "falla_inicial", "comodidad"))


def test_unsupported_negations_still_conservative():
    for text in ("No nos cobraron dos veces.", "El transporte llegó y perdimos la visita.",
                 "Nadie resolvió el problema.", "No se perdió la excursión."):
        result = content_options(text, [text], context=analyze_context(text))
        assert result["mode"] == "CITA_CONSERVADORA"
        assert result["problems"] == [""]


def test_positive_clean_room_not_forced_to_transport():
    text = "La habitación estaba limpia."
    options = content_options(text, [text], context=analyze_context(text))
    assert options["mode"] == "PARAFRASIS_CONTROLADA"
    assert options["problems"] == [""]
    assert "favorable" in options["categories"]


def test_failed_generation_preserves_local_recommendation():
    fail = failed_generation("OLLAMA_DESCONECTADO", local_recommendation="Revisar el traslado descrito.")
    assert fail["generacion_llm"] is False
    assert fail["fallback_visible"] is True
    assert fail["recomendacion_local_conservada"] == "Revisar el traslado descrito."
    assert fail["alternativa"] == "RECOMENDACION_TRADICIONAL_POR_REGLAS"


def test_validate_generation_attaches_fallback_on_error():
    result = validate_generation(
        {"estado": "ERROR", "codigo": "OLLAMA_DESCONECTADO"},
        CARTAGENA, "negativo", ["traslado"], True,
        context=analyze_context(CARTAGENA),
        local_recommendation="Contrastar el incidente de traslado.",
    )
    assert result["estado"] == "ERROR"
    assert result["recomendacion_local_conservada"] == "Contrastar el incidente de traslado."
    assert result["informe"] is None
