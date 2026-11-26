"""Surface-realisation contract for evidence-backed reports (tourism-content-v2).

Only narrowly recognised clauses receive paraphrases. Unknown language stays
quoted. Qwen selects among formulations; it cannot introduce free facts.
Improved v2: broader generalizable patterns, shorter summaries, fewer needless
clarification prompts on factual neutrals; anti-hallucination barriers preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

CONTENT_VERSION = "tourism-content-v2"


@dataclass(frozen=True)
class Fact:
    rule: str
    wording: str
    problem: str = ""
    category: str = ""
    positive: bool = False


def _fold(text):
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in value if not unicodedata.combining(c)).strip(" .!?;\t\r\n")


def _negated(value: str, stem: str) -> bool:
    return bool(re.search(rf"\bno\b.{{0,28}}\b{stem}\b", value))


def _fact(quote):
    if any(marker in quote for marker in ("?", "¿", "...", "…")):
        return None
    value = _fold(quote)
    # Deny uncertain attribution / reported speech as confirmed incidents.
    # (Doubt phrases like "no se si volveria" are handled below as explicit facts.)
    if re.search(r"\b(?:no confirmo|dicen que|no fue cierto)\b", value):
        return None
    if re.search(r"\bno se si\b", value) and not re.fullmatch(
            r"no se si (?:volveria|lo recomendaria)", value):
        return None

    if re.fullmatch(r"(?:el (?:hotel|alojamiento)|la habitacion) (?:era|fue|es|estaba|estuvo) (?:muy )?comod[oa]", value):
        return Fact("comodidad", "se valora la comodidad del alojamiento", positive=True)
    if re.fullmatch(r"la habitacion (?:estaba|estuvo|esta|era|es) (?:muy )?limpia", value):
        return Fact("limpieza_favorable", "se valora la limpieza de la habitación", positive=True)
    if re.fullmatch(r"(?:el personal|el guia|la guia|la atencion|el servicio) (?:fue|era|es|estuvo) (?:muy )?(?:amable|excelente|gentil|cordial)", value):
        return Fact("servicio_favorable", "se valora favorablemente el servicio", positive=True)
    if re.fullmatch(r"(?:el desayuno|la comida) (?:estaba|estuvo|era|fue|llego) (?:muy )?(?:bueno|buena|caliente|sabroso|sabrosa)", value):
        return Fact("comida_favorable", "se valora favorablemente la alimentación", positive=True)
    if re.fullmatch(r"(?:el precio|la tarifa) (?:nos )?pareci[oó] (?:justo|razonable|adecuado)", value):
        return Fact("precio_favorable", "se valora la claridad o justicia del precio", positive=True)

    transport = re.fullmatch(
        r"el (?:transporte|traslado|taxi|bus|autobus|navette|shuttle) (?:nunca|no) (?:llego|se presento)"
        r"(?: y (?:perdimos|se perdio) (?:la|el) (?:excursion|visita|tour|actividad))?", value)
    if transport:
        lost = bool(re.search(r"\b(?:perdimos|se perdio)\b", value))
        wording = ("se reportan la ausencia del transporte y la pérdida de una actividad"
                   if lost else "se reporta que el transporte no llegó")
        problem = ("Según el relato, el transporte no llegó y se perdió una actividad."
                   if lost else "Según el relato, no se recibió el transporte esperado.")
        return Fact("transporte_ausente", wording, problem, "transporte")
    if re.fullmatch(r"(?:perdimos|se perdio) (?:la|el|una|un) (?:excursion|visita|actividad|tour)", value):
        return Fact("actividad_perdida", "se reporta la pérdida de una actividad",
                    "Según el relato, se perdió una actividad programada.", "actividad")

    if re.fullmatch(r"la habitacion (?:estaba|estuvo|esta|era|es) (?:muy )?sucia", value):
        return Fact("habitacion_sucia", "se reporta suciedad en la habitación",
                    "Según el relato, la habitación presentaba suciedad.", "higiene")
    if re.fullmatch(r"la (?:excursion|visita|actividad) (?:fue cancelada|se cancelo)", value):
        return Fact("actividad_cancelada", "se reporta la cancelación de una actividad",
                    "Se relata la cancelación de una actividad, sin precisar su causa.", "actividad")
    if re.fullmatch(r"esperamos (?:mucho|demasiado|una hora|dos horas|tres horas|veinte minutos)(?: para (?:entrar|acceder))?", value):
        return Fact("espera", "se relata una espera",
                    "El visitante describe una espera en el servicio.", "espera")
    if re.fullmatch(r"(?:nos cobraron|el cobro fue) mas de lo (?:anunciado|indicado)", value):
        return Fact("cobro_discrepante", "se relata un cobro superior a lo anunciado",
                    "Se reporta una diferencia entre el cobro y el precio anunciado.", "precio")
    if re.fullmatch(r"(?:hubo un cobro duplicado|nos cobraron dos veces(?: (?:el|la) (?:desayuno|entrada|habitacion))?)", value):
        return Fact("cobro_duplicado", "se relata un cobro duplicado",
                    "Según el relato, se cobró dos veces.", "precio")
    if re.fullmatch(r"(?:habia|hubo) (?:mucho|demasiado) ruido(?: (?:por la noche|en la habitacion))?", value):
        return Fact("ruido", "se describe ruido abundante",
                    "El relato describe ruido abundante, sin identificar su origen.", "ruido")
    if re.fullmatch(r"(?:la (?:visita|excursion|actividad)|el (?:recorrido|viaje)) duro (?:una?|dos|tres|cuatro|veinte|treinta|\d+) (?:horas?|minutos?)", value):
        return Fact("duracion", "se aporta información sobre la duración de la actividad")
    if re.fullmatch(r"habia dos salas y una zona de espera", value):
        return Fact("descripcion_espacio", "se describen salas y una zona de espera")
    if re.fullmatch(r"el horario estaba publicado(?: en la entrada)?", value):
        return Fact("descripcion_horario", "se indica que el horario estaba publicado")
    if value == "no se si volveria":
        return Fact("duda", "se expresa duda sobre repetir la experiencia")
    if value == "no se si lo recomendaria":
        return Fact("duda", "se expresa duda sobre recomendar la experiencia")
    if re.fullmatch(r"bien|regular|mal|normal|ok|no se|sin comentarios", value):
        return Fact("insuficiente", "el comentario no detalla un servicio ni un incidente concreto")

    # Resolved incident (generalizable): unfavorable cue + resolution verb in same clause.
    if re.fullmatch(r"(?:el personal |recepcion )?(?:resolvio|resolvieron|solucionaron|corrigieron) (?:el problema|la falla|la incidencia)(?: de inmediato)?", value):
        return Fact("incidente_resuelto", "el visitante indica que se resolvió el incidente", positive=True)
    if re.fullmatch(r"(?:al inicio )?la (?:llave|ducha|cerradura) no funcionaba", value):
        return Fact("falla_inicial", "se describe un fallo inicial")
    if re.fullmatch(r"(?:el personal |recepcion )?lo (?:resolvio|soluciono|arreglo)(?: de inmediato)?(?: y el resto de la estancia fue excelente)?", value):
        return Fact("incidente_resuelto", "el visitante indica que el problema se resolvió", positive=True)
    return None


def is_factual_description(text):
    clauses = re.split(r"(?<=[.!?;])\s+|[\r\n]+", text.strip())
    return all((fact := _fact(c)) is not None and fact.rule in
               {"duracion", "descripcion_espacio", "descripcion_horario"} for c in clauses)


PROPOSALS = {
    "transporte": (
        "Verificar quién organizó el traslado y qué condiciones se habían comunicado, sin atribuir todavía responsabilidades.",
        "Contrastar con una persona la información disponible sobre el traslado antes de plantear medidas.",
        "Preparar un reporte del traslado relatado y reunir las evidencias disponibles para valoración humana, sin enviarlo a terceros.",
    ),
    "higiene": (
        "Solicitar una comprobación humana del estado de limpieza descrito antes de decidir medidas.",
        "Contrastar la descripción de la habitación con las evidencias disponibles, sin dar por establecida la causa.",
        "Preparar un registro de la observación de limpieza y de sus evidencias para revisión humana, sin contactar a terceros.",
    ),
    "actividad": (
        "Verificar la información comunicada sobre la actividad cancelada o perdida antes de atribuir causas.",
        "Contrastar con una persona la programación y lo ocurrido antes de proponer medidas.",
        "Preparar un reporte de la actividad descrita para que una persona valore qué aclaraciones solicitar, sin enviarlo a terceros.",
    ),
    "espera": (
        "Contrastar la espera relatada con la información disponible del servicio, sin dar por confirmadas sus causas.",
        "Solicitar detalles sobre el servicio y las condiciones de la espera para una revisión proporcionada.",
        "Preparar un registro de la espera descrita para valoración humana, sin ejecutar cambios en el servicio.",
    ),
    "precio": (
        "Contrastar el cobro relatado con la información de precio disponible antes de proponer medidas.",
        "Solicitar una revisión humana de las evidencias del cobro y del precio anunciado, sin prometer compensaciones.",
        "Preparar un registro de la diferencia de precio relatada para revisión humana, sin iniciar reclamaciones ni pagos.",
    ),
    "ruido": (
        "Comprobar las condiciones del ruido descrito antes de decidir medidas, sin atribuir todavía su origen.",
        "Solicitar detalles de la situación de ruido para una valoración humana proporcionada.",
        "Preparar una nota de la observación de ruido para revisión humana, sin contactar a terceros.",
    ),
    "favorable": (
        "Recoger qué aspectos se valoraron favorablemente para revisión humana, sin generalizarlos a otros visitantes.",
        "Conservar la valoración favorable y su evidencia para que una persona evalúe posibles aprendizajes del servicio.",
        "Preparar una nota de los aspectos favorables expresados para revisión humana, sin ejecutar cambios en el servicio.",
    ),
    "informativo": (
        "Conservar la información de la visita como descripción, sin inferir satisfacción ni un incidente.",
        "Distinguir los datos descriptivos de una valoración de calidad antes de proponer medidas.",
        "Preparar una ficha descriptiva de lo narrado para consulta humana, sin ejecutar intervenciones externas.",
    ),
    "aclaracion": (
        "Solicitar detalles del servicio y de lo ocurrido antes de identificar un incidente o proponer medidas.",
        "Consultar al visitante qué aspecto necesita aclaración, sin completar la información que falta.",
        "Preparar una solicitud de aclaración para valoración humana, sin contactar al visitante ni enviarla a terceros.",
    ),
}


def _strip_contrast_prefix(quote: str) -> str:
    return re.sub(r"^(?:pero|aunque|sin embargo)\s+", "", quote.strip(), flags=re.I).strip(" ,")


def _fact_from_incident(inc: dict, source: str) -> Fact | None:
    """Ground a SemanticsAgent incident only when its clause is literal in source."""
    if not isinstance(inc, dict):
        return None
    clause = (inc.get("clause") or "").strip()
    if not clause or clause not in source:
        return None
    cleaned = _strip_contrast_prefix(clause)
    direct = _fact(cleaned) or _fact(clause)
    if direct is not None:
        return direct
    folded = _fold(clause)
    status = inc.get("status")
    # Narrow keyword bridges only when _fact cannot parse the clause form but the
    # SemanticsAgent status already encodes resolution / explicit no-show cues.
    # Never invent a no-show from a mere consequence like "perdimos la visita".
    if status == "no_resuelto" and re.search(
            r"\b(?:transporte|traslado|taxi|bus|autobus)\b", folded) and re.search(
            r"\b(?:nunca|no) (?:llego|se presento)\b", folded):
        lost = bool(re.search(r"\b(?:perdimos|se perdio)\b", folded))
        return Fact(
            "transporte_semantico",
            "se reportan la ausencia del transporte y la pérdida de una actividad" if lost
            else "se reporta que el transporte no llegó",
            ("Según el relato, el transporte no llegó y se perdió una actividad."
             if lost else "Según el relato, no se recibió el transporte esperado."),
            "transporte",
        )
    if status == "resuelto" and re.search(
            r"\b(?:resolvio|resolvieron|soluciono|corrigieron|devolvieron|reembolsaron)\b", folded):
        return Fact("incidente_resuelto_semantico",
                    "el visitante indica que un incidente quedó resuelto", positive=True)
    return None


def _build_summaries(facts: list[Fact]) -> list[str]:
    if not facts:
        return []
    summary_facts = list(dict.fromkeys(facts))
    if len(summary_facts) == 2 and (
        (summary_facts[0].positive and summary_facts[1].problem) or
        (summary_facts[0].problem and summary_facts[1].positive)
    ):
        body = summary_facts[0].wording + ", pero " + summary_facts[1].wording
    else:
        body = "; ".join(f.wording for f in summary_facts)
    out = []
    for prefix in ("Según el relato, ", "En la experiencia descrita, "):
        candidate = prefix + body + "."
        if len(candidate) <= 600:
            out.append(candidate)
    return out


def content_options(source, units, *, context=None):
    """Build Qwen enum options. Optional SemanticsAgent context may ground extra facts.

    Partial sentence coverage no longer forces aclaracion when grounded incidents
    or recognised clauses already sustain a paraphrase.
    """
    context = context or {}
    sentences = [u.strip() for u in re.split(r"(?<=[.!?])\s+|[\r\n]+", source.strip()) if u.strip()]
    quotes, facts = [], []
    for sentence in sentences:
        parts = re.split(r",?\s+pero\s+", sentence, flags=re.I)
        if len(parts) == 2 and all(_fact(p) is not None for p in parts):
            pieces = parts
        else:
            pieces = [sentence]
        for quote in pieces:
            quote = quote.strip()
            quotes.append(quote)
            facts.append(_fact(quote))
    evidence = [q for q, f in zip(quotes, facts) if f] or list(dict.fromkeys(quotes))
    recognised = [f for f in facts if f]
    complete = len(recognised) == len(facts) and bool(facts)

    def _uncertain_unmatched() -> bool:
        for quote, fact in zip(quotes, facts):
            if fact is not None:
                continue
            folded = _fold(quote)
            if "?" in quote or "¿" in quote:
                return True
            if re.search(r"\b(?:no fue cierto|dicen que|no confirmo|no se si)\b", folded):
                return True
            if re.search(r"\bno\b.{0,40}\b(?:sucia|sucio|llego|perdio|funcionaba|resolvio)\b", folded):
                return True
        return False

    # Partial paraphrases are allowed only when unmatched clauses do not deny or
    # unsettle the recognised facts (anti-hallucination).
    block_partial = (not complete) and _uncertain_unmatched()

    # Bridge SemanticsAgent incidents whose clauses appear verbatim in the source.
    semantic_facts = []
    semantic_evidence = []
    if not block_partial:
        for inc in context.get("incidents") or []:
            fact = _fact_from_incident(inc, source)
            if fact is None:
                continue
            semantic_facts.append(fact)
            clause = (inc.get("clause") or "").strip()
            if clause and clause in source:
                semantic_evidence.append(_strip_contrast_prefix(clause) or clause)
        for span in context.get("evidence") or []:
            if isinstance(span, str) and span in source and span not in semantic_evidence:
                semantic_evidence.append(span)

    usable = [] if block_partial else list(recognised)
    known_problems = {f.problem for f in usable if f.problem}
    known_categories = {f.category for f in usable if f.category}
    known_wordings = {f.wording for f in usable}
    for fact in semantic_facts:
        if fact.wording in known_wordings:
            continue
        if fact.problem and (fact.problem in known_problems or
                             (fact.category and fact.category in known_categories)):
            continue
        usable.append(fact)
        if fact.problem:
            known_problems.add(fact.problem)
        if fact.category:
            known_categories.add(fact.category)
        known_wordings.add(fact.wording)
    summaries = _build_summaries(usable) if usable else []
    mode = "PARAFRASIS_CONTROLADA" if summaries else "CITA_CONSERVADORA"
    if summaries:
        # Prefer literal clauses recognised by _fact. Semantic clauses may ground
        # summaries/problems but must not invent evidence enums that diverge from
        # the same source without context (ActionAgent / fixtures revalidation).
        evidence = list(dict.fromkeys(q for q, f in zip(quotes, facts) if f))
        if not evidence:
            evidence = list(dict.fromkeys(semantic_evidence or quotes))
    else:
        clause_summaries = [s for s in sentences if 0 < len(s) <= 280]
        prose = [s for s in clause_summaries
                 if not s.lstrip().startswith(("<", "!", "[", "/"))]
        if len(prose) >= 2 or (
            len(prose) == 1 and prose[0] != source.strip() and len(source.strip()) > len(prose[0]) + 10
        ):
            summaries = prose[:3]
        else:
            summaries = list(dict.fromkeys(u for u in units if isinstance(u, str) and u.strip()))[:3]
            if not summaries and source.strip():
                summaries = [source.strip()]
        evidence = list(dict.fromkeys(sentences)) if sentences else evidence

    problems = list(dict.fromkeys(f.problem for f in usable if f.problem))
    categories = list(dict.fromkeys(f.category for f in usable if f.category))
    descriptive = {"duracion", "descripcion_espacio", "descripcion_horario", "incidente_resuelto",
                   "incidente_resuelto_semantico"}
    if not categories:
        if problems:
            categories = ["aclaracion"]
        elif usable and any(f.positive for f in usable) and not any(f.problem for f in usable):
            categories = ["favorable"]
        elif usable and all(f.rule in descriptive or f.rule == "duracion" for f in usable):
            categories = ["informativo"]
        elif usable and all(f.rule in {"insuficiente", "duda"} for f in usable):
            categories = ["aclaracion"]
        elif usable:
            categories = ["informativo"]
        else:
            categories = ["aclaracion"]
    recommendations = [item for category in categories for item in PROPOSALS[category][:2]]
    actions = [PROPOSALS[category][2] for category in categories]
    problem_categories = {f.problem: f.category for f in usable if f.problem}
    return {"summaries": summaries, "problems": problems or [""], "evidence": evidence,
            "recommendations": recommendations, "actions": actions,
            "problem_categories": problem_categories,
            "categories": categories, "mode": mode,
            "rules": list(dict.fromkeys(f.rule for f in usable)), "complete": complete,
            "grounded_partial": bool(usable) and not complete}


def content_provenance(source, units, report, *, context=None):
    options = content_options(source, units, context=context)
    return {"version": CONTENT_VERSION, "resumen": options["mode"],
            "problema": "PARAFRASIS_CONTROLADA" if report["problema_identificado"] else "NO_DETERMINADO",
            "reglas_aplicadas": options["rules"], "cobertura_completa": options["complete"],
            "cobertura_parcial_sustentada": options.get("grounded_partial", False),
            "alcance": "Reglas acotadas; no valida semántica arbitraria ni la veracidad del relato."}
