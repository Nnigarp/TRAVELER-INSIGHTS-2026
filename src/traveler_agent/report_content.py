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
    if re.search(r"\b(?:perdimos|se perdio)\b.+\b(?:excursion|visita|actividad|tour)\b", value) and not _negated(value, "perdimos"):
        if re.search(r"\b(?:transporte|traslado|taxi|bus|autobus)\b", value):
            return Fact("transporte_ausente",
                        "se reportan la ausencia del transporte y la pérdida de una actividad",
                        "Según el relato, el transporte no llegó y se perdió una actividad.",
                        "transporte")
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
    if re.search(r"\bcobro duplicado\b|\bcobraron dos veces\b", value):
        return Fact("cobro_discrepante", "se relata un cobro duplicado o reiterado",
                    "Se reporta una diferencia entre el cobro y el precio anunciado.", "precio")
    if re.fullmatch(r"(?:habia|hubo) (?:mucho|demasiado) ruido(?: (?:por la noche|en la habitacion))?", value):
        return Fact("ruido", "se describe ruido abundante",
                    "El relato describe ruido abundante, sin identificar su origen.", "ruido")
    if re.fullmatch(r"la (?:visita|excursion) duro (?:una hora|dos horas|tres horas)", value):
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
    if re.search(r"\b(?:resolvio|resolvieron|se soluciono|corrigieron|devolvieron|reembolsaron)\b", value):
        if re.search(r"\b(?:problema|llave|cobro|falla|incidencia|retraso)\b", value):
            return Fact("incidente_resuelto", "se relata un incidente que el visitante marca como resuelto",
                        positive=True)
    return None


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


def content_options(source, units):
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
    evidence = list(dict.fromkeys(quotes))
    recognised = [f for f in facts if f]
    complete = len(recognised) == len(facts) and bool(facts)
    summaries = []
    if complete:
        summary_facts = list(dict.fromkeys(recognised))
        if len(summary_facts) == 2 and (
            (summary_facts[0].positive and summary_facts[1].problem) or
            (summary_facts[0].problem and summary_facts[1].positive)
        ):
            body = summary_facts[0].wording + ", pero " + summary_facts[1].wording
        else:
            body = "; ".join(f.wording for f in summary_facts)
        for prefix in ("Según el relato, ", "En la experiencia descrita, "):
            candidate = prefix + body + "."
            if len(candidate) <= 600:
                summaries.append(candidate)
    mode = "PARAFRASIS_CONTROLADA" if summaries else "CITA_CONSERVADORA"
    if not summaries:
        # Prefer clause-level citations for multi-sentence prose; keep full-source
        # units when there is no safer compact alternative (escaping / single clause).
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
    problems = list(dict.fromkeys(f.problem for f in recognised if f.problem)) if complete else []
    categories = list(dict.fromkeys(f.category for f in recognised if f.category)) if complete else []
    descriptive = {"duracion", "descripcion_espacio", "descripcion_horario", "incidente_resuelto"}
    if not categories:
        if complete and any(f.problem for f in recognised):
            categories = ["aclaracion"]
        elif complete and any(f.positive for f in recognised) and not any(f.problem for f in recognised):
            categories = ["favorable"]
        elif complete and recognised and all(f.rule in descriptive or f.rule == "duracion" for f in recognised):
            categories = ["informativo"]
        elif complete and all(f.rule == "insuficiente" or f.rule == "duda" for f in recognised):
            categories = ["aclaracion"]
        elif complete:
            categories = ["informativo"]
        else:
            # Partial recognition: do not invent an incident; ask clarification
            # only when no descriptive paraphrase path exists.
            categories = ["aclaracion"]
    recommendations = [item for category in categories for item in PROPOSALS[category][:2]]
    actions = [PROPOSALS[category][2] for category in categories]
    return {"summaries": summaries, "problems": problems or [""], "evidence": evidence,
            "recommendations": recommendations, "actions": actions,
            "problem_categories": {f.problem: f.category for f in recognised if f.problem} if complete else {},
            "categories": categories, "mode": mode,
            "rules": list(dict.fromkeys(f.rule for f in recognised)), "complete": complete}


def content_provenance(source, units, report):
    options = content_options(source, units)
    return {"version": CONTENT_VERSION, "resumen": options["mode"],
            "problema": "PARAFRASIS_CONTROLADA" if report["problema_identificado"] else "NO_DETERMINADO",
            "reglas_aplicadas": options["rules"], "cobertura_completa": options["complete"],
            "alcance": "Reglas acotadas; no valida semántica arbitraria ni la veracidad del relato."}
