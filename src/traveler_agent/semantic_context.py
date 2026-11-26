"""Contextual aspect polarity, incidents and evidence selection (rule-based, exploratory).

Generalizable lexical patterns for Spanish tourism reviews. Not ABSA, not trained,
and not tuned to any fixed acceptance battery. Does not assign the global label.
"""
from __future__ import annotations

import re
import unicodedata
from .aspects import ASPECTS, find_aspects

CONTENT_TAG = "semantic-context-v1"

_FAVORABLE = (
    r"amable|excelente|impecable|limpia|limpio|puntual|comod[oa]|"
    r"justo|razonable|rapid[oa]|eficiente|gentil|bacano|redondo|agradable|"
    r"sabros[oa]|caliente|fresca|ordenado|cuidada|clara|claro|recomend"
)
_UNFAVORABLE = (
    r"sucia|sucio|pesim[oa]|caro|costoso|demora|retraso|tardo|falla|"
    r"fallo|nunca llego|no (?:llego|llegaron|respondio|resolvio)|sin respuesta|"
    r"basura|mal olor|ruido|insegur|peligro|cobraron mas|cargo extra|cancelad|"
    r"perdimos|no pudimos|colgo|ignor|descuido|fri[oa]|danad|roto|no funcionaba"
)
# Patterns run on accent-folded text.
_RESOLUTION = (
    r"resolvio|resolvieron|se (?:resolvio|soluciono|corrigio)|corrigieron|"
    r"lo solucionaron|lo resolvio|despues(?: de)? (?:lo )?arreglaron|reembolsaron|devolvieron|"
    r"nos cambiaron|quedo resuelto|incidente resuelto"
)
_UNRESOLVED = (
    r"nadie (?:respondio|resolvio|aparecio)|sin (?:respuesta|solucion)|sigue (?:sin|abierto)|"
    r"no (?:se )?(?:resolvio|soluciono|corrigio)|quedo abierta|mensaje automatico|"
    r"colgo la llamada|nunca (?:llego|respondieron)"
)
_CONSEQUENCE = (
    r"perdimos|se perdio|no pudimos|anulo|quedamos sin|missed|"
    r"perdimos la (?:excursion|visita|actividad|manana)|sin conexion"
)
_DOUBT = (
    r"no se si|no estoy segur|tampoco diria|no logro decidir|"
    r"regular\b|sin comentarios|^ok$|^bien$|^mal$|^normal$"
)

_CLAUSE = re.compile(r"(?<=[.!?;])\s+|(?:\s*,\s*|\s+)(?=(?:pero|aunque|sin embargo)\b)", re.I)


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in value if not unicodedata.combining(c))


def _clauses(text: str) -> list[str]:
    parts = [p.strip() for p in _CLAUSE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _polarity_in(span: str) -> str | None:
    folded = _fold(span)
    # Negated praise → unfavorable cue, not favorable.
    if re.search(rf"\bno (?:era |estaba |fue |estuvo |muy |nada )*(?:{_FAVORABLE})", folded):
        return "desfavorable"
    folded = re.sub(r"\b(?:sin|no hubo|no habia) (?:retrasos?|demoras?|ruido)\b", "", folded)
    fav = bool(re.search(_FAVORABLE, folded, re.I))
    unf = bool(re.search(_UNFAVORABLE, folded, re.I))
    if fav and not unf:
        return "favorable"
    if unf and not fav:
        return "desfavorable"
    if fav and unf:
        return "mixto"
    return None


def assess_aspects(text: str) -> dict:
    """Map each detected aspect to local polarity cues in nearby clauses."""
    found = find_aspects(text)
    clauses = _clauses(text)
    assessment = {}
    for aspect, terms in found.items():
        local = []
        for clause in clauses:
            if any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", clause, re.I) for t in terms):
                pol = _polarity_in(clause)
                local.append({"clause": clause[:260], "polarity": pol or "indeterminada"})
        # Aspect mentioned without local cue → indeterminate.
        if not local:
            local = [{"clause": "", "polarity": "indeterminada"}]
        labels = {row["polarity"] for row in local}
        if labels <= {"favorable"}:
            summary = "favorable"
        elif labels <= {"desfavorable"}:
            summary = "desfavorable"
        elif "mixto" in labels or ({"favorable", "desfavorable"} <= labels):
            summary = "mixto"
        else:
            summary = "indeterminada"
        assessment[aspect] = {"terms": terms, "polarity": summary, "mentions": local}
    return assessment


def detect_incidents(text: str) -> list[dict]:
    """Detect open/resolved incident sketches with optional consequences."""
    incidents = []
    for clause in _clauses(text):
        folded = _fold(clause)
        if re.search(r"\b(?:dicen que|no confirmo|no se si|si hubiera)\b|[¿?]", folded):
            continue
        folded = re.sub(r"\b(?:no hubo|no habia|sin|no) (?:ningun |ninguna )?(?:retrasos?|demoras?|fallos?|ruido|perdimos|se perdio)\b", "", folded)
        unfavorable = bool(re.search(_UNFAVORABLE, folded, re.I))
        resolved = bool(re.search(_RESOLUTION, folded, re.I))
        denied_resolution = re.search(r"\b(?:no|nunca|nadie) (?:nos |lo |se |la )*(?:" + _RESOLUTION + r")\b", folded)
        unresolved = bool(denied_resolution) or bool(re.search(_UNRESOLVED, folded, re.I)) or (
            unfavorable and not resolved and bool(re.search(
                r"nunca|nadie|sin |no (?:lleg|respond|resol)", folded))
        )
        consequence = bool(re.search(_CONSEQUENCE, folded, re.I))
        if not (unfavorable or resolved or unresolved or consequence):
            continue
        status = "resuelto" if resolved and not unresolved else (
            "no_resuelto" if (unresolved or (unfavorable and consequence)) else
            "mencionado"
        )
        # Aspect hint from dictionary hits inside the clause.
        aspect_hits = [name for name, terms in ASPECTS.items()
                       if any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", clause, re.I) for t in terms)]
        incidents.append({
            "status": status,
            "clause": clause[:260],
            "consequence": consequence,
            "aspects": aspect_hits,
            "resolved_marker": resolved and not unresolved,
            "unresolved_marker": unresolved,
        })
    # Deduplicate by clause text.
    seen, unique = set(), []
    for row in incidents:
        key = row["clause"].casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique[:6]


def select_evidence(text: str, aspects: dict, incidents: list[dict], *, max_spans: int = 4) -> list[str]:
    """Prefer incident and polarity-bearing clauses over generic aspect mentions."""
    ranked = []
    for inc in incidents:
        weight = 3 if inc["status"] == "no_resuelto" else 2 if inc["status"] == "resuelto" else 1
        if inc["consequence"]:
            weight += 1
        ranked.append((weight, inc["clause"]))
    for clause in _clauses(text):
        pol = _polarity_in(clause)
        if pol in {"desfavorable", "mixto", "favorable"}:
            ranked.append((2 if pol != "favorable" else 1, clause[:260]))
        elif any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", clause, re.I)
                 for terms in aspects.values() for t in terms):
            ranked.append((0, clause[:260]))
    ranked.sort(key=lambda x: (-x[0], len(x[1])))
    out = []
    for _, span in ranked:
        if span and span not in out:
            out.append(span)
        if len(out) >= max_spans:
            break
    return out


def ambiguity_signals(text: str, aspect_assessment: dict, incidents: list[dict]) -> dict:
    folded = _fold(text.strip())
    doubt = bool(re.search(_DOUBT, folded, re.I))
    mixed_aspects = sum(1 for a in aspect_assessment.values() if a["polarity"] == "mixto")
    open_incidents = sum(1 for i in incidents if i["status"] == "no_resuelto")
    resolved = sum(1 for i in incidents if i["status"] == "resuelto")
    favorable = sum(1 for a in aspect_assessment.values() if a["polarity"] == "favorable")
    unfavorable = sum(1 for a in aspect_assessment.values() if a["polarity"] == "desfavorable")
    mixed_global = mixed_aspects > 0 or (favorable and unfavorable and open_incidents)
    return {
        "doubt": doubt,
        "mixed_aspects": mixed_global,
        "open_incidents": open_incidents,
        "resolved_incidents": resolved,
        "ambiguous": bool(doubt or (mixed_global and open_incidents == 0 and resolved == 0)),
    }


def analyze_context(text: str) -> dict:
    aspects = find_aspects(text)
    assessment = assess_aspects(text)
    incidents = detect_incidents(text)
    evidence = select_evidence(text, aspects, incidents)
    signals = ambiguity_signals(text, assessment, incidents)
    return {
        "version": CONTENT_TAG,
        "aspects": aspects,
        "aspect_assessment": assessment,
        "incidents": incidents,
        "evidence": evidence,
        "ambiguity": signals,
        "scope": "Exploratorio por patrones léxicos; no sustituye la etiqueta del clasificador.",
    }
