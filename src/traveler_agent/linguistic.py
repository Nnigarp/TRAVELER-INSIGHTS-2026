"""Conservative, non-destructive linguistic signals and verifiable source spans.

Heuristics are descriptive. They do not claim contextual sentiment, irony
understanding, ABSA, or validated language identification.
"""
from __future__ import annotations

import re
import unicodedata
from .aspects import ASPECTS

_LANGUAGE_TERMS = {
    "es": frozenset({"el", "la", "los", "las", "una", "un", "fue", "estuvo", "pero", "muy", "servicio", "guía", "recorrido", "que", "habitación", "no", "estaba"}),
    "en": frozenset({"the", "and", "was", "were", "very", "but", "hotel", "tour", "room", "service", "great", "terrible", "not", "staff"}),
    "it": frozenset({"il", "lo", "gli", "una", "un", "era", "molto", "ma", "servizio", "guida", "camera", "non", "personale", "bellissimo"}),
}
_NEGATIONS = ("no", "nunca", "jamás", "sin", "ni", "not", "never", "without", "non", "senza")
_CONTRASTS = ("pero", "aunque", "sin embargo", "but", "however", "although", "ma", "però", "tuttavia")
_WORDS = re.compile(r"\b[^\W_]+\b", re.UNICODE)


def normalize_for_comparison(text: str) -> str:
    """Produce a derived matching key while retaining original punctuation and emoji."""
    if not isinstance(text, str):
        raise TypeError("Se esperaba un texto.")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def detect_language_heuristic(text: str) -> dict:
    """Lightweight indicative es/en/it detector; abstain on weak or tied evidence."""
    words = [m.group().casefold() for m in _WORDS.finditer(text)]
    counts = {lang: sum(word in lexicon for word in words) for lang, lexicon in _LANGUAGE_TERMS.items()}
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    top, runner_up = ranked[0], ranked[1]
    decisive = top[1] >= 2 and top[1] > runner_up[1]
    return {
        "idioma_indicativo": top[0] if decisive else "desconocido",
        "metodo": "stopwords_heuristic_v1_no_validado",
        "coincidencias": counts,
        "nota": "Detección exploratoria; puede fallar en textos cortos, dialectos o mezcla de idiomas.",
    }


def _term_positions(text: str, terms: tuple[str, ...] | list[str]) -> list[dict]:
    matches = []
    for term in terms:
        for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, flags=re.IGNORECASE):
            matches.append({"inicio": match.start(), "fin": match.end(), "texto": match.group()})
    return sorted(matches, key=lambda x: (x["inicio"], x["fin"], x["texto"]))


def aspect_spans(text: str) -> list[dict]:
    """Literal offsets into original Python string, [start, end)."""
    spans = []
    for aspect, terms in ASPECTS.items():
        for span in _term_positions(text, terms):
            spans.append({"aspecto": aspect, **span})
    return sorted(spans, key=lambda x: (x["inicio"], x["fin"], x["aspecto"]))


def contextual_signals(text: str) -> dict:
    """Find cues, not semantic conclusions; never assign aspect polarity."""
    negations = _term_positions(text, _NEGATIONS)
    contrasts = _term_positions(text, _CONTRASTS)
    return {
        "negaciones": negations,
        "contrastes": contrasts,
        "posible_opinion_mixta": bool(contrasts),
        "advertencia": "Conectores y negaciones son indicios; no confirman ironía ni polaridad mixta.",
    }


def linguistic_profile(text: str) -> dict:
    if not isinstance(text, str):
        raise TypeError("Se esperaba un texto.")
    return {
        "idioma": detect_language_heuristic(text),
        "texto_normalizado_solo_comparacion": normalize_for_comparison(text),
        "evidencias_posicionales": aspect_spans(text),
        "indicios_contextuales": contextual_signals(text),
    }
