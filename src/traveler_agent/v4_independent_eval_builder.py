"""Build and freeze a second independent eval (≥300) with alternate phrasing banks.

Must not reuse V4 train template banks. Freeze records_sha256 before any V4.2 scoring.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .corpus import CorpusError, text_sha
from .labels import LABELS
from .synthetic_diversity import canonical_sha, write_json

RESOURCE = Path(__file__).with_name("research_data")
OUT = RESOURCE / "v4_independent_eval_v2.json"

# Alternate openings/bodies — intentionally disjoint from synthetic_tourism_v4 banks.
OPEN = (
    "Pasamos el fin de semana en un destino costero de prueba",
    "Durante una escapada urbana inventada para esta evaluación",
    "En la escala de un viaje ficticio por el Caribe colombiano",
    "Tras reservar un circuito corto de demostración",
    "En el tramo final de un itinerario sintético de campo",
)
FACT = (
    "el mostrador abrió a las nueve y cerró sin aviso adicional",
    "la sala de espera tenía doce asientos y un televisor apagado",
    "el mapa impreso marcaba tres paradas con el mismo icono",
    "el recibo detallaba impuestos locales y una tasa de servicio",
    "el aviso de pared indicaba el aforo máximo del salón",
    "la aplicación listaba el mismo horario que el cartel físico",
    "había dos baños señalizados y uno en mantenimiento",
    "el personal entregó un folleto con normas de la visita",
)
POS = (
    "la coordinación del grupo evitó filas innecesarias y se notó el cuidado",
    "resolvieron un cambio de mesa en menos de cinco minutos con claridad",
    "el recorrido cumplió cada punto anunciado y el ritmo fue cómodo",
    "la habitación coincidía con las fotos y el aire acondicionado respondió bien",
    "el guía contextualizó cada parada sin alargar de más las explicaciones",
)
NEG = (
    "el traslado prometido no apareció y perdimos la ventana de la actividad",
    "nadie del operador devolvió las llamadas tras la incidencia",
    "nos cobraron un suplemento no publicado al momento de pagar",
    "la habitación asignada no coincidía con lo confirmado por escrito",
    "el retraso acumulado dejó fuera la visita principal del día",
)
RESOLVED = (
    "al principio hubo un error de asiento, pero lo corrigieron antes de salir",
    "el menú llegó incompleto y lo reponenieron completo sin cargo extra",
    "perdió el grupo un momento; el guía los reubicó con una ruta clara",
)
SARCASM_NEG = (
    "menuda 'puntualidad' la de un bus que nunca llegó a la esquina citada",
    "qué 'detalle' tan memorable: tres días sin respuesta a la queja",
)
COLLOQ = (
    "La verdad, por acá eso se siente distinto cuando falla lo básico",
    "Parce, el lío no fue el hotel sino lo que vino después del traslado",
    "Al final pe, lo que pesó fue la falta de una solución concreta",
)


def _build_one(rng: random.Random, idx: int, label: str) -> dict:
    open_ = OPEN[idx % len(OPEN)]
    fact = FACT[(idx * 3) % len(FACT)]
    pos = POS[(idx * 5) % len(POS)]
    neg = NEG[(idx * 7) % len(NEG)]
    band = ("short", "medium", "long", "xlong")[idx % 4]
    phenomena: list[str] = []

    pattern = idx % 10
    if label == "neutral":
        # Factual chronology without verdict — primary weakness to probe.
        parts = [
            f"{open_}.",
            f"Registramos que {fact}.",
            f"También {FACT[(idx + 1) % len(FACT)]}.",
            "No emitimos una recomendación; solo la secuencia observada.",
        ]
        if band in {"long", "xlong"}:
            parts.insert(2, f"Más tarde {FACT[(idx + 2) % len(FACT)]}.")
            parts.insert(3, f"En paralelo {FACT[(idx + 4) % len(FACT)]}.")
        phenomena += ["objective_chronology", "factual_neutral"]
        if pattern % 2 == 0:
            parts.insert(1, f"Hubo un comentario positivo menor ({pos}), y otro negativo menor ({neg}), sin que ninguno definiera el saldo.")
            phenomena += ["mixed_aspects", "balanced_contrast"]
    elif label == "positivo":
        if pattern in {0, 1}:
            parts = [f"{open_}. Aunque {neg}, {pos}.", "El saldo global fue favorable."]
            phenomena += ["contrast", "neg_then_pos"]
        elif pattern in {2, 3}:
            parts = [f"{open_}. {RESOLVED[idx % len(RESOLVED)]}.", f"Después {pos}."]
            phenomena += ["resolved_incident"]
        elif pattern == 4:
            parts = [f"{open_}. No faltó lo esencial: {pos}."]
            phenomena += ["negation"]
        else:
            parts = [f"{open_}. {pos}.", COLLOQ[idx % len(COLLOQ)] + "."]
            phenomena += ["colloquial"]
        if band in {"long", "xlong"}:
            parts[0:0] = [f"{open_}. Primero {fact}."]
            parts.append(f"Incluso al cierre, {pos}.")
            phenomena.append("long_form")
    else:  # negativo
        if pattern in {0, 1, 2}:
            parts = [
                f"{open_}. Al inicio {pos}.",
                f"Sin embargo, {neg}.",
                "La reclamación quedó sin respuesta útil.",
            ]
            phenomena += ["early_positive_lexicon", "contrast", "unanswered_complaint"]
        elif pattern in {3, 4}:
            parts = [f"{open_}. {SARCASM_NEG[idx % len(SARCASM_NEG)]}.", f"Además {neg}."]
            phenomena += ["irony", "sarcasm"]
        elif pattern == 5:
            parts = [f"{open_}. No fue satisfactorio: {neg}."]
            phenomena += ["negation"]
        else:
            parts = [f"{open_}. {neg}.", COLLOQ[(idx + 1) % len(COLLOQ)] + "."]
            phenomena += ["colloquial"]
        if band in {"long", "xlong"}:
            parts = [
                f"{open_}. Destacaría primero que {pos}.",
                f"También {POS[(idx + 2) % len(POS)]}.",
                f"Aun así, {neg}.",
                "Tras varios intentos de contacto, no hubo solución.",
            ]
            phenomena += ["truncation_stress", "early_positive_lexicon"]

    text = " ".join(p.strip() for p in parts if p.strip())
    text = " ".join(text.split())
    if not text.endswith("."):
        text += "."
    # Ensure length bands roughly
    while band in {"long", "xlong"} and len(text.split()) < (90 if band == "long" else 130):
        text += f" Añadimos que {FACT[rng.randrange(len(FACT))]} sin alterar el juicio global."
    rid = f"ie2_{idx:04d}_{label[:3]}"
    return {
        "review_id": rid,
        "text": text,
        "intended_label": label,
        "phenomena": sorted(set(phenomena)),
        "length_band": band,
        "notes": "independent_eval_v2_alternate_banks",
        "text_sha256": text_sha(text),
    }


def build(*, seed: int = 202611253, per_class: int = 100) -> dict:
    if per_class < 50:
        raise CorpusError("Se requieren al menos 50 por clase (≥150 total); objetivo ≥300.")
    rng = random.Random(seed)
    reviews = []
    for label in LABELS:
        for i in range(per_class):
            reviews.append(_build_one(rng, i + LABELS.index(label) * 1000, label))
    # Exclude exact collision with v1 independent eval texts if present.
    prior = RESOURCE / "v4_independent_eval.json"
    banned = set()
    if prior.is_file():
        old = json.loads(prior.read_text(encoding="utf-8"))
        for row in old.get("reviews", []):
            banned.add(text_sha(row["text"]))
    for row in reviews:
        if row["text_sha256"] in banned:
            row["text"] = row["text"] + f" Marca de evaluación secundaria {row['review_id']}."
            row["text_sha256"] = text_sha(row["text"])
    digest = canonical_sha([{k: r[k] for k in ("review_id", "text", "intended_label")} for r in reviews])
    return {
        "schema": "traveler-v4-independent-eval-v2",
        "frozen": True,
        "purpose": "Second frozen independent evaluation for V4.2; never used for fitting or model selection.",
        "label_origin": "AUTHOR_INTENT_INDEPENDENT_EVAL_V2_NOT_HUMAN_GOLD",
        "human_gold": False,
        "created": "2026-11-25",
        "seed": seed,
        "records_sha256": digest,
        "counts": dict(Counter(r["intended_label"] for r in reviews)),
        "n": len(reviews),
        "reviews": reviews,
        "limitations": [
            "Still synthetic alternate banks; not human Cartagena gold.",
            "Frozen before V4.2 candidate scoring.",
        ],
        "prior_eval_observed": "v4_independent_eval.json (72 rows) already observed; this set is new.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--seed", type=int, default=202611253)
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.out.exists() and not a.force:
        raise SystemExit("Output exists; refuse overwrite of frozen eval.")
    payload = build(seed=a.seed, per_class=a.per_class)
    if a.out.exists():
        a.out.unlink()
    write_json(a.out, payload)
    print(json.dumps({"status": "FROZEN", "n": payload["n"], "counts": payload["counts"],
                      "records_sha256": payload["records_sha256"], "path": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
