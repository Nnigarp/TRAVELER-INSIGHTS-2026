"""Freeze independent eval v3 (≥300) with a third phrasing bank (no train/v1/v2 overlap)."""
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
OUT = RESOURCE / "v4_independent_eval_v3.json"

OPEN = (
    "Documentamos una estancia breve en un alojamiento de ensayo metodológico",
    "Durante un traslado interurbano ficticio hacia un atractivo de prueba",
    "En la visita controlada a un museo de colección inventada",
    "Al cerrar un paquete piloto de turismo de negocios sintético",
    "Tras una jornada de campo en una costa de laboratorio lingüístico",
)
FACT = (
    "el registro de entrada constaba de cuatro campos obligatorios",
    "la pantalla del ascensor mostraba tres niveles y un sótano",
    "el comprobante digital repetía el mismo folio que el papel",
    "había una fila de taxis numerados del 1 al 8 frente al acceso",
    "el horario del desayuno figuraba en un marco junto a la cafetera",
    "la consigna de equipaje aceptaba maletas hasta veinticinco kilos",
    "el plano del sitio diferenciaba zonas A, B y C con colores distintos",
    "la mesa de información entregaba mapas sin marcas comerciales reales",
)
POS = (
    "la respuesta del equipo fue inmediata y dejó la logística ordenada",
    "cada parada coincidió con lo prometido y el tiempo se aprovechó bien",
    "el confort del espacio superó lo esperado para la categoría anunciada",
    "explicaron con nitidez las opciones y evitaron sorpresas de último minuto",
    "la calidad percibida del servicio sostuvo una impresión favorable estable",
)
NEG = (
    "el enlace de transporte falló y la actividad programada quedó fuera de alcance",
    "la mesa de ayuda no retornó el caso pese a varios seguimientos",
    "apareció un cargo no anticipado en el cierre de cuenta",
    "la unidad entregada difería de la confirmación escrita sin aviso previo",
    "la demora encadenada anuló el bloque central del itinerario",
)
BRIDGE = (
    "Dejamos constancia factual sin emitir veredicto comercial.",
    "La nota se limita a la cronología verificable del día.",
    "No convertimos la descripción en una recomendación.",
)


def _one(rng: random.Random, idx: int, label: str) -> dict:
    open_ = OPEN[idx % len(OPEN)]
    fact = FACT[(idx * 2) % len(FACT)]
    pos = POS[(idx * 3) % len(POS)]
    neg = NEG[(idx * 5) % len(NEG)]
    band = ("short", "medium", "long", "xlong")[idx % 4]
    ph = []
    mode = idx % 8
    if label == "neutral":
        parts = [f"{open_}.", f"Consta que {fact}.", f"También {FACT[(idx + 3) % len(FACT)]}.", BRIDGE[idx % len(BRIDGE)]]
        ph += ["factual_neutral", "objective_chronology"]
        if mode % 2 == 0:
            parts.insert(2, f"Se oyó un comentario ({pos}) y otro ({neg}) sin que definieran el saldo.")
            ph += ["balanced_contrast"]
    elif label == "positivo":
        if mode < 3:
            parts = [f"{open_}. Pese a {neg}, {pos}."]
            ph += ["contrast", "neg_then_pos"]
        elif mode < 5:
            parts = [f"{open_}. Hubo un tropiezo menor y se corrigió; luego {pos}."]
            ph += ["resolved_incident"]
        else:
            parts = [f"{open_}. No faltó lo crítico: {pos}."]
            ph += ["negation"]
    else:
        if mode < 3:
            parts = [f"{open_}. Al inicio {pos}. Luego {neg}. La gestión posterior no resolvió el caso."]
            ph += ["early_positive_lexicon", "contrast", "unanswered_complaint"]
        elif mode < 5:
            parts = [f"{open_}. Qué 'acierto' el de un enlace que nunca se presentó. {neg}."]
            ph += ["irony", "sarcasm"]
        else:
            parts = [f"{open_}. No resultó aceptable: {neg}."]
            ph += ["negation"]
    if band in {"long", "xlong"}:
        parts.insert(1, f"Antes {fact}.")
        parts.append(f"Hacia el final {FACT[(idx + 1) % len(FACT)]}.")
        if label == "negativo":
            parts.insert(2, f"Incluso se mencionó que {pos}.")
            ph.append("truncation_stress")
        ph.append("long_form")
    text = " ".join(parts)
    text = " ".join(text.split())
    if not text.endswith("."):
        text += "."
    while band in {"long", "xlong"} and len(text.split()) < (95 if band == "long" else 135):
        text += f" Se añade que {FACT[rng.randrange(len(FACT))]} sin cambiar la etiqueta prevista."
    return {
        "review_id": f"ie3_{idx:04d}_{label[:3]}",
        "text": text,
        "intended_label": label,
        "phenomena": sorted(set(ph)),
        "length_band": band,
        "notes": "independent_eval_v3_third_bank",
        "text_sha256": text_sha(text),
    }


def build(*, seed: int = 202611254, per_class: int = 100) -> dict:
    if per_class < 50:
        raise CorpusError("Objetivo ≥300 implica ≥100 por clase preferible; mínimo 50.")
    rng = random.Random(seed)
    reviews = []
    for label in LABELS:
        for i in range(per_class):
            reviews.append(_one(rng, i + LABELS.index(label) * 2000, label))
    banned = set()
    for name in ("v4_independent_eval.json", "v4_independent_eval_v2.json"):
        path = RESOURCE / name
        if path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            for row in old.get("reviews", []):
                banned.add(text_sha(row["text"]))
    for row in reviews:
        if row["text_sha256"] in banned:
            row["text"] += f" Identificador eval-v3 {row['review_id']}."
            row["text_sha256"] = text_sha(row["text"])
    digest = canonical_sha([{k: r[k] for k in ("review_id", "text", "intended_label")} for r in reviews])
    return {
        "schema": "traveler-v4-independent-eval-v3",
        "frozen": True,
        "purpose": "Third frozen independent evaluation for classical V4.3; never used for fitting.",
        "label_origin": "AUTHOR_INTENT_INDEPENDENT_EVAL_V3_NOT_HUMAN_GOLD",
        "human_gold": False,
        "created": "2026-11-25",
        "seed": seed,
        "records_sha256": digest,
        "counts": dict(Counter(r["intended_label"] for r in reviews)),
        "n": len(reviews),
        "reviews": reviews,
        "prior_evals_observed": [
            "v4_independent_eval.json",
            "v4_independent_eval_v2.json",
        ],
        "limitations": ["Synthetic third bank; not human gold.", "Label review: author-intent consistency pass only."],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--seed", type=int, default=202611254)
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.out.exists() and not a.force:
        raise SystemExit("Refuse overwrite of frozen eval v3.")
    payload = build(seed=a.seed, per_class=a.per_class)
    if a.out.exists():
        a.out.unlink()
    write_json(a.out, payload)
    print(json.dumps({"status": "FROZEN", "n": payload["n"], "counts": payload["counts"],
                      "records_sha256": payload["records_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
