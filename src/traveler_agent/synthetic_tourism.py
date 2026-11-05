"""Controlled original synthetic tourist-review research corpus.

All reviews are invented. Labels express the generator's intended sentiment,
not independent human annotation. No LLM API, scraping or paid dependency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from .corpus import CorpusError, file_sha, text_sha
from .corpus_automation import privacy_flags

GENERATOR = "curated_compositional_scenarios_v1"
MODEL = "NO_LLM_LOCAL_RULE_BASED"
VERSION = "1.0.0"
CATEGORIES = ("hotel", "restaurante", "museo", "playa", "transporte", "recorrido", "guia")
ASPECTS = ("servicio", "higiene", "precio", "espera", "seguridad", "acceso",
           "alimentacion", "instalaciones")
LABELS = ("negativo", "neutral", "positivo")
REGIONS = ("colombia", "peru", "general")
FIELDS = ("review_id", "text", "target_label", "category", "aspect", "scenario_family",
          "style", "region", "hard_case", "synthetic", "label_origin", "generator_method",
          "generator_model", "generator_version", "seed", "generated_date", "text_sha256")

# Domains include actual plausible context, without suggesting real businesses.
CONTEXT = {
    "hotel": (
        "Durante una noche en un alojamiento ficticio cerca del centro",
        "Al llegar a un hospedaje imaginario después del viaje",
        "En un hotel inventado durante nuestras vacaciones",
    ),
    "restaurante": (
        "Mientras almorzábamos en un restaurante ficticio",
        "En un comedor imaginario junto a la plaza",
        "Durante la cena en un local inventado para esta prueba",
    ),
    "museo": (
        "En la visita a un museo ficticio de historia local",
        "Durante una exposición inventada para este escenario",
        "Mientras recorríamos las salas de un museo imaginario",
    ),
    "playa": (
        "En una playa ficticia durante una excursión",
        "Mientras pasábamos la tarde en una costa imaginaria",
        "Al visitar una zona de playa inventada para la prueba",
    ),
    "transporte": (
        "En un traslado turístico ficticio hacia el terminal",
        "Durante un viaje imaginario en transporte colectivo",
        "Al tomar una ruta inventada entre atractivos turísticos",
    ),
    "recorrido": (
        "Durante un recorrido turístico ficticio por el casco antiguo",
        "En una caminata guiada imaginaria por varios monumentos",
        "Al participar en una excursión inventada de medio día",
    ),
    "guia": (
        "Durante la explicación de un guía turístico ficticio",
        "Mientras escuchábamos a una guía imaginaria en la visita",
        "Al acompañarnos un orientador inventado para el circuito",
    ),
}
# Each aspect has multiple independently authored factual framings per label.
# Intent is explicit; negations/contrast can be added as a hard case.
ASPECT_TEXT = {
    "servicio": {
        "positivo": ("nos atendieron con paciencia y resolvieron nuestras dudas",
                     "la atención resultó cordial y se interesaron por ayudarnos"),
        "neutral": ("el personal indicó dónde solicitar información",
                    "nos entregaron los datos de contacto del encargado"),
        "negativo": ("nadie respondió cuando pedimos ayuda",
                     "el trato fue descortés y nos dejaron sin orientación"),
    },
    "higiene": {
        "positivo": ("el espacio estaba limpio y cuidado",
                     "encontramos las áreas impecables y ordenadas"),
        "neutral": ("había un equipo encargado de la limpieza",
                    "vimos un aviso con el horario de aseo"),
        "negativo": ("había suciedad visible y malos olores",
                     "encontramos basura acumulada y superficies descuidadas"),
    },
    "precio": {
        "positivo": ("el costo fue justo para lo que ofrecían",
                     "el precio nos pareció razonable frente a la experiencia"),
        "neutral": ("el valor aparecía publicado en una lista",
                    "la tarifa estaba indicada antes de pagar"),
        "negativo": ("nos cobraron más de lo anunciado sin explicarlo",
                     "el precio era excesivo para lo recibido"),
    },
    "espera": {
        "positivo": ("entramos sin demora y todo empezó a tiempo",
                     "la fila avanzó rápido y nos atendieron puntualmente"),
        "neutral": ("el turno quedó programado para la tarde",
                    "nos informaron de la hora prevista para ingresar"),
        "negativo": ("tuvimos que esperar muchísimo sin información",
                     "hubo un retraso largo y nadie explicó la causa"),
    },
    "seguridad": {
        "positivo": ("las indicaciones de seguridad fueron claras y útiles",
                     "había personal pendiente de prevenir riesgos"),
        "neutral": ("había carteles con normas de seguridad",
                    "recibimos instrucciones sobre rutas de emergencia"),
        "negativo": ("nos sentimos inseguros por la falta de vigilancia",
                     "no había medidas visibles ante un riesgo evidente"),
    },
    "acceso": {
        "positivo": ("fue sencillo llegar y la ruta estaba bien señalizada",
                     "la entrada contaba con accesos prácticos y claros"),
        "neutral": ("la entrada principal estaba en el segundo nivel",
                    "se indicaba el acceso por una puerta lateral"),
        "negativo": ("no encontramos señalización para llegar a la entrada",
                     "el acceso era confuso y dificultó nuestro ingreso"),
    },
    "alimentacion": {
        "positivo": ("las opciones de comida estaban frescas y sabrosas",
                     "nos gustó la calidad de los alimentos disponibles"),
        "neutral": ("el menú indicaba las opciones de alimentación",
                    "nos mostraron dónde comprar alimentos durante la visita"),
        "negativo": ("la comida llegó fría y en malas condiciones",
                     "las opciones de alimentación eran escasas y deficientes"),
    },
    "instalaciones": {
        "positivo": ("las instalaciones estaban cómodas y bien mantenidas",
                     "los espacios funcionaban correctamente y eran agradables"),
        "neutral": ("había dos salas y una zona de descanso",
                    "las instalaciones incluían una sala de espera"),
        "negativo": ("varias instalaciones estaban dañadas y sin mantenimiento",
                     "los espacios tenían desperfectos que afectaron la visita"),
    },
}
# The regional vocabulary is consciously limited: it does not claim to model
# authentic regional speech distributions.
REGIONAL = {
    "colombia": ("Qué bacano encontrar algo así", "La verdad, estuvo una nota"),
    "peru": ("Qué chévere fue esa parte", "En general, estuvo bien paja"),
    "general": ("Fue una experiencia particular", "Quedó bastante por comentar"),
}
# Hard cases are always explicit; mixed texts still have a primary target.
HARD_CASES = ("ninguno", "contraste", "negacion", "intensificador",
              "regionalismo", "falta_ortografica", "opinion_mixta", "ironia_simulada")
ENDING = {
    "positivo": ("La impresión final fue favorable.",
                 "Me quedé con una valoración positiva.",
                 "En conjunto, nos gustó la experiencia."),
    "neutral": ("Lo menciono como un dato, sin valorarlo.",
                "Es una descripción de lo ocurrido.",
                "No tengo una valoración favorable ni desfavorable."),
    "negativo": ("La impresión final fue desfavorable.",
                 "Me quedé con una valoración negativa.",
                 "En conjunto, fue una mala experiencia."),
}
ORTHO = {"atención": "atencion", "instalaciones": "instalacoines",
         "información": "informacion", "experiencia": "experiensia"}
NEGATION = {
    "positivo": "No fue para nada una mala experiencia.",
    "neutral": "No puedo decir que haya sido buena o mala.",
    "negativo": "No fue una experiencia satisfactoria.",
}
CONTRAST = {
    "positivo": "Aunque hubo algún detalle menor, mi balance es positivo.",
    "neutral": "Sin embargo, no tengo elementos para evaluarlo.",
    "negativo": "Aunque hubo cosas normales, el balance fue negativo.",
}
IRONY = {
    "positivo": "Qué sorpresa tan agradable, esta vez sí salió bien.",
    "neutral": "Qué emocionante: simplemente vimos el horario indicado.",
    "negativo": "Excelente, justo lo que faltaba: otro problema en la visita.",
}


def _style_text(context: str, statement: str, ending: str, style: int) -> str:
    if style == 0:
        return f"{context}, {statement}. {ending}"
    if style == 1:
        return f"{ending} {context}, {statement}."
    return f"{context}: {statement}; {ending[0].lower() + ending[1:]}"


def _hard_suffix(case: str, sentiment: str, region: str, rng: random.Random) -> str:
    if case == "negacion":
        return NEGATION[sentiment]
    if case in {"contraste", "opinion_mixta"}:
        return CONTRAST[sentiment] if case == "contraste" else (
            "Hubo un detalle distinto, pero mi valoración global sigue siendo "
            + sentiment + "."
        )
    if case == "intensificador":
        return ("De verdad, realmente, muy " +
                ("satisfactorio." if sentiment == "positivo" else
                 "corriente." if sentiment == "neutral" else "decepcionante."))
    if case == "regionalismo":
        return rng.choice(REGIONAL[region]) + ". " + CONTRAST[sentiment]
    if case == "ironia_simulada":
        return IRONY[sentiment]
    return ""


def generate_records(*, seed: int = 42, variants: int = 2,
                     generated_date: str = "2026-11-05") -> list[dict]:
    """336 examples by default, grouped by 56 scenario families.

    The 3 target classes coexist inside EACH family, so a family cannot leak
    its paraphrases into distinct train/validation/test partitions.
    """
    if not isinstance(seed, int) or seed < 0 or variants not in (1, 2, 3):
        raise CorpusError("Semilla debe ser >=0 y variants debe ser 1, 2 o 3.")
    try:
        date.fromisoformat(generated_date)
    except ValueError as exc:
        raise CorpusError("Fecha inválida; usar ISO 8601 YYYY-MM-DD.") from exc
    rng = random.Random(seed)
    result = []
    for category in CATEGORIES:
        for aspect in ASPECTS:
            family = f"synthetic-scenario:{category}:{aspect}"
            region = REGIONS[(CATEGORIES.index(category) + ASPECTS.index(aspect)) % len(REGIONS)]
            for target in LABELS:
                for variant in range(variants):
                    context = CONTEXT[category][(variant + ASPECTS.index(aspect)) % 3]
                    statement = ASPECT_TEXT[aspect][target][variant % 2]
                    ending = ENDING[target][(variant + CATEGORIES.index(category)) % 3]
                    case = HARD_CASES[(CATEGORIES.index(category) * 3
                                       + ASPECTS.index(aspect) + variant) % len(HARD_CASES)]
                    text = _style_text(context, statement, ending, variant)
                    suffix = _hard_suffix(case, target, region, rng)
                    if suffix:
                        text += " " + suffix
                    if case == "falta_ortografica":
                        for proper, misspelling in ORTHO.items():
                            if proper in text:
                                text = text.replace(proper, misspelling, 1)
                                break
                        else:
                            text += " Todo estubo tal como lo describo."
                    review_id = ("syn_" + hashlib.sha256(
                        f"{VERSION}:{seed}:{family}:{target}:{variant}".encode()
                    ).hexdigest()[:20])
                    result.append({
                        "review_id": review_id, "text": text, "target_label": target,
                        "category": category, "aspect": aspect,
                        "scenario_family": family, "style": f"narrative_{variant}",
                        "region": region, "hard_case": case,
                        "synthetic": "YES", "label_origin": "GENERATOR_INTENT_NOT_HUMAN_GOLD",
                        "generator_method": GENERATOR, "generator_model": MODEL,
                        "generator_version": VERSION, "seed": str(seed),
                        "generated_date": generated_date, "text_sha256": text_sha(text),
                    })
    return result


def audit_records(records: list[dict], *, similarity_threshold: float = .94) -> dict:
    if not .85 <= similarity_threshold <= 1:
        raise CorpusError("Umbral de similitud inválido.")
    if not records:
        raise CorpusError("No hay registros sintéticos.")
    ids, seen = set(), {}
    near_cross_family = []
    for row in records:
        if set(row) != set(FIELDS):
            raise CorpusError("Esquema/procedencia sintética inválidos.")
        if row["synthetic"] != "YES" or row["label_origin"] != "GENERATOR_INTENT_NOT_HUMAN_GOLD":
            raise CorpusError("No se admiten etiquetas humanas falsamente atribuidas.")
        if row["target_label"] not in LABELS or row["category"] not in CATEGORIES or row["aspect"] not in ASPECTS:
            raise CorpusError("Etiqueta, dominio o aspecto inválido.")
        if row["review_id"] in ids:
            raise CorpusError("ID repetido.")
        ids.add(row["review_id"])
        if not 40 <= len(row["text"]) <= 700 or not row["text"].endswith("."):
            raise CorpusError("Reseña demasiado corta/larga o incompleta.")
        if row["text_sha256"] != text_sha(row["text"]):
            raise CorpusError("Huella de texto inconsistente.")
        if privacy_flags(row["text"]):
            raise CorpusError("El generador produjo un posible dato personal.")
        key = row["text_sha256"]
        if key in seen:
            raise CorpusError("Duplicado exacto: " + row["review_id"])
        seen[key] = row["scenario_family"]
        # Conservative check: every generated row must retain an explicit,
        # unique source-of-truth target in its ending, not inferred from text.
        if not row["scenario_family"].startswith("synthetic-scenario:"):
            raise CorpusError("Familia de generación no reconocida.")

    # Compare only records in different families with matching first-30-char
    # text prefixes (cheap blocking). This is a CANDIDATE audit, not semantic
    # paraphrase detection or proof of independent folds.
    blocks = {}
    for row in records:
        key = re.sub(r"\s+", " ", row["text"].casefold())[:30]
        for other in blocks.get(key, []):
            if other["scenario_family"] != row["scenario_family"]:
                sim = SequenceMatcher(None, other["text"].casefold(), row["text"].casefold()).ratio()
                if sim >= similarity_threshold:
                    near_cross_family.append({
                        "review_ids": [other["review_id"], row["review_id"]],
                        "similarity": round(sim, 4), "action": "REVIEW_OR_REGROUP",
                    })
        blocks.setdefault(key, []).append(row)
    counts = Counter(r["target_label"] for r in records)
    family_counts = Counter(r["scenario_family"] for r in records)
    # Structural class balance is enforced before training.
    if set(counts) != set(LABELS) or min(counts.values()) < 15:
        raise CorpusError("Balance de clases insuficiente.")
    if len(family_counts) < 15:
        raise CorpusError("Se necesitan al menos 15 familias.")
    return {
        "purpose": "INTERNAL_SYNTHETIC_QUALITY_ONLY",
        "records": len(records), "families": len(family_counts),
        "categories": dict(sorted(Counter(r["category"] for r in records).items())),
        "labels": dict(sorted(counts.items())),
        "hard_cases": dict(sorted(Counter(r["hard_case"] for r in records).items())),
        "regions": dict(sorted(Counter(r["region"] for r in records).items())),
        "unique_text_ratio": len(seen) / len(records),
        "unique_token_count": len(set(re.findall(r"\w+", " ".join(r["text"] for r in records).casefold()))),
        "near_duplicate_candidates_cross_family": near_cross_family,
        "limitations": [
            "Template composition is still artificial and may share lexical patterns.",
            "Candidate near-duplicate detection is blocked and incomplete.",
            "Target labels are intended by generator, NOT human sentiment gold.",
            "Irony and mixed examples are scripted and may contain ambiguous pragmatics.",
        ],
    }


def write_corpus(records: list[dict], out_dir: Path, audit: dict, *, seed: int) -> dict:
    out = Path(out_dir)
    if "processed" not in out.parts and "private" not in out.parts:
        raise CorpusError("Salida debe permanecer en data/processed o data/private.")
    if out.exists() and any(out.iterdir()):
        raise CorpusError("Directorio no vacío: no sobrescribir resultados.")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "synthetic_reviews.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "source_type": "SYNTHETIC_GENERATED",
        "not_human_testimonials": True,
        "not_scientific_external_validation": True,
        "seed": seed, "method": GENERATOR, "model": MODEL, "version": VERSION,
        "rows_sha256": file_sha(path), "audit": audit,
    }
    (out / "synthetic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description="Corpus turístico SINTÉTICO original, sin API ni fuentes externas")
    p.add_argument("--out", type=Path, default=Path("data/processed/synthetic-tourism-v1"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--date", default="2026-11-05")
    a = p.parse_args(argv)
    try:
        records = generate_records(seed=a.seed, variants=a.variants, generated_date=a.date)
        audit = audit_records(records)
        manifest = write_corpus(records, a.out, audit, seed=a.seed)
    except (CorpusError, ValueError, OSError) as exc:
        p.exit(2, f"Synthetic generation blocked: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": "SYNTHETIC_ONLY", "n": audit["records"],
        "families": audit["families"], "labels": audit["labels"],
        "sha256": manifest["rows_sha256"],
        "near_duplicates_for_review": len(audit["near_duplicate_candidates_cross_family"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
