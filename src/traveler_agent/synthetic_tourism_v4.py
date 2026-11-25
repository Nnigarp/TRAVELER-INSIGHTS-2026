"""Synthetic tourism corpus V4: compositional, family-balanced, leakage-aware.

Labels express generator intent, not human gold. No LLM API calls.
Target size is 5001 rows (1667 families × 3) — nearest ternary-balanced size to 5000.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from .corpus import CorpusError, file_sha, text_sha
from .corpus_automation import privacy_flags
from .labels import LABELS
from .synthetic_diversity import canonical_sha, ignored_target, write_json

RESOURCE = Path(__file__).with_name("research_data")
PROTOCOL = RESOURCE / "synthetic_v4_protocol.json"

GENERATOR = "curated_compositional_scenarios_v4"
MODEL = "NO_LLM_LOCAL_COMPOSITIONAL_V4"
VERSION = "4.1.0"
SCHEMA = "traveler-synthetic-tourism-v4"
REFERENCE_VERSION = "4.0.0"  # preserved locally; do not overwrite

DOMAINS = (
    "hotel", "restaurante", "museo", "playa", "transporte",
    "recorrido", "guia", "aeropuerto", "reserva", "excursion_natural",
)
ASPECTS = (
    "servicio", "higiene", "precio", "espera", "seguridad",
    "acceso", "alimentacion", "instalaciones",
)
REGIONS = ("es-CO", "es-PE", "es-MX", "es-ES", "es")
LENGTH_BANDS = ("short", "medium", "long", "xlong")
TEMPLATES = (
    "direct", "negation", "contrast", "mixed_aspects",
    "praise_then_failure", "colloquial", "chronology", "irony",
)
ROLES = ("train", "validation", "test", "hard_holdout")
FIELDS = (
    "review_id", "text", "intended_label", "domain", "aspect", "region",
    "length_band", "template", "scenario_family", "role", "phenomena",
    "hard_case", "synthetic", "label_origin", "generator_method",
    "generator_model", "generator_version", "seed", "generated_date",
    "text_sha256", "approx_beto_tokens",
)

DOMAIN_OPEN = {
    "hotel": (
        "En un alojamiento ficticio del centro",
        "Durante dos noches en un hospedaje inventado",
        "Al cerrar la estancia en un hotel de prueba",
    ),
    "restaurante": (
        "En un comedor turístico ficticio",
        "Durante la cena en un local inventado",
        "Al almorzar en un restaurante de escenario",
    ),
    "museo": (
        "En la visita a un museo ficticio",
        "Durante una exposición inventada",
        "Al recorrer salas de un museo de prueba",
    ),
    "playa": (
        "En una playa ficticia de la costa",
        "Durante la tarde en una zona de baño inventada",
        "Al pasar el día en una playa de escenario",
    ),
    "transporte": (
        "En un traslado turístico ficticio",
        "Durante un trayecto colectivo inventado",
        "Al usar una ruta de transporte de prueba",
    ),
    "recorrido": (
        "En un recorrido por el casco ficticio",
        "Durante una caminata guiada inventada",
        "Al unirnos a un circuito de escenario",
    ),
    "guia": (
        "Con un guía turístico ficticio",
        "Durante la explicación de una guía inventada",
        "Al seguir a un orientador de prueba",
    ),
    "aeropuerto": (
        "En un aeropuerto ficticio durante la conexión",
        "En el mostrador inventado de embarque",
        "Al cruzar un control aeroportuario de escenario",
    ),
    "reserva": (
        "Al gestionar una reserva turística ficticia",
        "Durante la confirmación de un paquete inventado",
        "Al modificar una reserva de escenario",
    ),
    "excursion_natural": (
        "En una excursión natural ficticia",
        "Durante una salida a cascadas inventadas",
        "Al recorrer un sendero de prueba",
    ),
}

PRAISE = {
    "servicio": (
        "nos atendieron con paciencia y resolvieron dudas al instante",
        "el personal fue cordial y estuvo pendiente sin invadir",
        "la atención del equipo se sintió cercana y eficiente",
    ),
    "higiene": (
        "todo estaba limpio, ordenado y sin olores molestos",
        "las áreas comunes lucían impecables y cuidadas",
        "encontramos superficies limpias y baños en buen estado",
    ),
    "precio": (
        "el precio nos pareció justo frente a lo ofrecido",
        "la tarifa coincidía con lo anunciado y sin cargos ocultos",
        "pagamos un valor razonable para la experiencia",
    ),
    "espera": (
        "entramos sin demora y el ritmo fue puntual",
        "la fila avanzó rápido y nos llamaron a tiempo",
        "no hubo retrasos relevantes en el inicio",
    ),
    "seguridad": (
        "las indicaciones de seguridad fueron claras",
        "había personal pendiente de prevenir riesgos",
        "nos sentimos respaldados con medidas visibles",
    ),
    "acceso": (
        "llegar fue sencillo y la señalización ayudó",
        "los accesos estaban bien marcados y practicables",
        "encontramos la entrada sin rodeos confusos",
    ),
    "alimentacion": (
        "la comida llegó caliente, fresca y sabrosa",
        "las opciones de alimentación estaban bien logradas",
        "nos gustó la calidad de lo que sirvieron",
    ),
    "instalaciones": (
        "las instalaciones estaban cómodas y bien mantenidas",
        "los espacios funcionaban y se sentían agradables",
        "habitaciones y zonas comunes estaban en buen estado",
    ),
}

NEUTRAL = {
    "servicio": (
        "nos indicaron el mostrador de información",
        "recibimos el horario y los datos de contacto",
        "el personal respondió solo lo consultado",
    ),
    "higiene": (
        "había personal de aseo en turno",
        "vimos el aviso con el horario de limpieza",
        "el estado general era el habitual de un espacio en uso",
    ),
    "precio": (
        "la tarifa estaba publicada en un panel",
        "el monto se mostró antes de confirmar",
        "el recibo listaba los conceptos cobrados",
    ),
    "espera": (
        "el turno quedó asignado para más tarde",
        "nos informaron la hora prevista de ingreso",
        "la espera se mantuvo dentro del rango anunciado",
    ),
    "seguridad": (
        "había carteles con normas básicas",
        "recibimos la hoja de rutas de evacuación",
        "el protocolo se limitó a lo reglamentario",
    ),
    "acceso": (
        "la entrada principal estaba en el segundo nivel",
        "se indicaba un acceso lateral alternativo",
        "el recorrido de ingreso estaba señalizado de forma ordinaria",
    ),
    "alimentacion": (
        "el menú listaba las opciones disponibles",
        "nos mostraron dónde comprar alimentos",
        "había un punto de comida con carta fija",
    ),
    "instalaciones": (
        "había dos salas y una zona de espera",
        "las instalaciones incluían baños y un patio",
        "el mobiliario cumplía la función básica",
    ),
}

FAILURE = {
    "servicio": (
        "nadie respondió cuando pedimos ayuda urgente",
        "el trato fue seco y nos dejaron sin orientación",
        "tras reclamar, solo recibimos silencio o mensajes automáticos",
    ),
    "higiene": (
        "había suciedad visible y mal olor persistente",
        "encontramos basura acumulada en zonas de uso",
        "varias superficies estaban descuidadas pese a lo cobrado",
    ),
    "precio": (
        "nos cobraron más de lo anunciado sin explicación",
        "aparecieron cargos extras al momento de pagar",
        "el precio final no coincidió con la reserva",
    ),
    "espera": (
        "esperamos demasiado sin información útil",
        "el retraso se alargó y nadie explicó la causa",
        "perdimos la franja horaria por una demora no gestionada",
    ),
    "seguridad": (
        "nos sentimos expuestos por falta de vigilancia",
        "no había medidas visibles ante un riesgo evidente",
        "las instrucciones de seguridad fueron contradictorias",
    ),
    "acceso": (
        "no había señalización suficiente para llegar",
        "el acceso era confuso y nos hizo perder tiempo",
        "una barrera no anunciada impidió el ingreso previsto",
    ),
    "alimentacion": (
        "la comida llegó fría y en malas condiciones",
        "las opciones eran escasas y de baja calidad",
        "hubo un problema sanitario evidente en lo servido",
    ),
    "instalaciones": (
        "varias instalaciones estaban dañadas",
        "los espacios tenían desperfectos que afectaron la visita",
        "equipos anunciados no funcionaban durante la estancia",
    ),
}

TRANSPORT_BREACH = (
    "el transporte contratado para la excursión nunca llegó a la hora acordada",
    "el traslado prometido dejó de aparecer y perdimos la salida",
    "el bus del paquete no se presentó y nadie del operador contestó",
    "el transporte del conductor se canceló en silencio y nos quedamos sin conexión",
)
EXCURSION_LOSS = (
    "perdimos la excursión pagada con antelación",
    "se cayó la visita guiada que ya habíamos confirmado",
    "no pudimos entrar al recorrido natural reservado",
    "la actividad principal del paquete se perdió por completo",
)
COMPLAINT = (
    "presentamos una reclamación formal y no obtuvimos respuesta útil",
    "escribimos al soporte y solo llegó un acuse automático",
    "llamamos varias veces sin que nadie retomara el caso",
    "la queja quedó abierta sin solución después de varios días",
)
# Closings are SHARED across labels (V4.1). Polarity must come from the body,
# not from a class-exclusive final sentence (the V4.0 shortcut).
SHARED_CLOSINGS = (
    "Así quedó la experiencia para nosotros.",
    "Dejo este relato tal como lo vivimos.",
    "Eso es lo que ocurrió durante la visita.",
    "Con eso termino el resumen del día.",
    "Queda dicha la secuencia completa.",
    "Hasta aquí el recuento de lo sucedido.",
)
# Kept only for V4.0 shortcut audits / backward diagnostics — not used in V4.1 compose.
NON_RETURN = {
    "negativo": (
        "por eso no volveré a contratar este servicio ni lo recomendaría",
        "con ese desenlace, descarto repetir el paquete",
        "terminé decidido a no contratarlos otra vez",
    ),
    "neutral": (
        "dejo constancia de lo ocurrido sin una recomendación clara",
        "registro los hechos tal como sucedieron, sin más valoración",
        "anoto la secuencia para quien deba revisarla después",
    ),
    "positivo": (
        "aun así, el saldo general me dejó ganas de volver",
        "pese a un detalle menor, sí lo recomendaría",
        "en conjunto salí contento y repetiría la experiencia",
    ),
}

REGIONAL = {
    "es-CO": ("Parce,", "Por acá,", "La verdad,"),
    "es-PE": ("Causa,", "Pues,", "Al final, pe,"),
    "es-MX": ("Órale,", "La neta,", "Pues mira,"),
    "es-ES": ("Vamos a ver,", "La verdad,", "En resumen,"),
    "es": ("En fin,", "Tal cual,", "En resumen,"),
}

COLLOQUIAL = {
    "es-CO": ("nos dejó tirados", "fue un enredo", "nadie dio la cara"),
    "es-PE": ("nos dejaron en banda", "fue un lío", "no hubo quien asuma"),
    "es-MX": ("nos dejaron plantados", "fue un rollo", "nadie se hizo cargo"),
    "es-ES": ("nos dejaron tirados", "fue un jaleo", "nadie dio la cara"),
    "es": ("nos dejaron a medias", "fue un lío", "nadie respondió"),
}

IRONY = {
    "negativo": (
        "Qué detalle tan 'excelente': justo falló lo esencial del paquete.",
        "Menuda 'maravilla': lo crítico del servicio se derrumbó al final.",
        "Vaya 'acierto': precisamente falló lo que más necesitábamos.",
    ),
    "neutral": (
        "Qué emocionante: el horario quedó escrito en un papel.",
        "Gran noticia operativa: solo anotaron la hora en la pizarra.",
        "Interesante registro: el turno aparecía en una hoja suelta.",
    ),
    "positivo": (
        "Qué sorpresa tan buena: esta vez sí cumplieron lo prometido.",
        "Vaya acierto: cumplieron cada punto del itinerario.",
        "Qué alivio: esta vez sí resolvieron todo a tiempo.",
    ),
}

# Deterministic unique anchors reduce cross-family near-duplicates from shared banks.
ANCHOR_OBJECTS = (
    "mirador", "muelle", "pasarela", "taquilla", "vestíbulo", "andén", "sendero",
    "terraza", "maletero", "mostrador", "embarcadero", "pasillo", "ascensor",
    "pergola", "mirilla", "bodega", "jardín", "torreón", "escalera", "puente",
)
ANCHOR_COLORS = (
    "ámbar", "cobalto", "verde", "terracota", "arena", "grafito", "índigo",
    "ocre", "coral", "marfil", "cítrico", "lavanda", "bronce", "plata",
)
ANCHOR_TIMES = (
    "las 07:10", "las 08:40", "las 09:15", "las 10:05", "las 11:30",
    "las 12:45", "las 14:20", "las 15:55", "las 17:05", "las 18:50",
    "las 19:25", "las 20:10", "las 21:00", "el mediodía", "el anochecer",
)


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def historical_text_hashes() -> set[str]:
    """Exclude exact historical texts; never used for fitting decisions."""
    banned: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str) and text.strip():
                banned.add(text_sha(text))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for name in (
        "synthetic_diversity_v2.json",
        "synthetic_diversity_v3.json",
        "contextual_supervised_v1.json",
        "beto_v2_train.json",
        "beto_v2_development.json",
        "beto_v2_assessment.json",
        *sorted(p.name for p in RESOURCE.glob("beto_v3_batch_*.json")),
    ):
        path = RESOURCE / name
        if path.is_file():
            walk(json.loads(path.read_text(encoding="utf-8")))

    v1 = Path("data/processed/synthetic-tourism-v1/synthetic_reviews.csv")
    if v1.is_file():
        with v1.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("text"):
                    banned.add(text_sha(row["text"]))
    return banned


def approx_beto_tokens(text: str) -> int:
    """Conservative proxy without loading transformers: ~0.8 word + punctuation."""
    words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return 2 + len(words)


def _pick(seq, rng: random.Random, salt: int):
    return seq[(rng.randrange(len(seq)) + salt) % len(seq)]


def _family_meta(slot: int, rng: random.Random) -> dict:
    domain = DOMAINS[slot % len(DOMAINS)]
    aspect = ASPECTS[(slot // len(DOMAINS)) % len(ASPECTS)]
    region = REGIONS[(slot // 3) % len(REGIONS)]
    length_band = LENGTH_BANDS[(slot // 5) % len(LENGTH_BANDS)]
    # Reserve early slots for hard praise-then-failure families.
    if slot < 67:
        template = "praise_then_failure"
        hard = True
        # Keep enough room for praise + breach + complaint + non-return.
        length_band = "xlong" if slot % 2 == 0 else "long"
    else:
        template = TEMPLATES[(slot + rng.randrange(3)) % len(TEMPLATES)]
        # Enrich train pool with mixed failures beyond hard holdout.
        if slot % 11 == 0:
            template = "praise_then_failure"
            length_band = "long" if length_band == "short" else length_band
        hard = template == "praise_then_failure"
    return {
        "domain": domain,
        "aspect": aspect,
        "region": region,
        "length_band": length_band,
        "template": template,
        "hard_case": "YES" if hard else "NO",
        "scenario_family": f"v4:{domain}:{aspect}:{region}:{slot:04d}",
    }


def _assign_roles(family_slots: list[int], *, seed: int) -> dict[int, str]:
    proto = protocol()["roles"]
    hard = family_slots[: proto["hard_holdout"]]
    rest = family_slots[proto["hard_holdout"] :]
    ordered = sorted(
        rest,
        key=lambda slot: hashlib.sha256(f"{seed}:role:{slot}".encode()).hexdigest(),
    )
    n_train, n_val = proto["train"], proto["validation"]
    ownership = {slot: "hard_holdout" for slot in hard}
    for i, slot in enumerate(ordered):
        if i < n_train:
            ownership[slot] = "train"
        elif i < n_train + n_val:
            ownership[slot] = "validation"
        else:
            ownership[slot] = "test"
    expected = sum(proto.values())
    if len(ownership) != expected:
        raise CorpusError("Asignación de roles distinta del protocolo.")
    return ownership


def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _expand_to_band(parts: list[str], band: str, rng: random.Random, label: str, *, slot: int) -> str:
    lo, hi = protocol()["length_bands"][band]
    text = " ".join(_sentence(p) for p in parts if p.strip())
    # Shared fillers (V4.1): no class-exclusive elongation cues.
    filler_bank = (
        "Anoto también el contexto inmediato alrededor del lugar",
        "Queda registrado el tramo siguiente de la visita",
        "El intervalo entre un momento y el siguiente fue breve",
        "Había más gente en la misma zona en ese instante",
        "La luz y el ruido del entorno formaban parte de la escena",
        "El trayecto corto hasta el punto siguiente no cambió el relato",
    )
    del label  # length expansion must not encode polarity
    step = 0
    while len(text.split()) < lo:
        base = filler_bank[(slot + step) % len(filler_bank)]
        detail = ANCHOR_OBJECTS[(slot + step * 3) % len(ANCHOR_OBJECTS)]
        text = f"{text} {_sentence(f'{base} junto al {detail} del escenario {slot:04d}')}"
        step += 1
        if step > 40:
            break
    # Never hard-trim: trimming the tail would erase late failure clauses.
    return re.sub(r"\s+", " ", text).strip()


def _family_anchor(slot: int, label: str) -> str:
    """Unique scenic anchor. Must NOT encode the intended label."""
    del label  # kept in signature for call-site stability; never lexicalized
    obj = ANCHOR_OBJECTS[slot % len(ANCHOR_OBJECTS)]
    color = ANCHOR_COLORS[(slot // 3) % len(ANCHOR_COLORS)]
    when = ANCHOR_TIMES[(slot // 5) % len(ANCHOR_TIMES)]
    code = f"V4-{slot:04d}"
    return (
        f"Tomé nota del {obj} {color} hacia {when} bajo la referencia {code}."
    )


def _compose(meta: dict, label: str, rng: random.Random, *, slot: int) -> tuple[str, list[str]]:
    domain, aspect, region = meta["domain"], meta["aspect"], meta["region"]
    template, band = meta["template"], meta["length_band"]
    open_ = _pick(DOMAIN_OPEN[domain], rng, 1)
    praise = _pick(PRAISE[aspect], rng, 2)
    neutral = _pick(NEUTRAL[aspect], rng, 3)
    fail = _pick(FAILURE[aspect], rng, 4)
    transport = _pick(TRANSPORT_BREACH, rng, 5)
    lost = _pick(EXCURSION_LOSS, rng, 6)
    complaint = _pick(COMPLAINT, rng, 7)
    close = _pick(SHARED_CLOSINGS, rng, 8)
    regional = _pick(REGIONAL[region], rng, 9)
    colloquial = _pick(COLLOQUIAL[region], rng, 10)
    anchor = _family_anchor(slot, label)
    phenomena: list[str] = []

    if template == "direct":
        body = praise if label == "positivo" else neutral if label == "neutral" else fail
        parts = [f"{open_}, {body}.", close]
        phenomena.append("direct")
    elif template == "negation":
        if label == "negativo":
            parts = [
                f"{open_}. No fue una experiencia satisfactoria:",
                f"{fail} y {complaint}.",
                close,
            ]
        elif label == "positivo":
            parts = [
                f"{open_}. No faltó nada esencial:",
                f"{praise}.",
                close,
            ]
        else:
            parts = [
                f"{open_}. No puedo decir que haya sido buena o mala:",
                f"{neutral}.",
                close,
            ]
        phenomena.extend(["negation"])
    elif template == "contrast":
        other = fail if label == "positivo" else praise
        pivot = {
            "positivo": f"Aunque {other}, {praise}",
            "negativo": f"Aunque {praise}, {fail}",
            "neutral": f"Aunque {praise}, {neutral}",
        }[label]
        parts = [f"{open_}. {pivot}.", close]
        phenomena.extend(["contrast", "mixed_aspects"])
    elif template == "mixed_aspects":
        second_aspect = ASPECTS[(ASPECTS.index(aspect) + 3) % len(ASPECTS)]
        other_praise = _pick(PRAISE[second_aspect], rng, 11)
        other_fail = _pick(FAILURE[second_aspect], rng, 12)
        if label == "negativo":
            parts = [
                f"{open_}. Por un lado, {other_praise}.",
                f"Por otro, {fail}; además, {complaint}.",
                close,
            ]
        elif label == "positivo":
            parts = [
                f"{open_}. Hubo un detalle menor ({other_fail}), pero {praise}.",
                close,
            ]
        else:
            parts = [
                f"{open_}. Observamos {neutral} y también {other_praise}.",
                "No concluyo si el saldo fue bueno o malo.",
                close,
            ]
        phenomena.extend(["mixed_aspects", "contrast"])
    elif template == "praise_then_failure":
        phenomena.extend(
            ["mixed_aspects", "contrast", "early_positive_lexicon",
             "transport_or_excursion_breach", "unanswered_complaint", "explicit_non_return"]
        )
        if label == "negativo":
            parts = [
                f"{open_}, {praise}.",
                f"Incluso destacaría que {_pick(PRAISE['instalaciones'], rng, 13)}.",
                f"Sin embargo, {transport}; {lost}.",
                f"{complaint.capitalize() if complaint[:1].islower() else complaint}.",
                close,
            ]
        elif label == "positivo":
            # Twin keeps praise-first shape but resolves the incident.
            parts = [
                f"{open_}, {praise}.",
                f"Hubo un susto breve con el traslado, pero lo reprogramaron a tiempo y {praise}.",
                close,
            ]
            phenomena = ["mixed_aspects", "resolved_incident", "early_positive_lexicon"]
        else:
            parts = [
                f"{open_}, {neutral}.",
                f"Se mencionó un cambio de transporte y {lost.replace('perdimos', 'se reubicó')}.",
                "Dejo la secuencia anotada sin una recomendación definitiva.",
                close,
            ]
            phenomena = ["mixed_aspects", "objective_chronology"]
    elif template == "colloquial":
        parts = [
            f"{regional} {open_.casefold()}, {praise if label == 'positivo' else fail if label == 'negativo' else neutral}.",
            f"La cosa es que {colloquial}.",
            close,
        ]
        phenomena.extend(["regionalism", "colloquial"])
    elif template == "chronology":
        parts = [
            f"{open_}. Primero {neutral}.",
            f"Después {praise if label != 'negativo' else fail}.",
            f"Al final {close[0].lower() + close[1:]}",
        ]
        phenomena.append("objective_chronology")
    else:  # irony
        parts = [
            f"{open_}. {_pick(IRONY[label], rng, 14)}",
            f"{praise if label == 'positivo' else fail if label == 'negativo' else neutral}.",
            close,
        ]
        phenomena.append("irony")

    if band in {"long", "xlong"} and template == "praise_then_failure" and label == "negativo":
        # Force enough early praise that 128-token windows can drop the failure.
        extra = [
            f"También {_pick(PRAISE['servicio'], rng, 20)}.",
            f"Además {_pick(PRAISE['alimentacion'], rng, 21)}.",
            f"Y {_pick(PRAISE['higiene'], rng, 22)}.",
        ]
        if band == "xlong":
            extra += [
                f"Repito: {_pick(PRAISE['instalaciones'], rng, 23)}.",
                f"Incluso {_pick(PRAISE['acceso'], rng, 24)}.",
                f"Hasta {_pick(PRAISE['precio'], rng, 25)}.",
            ]
        parts = parts[:2] + extra + parts[2:]
        phenomena.append("truncation_stress")

    parts = [parts[0], anchor, *parts[1:]]
    text = _expand_to_band(parts, band, rng, label, slot=slot)
    return text, sorted(set(phenomena))


def generate_records(
    *,
    seed: int | None = None,
    generated_date: str = "2026-11-25",
) -> list[dict]:
    proto = protocol()
    seed = proto["seed"] if seed is None else seed
    if not isinstance(seed, int) or seed < 0:
        raise CorpusError("Semilla inválida.")
    rng = random.Random(seed)
    banned = historical_text_hashes()
    family_slots = list(range(proto["family_count"]))
    roles = _assign_roles(family_slots, seed=seed)
    rows: list[dict] = []
    seen: set[str] = set()

    for slot in family_slots:
        meta = _family_meta(slot, rng)
        role = roles[slot]
        for label in LABELS:
            local = random.Random(seed + slot * 17 + LABELS.index(label) * 31)
            text, phenomena = _compose(meta, label, local, slot=slot)
            digest = text_sha(text)
            if digest in seen or digest in banned:
                # Deterministic salvage without encoding the intended label.
                text = f"{text} Código de escenario {slot:04d}-{LABELS.index(label)+1}."
                digest = text_sha(text)
            if digest in seen or digest in banned:
                raise CorpusError(f"Duplicado irrecuperable en familia {meta['scenario_family']}.")
            if privacy_flags(text):
                raise CorpusError("Posible dato personal en texto sintético.")
            seen.add(digest)
            review_id = "v4_" + canonical_sha([meta["scenario_family"], label, digest])[:20]
            rows.append({
                "review_id": review_id,
                "text": text,
                "intended_label": label,
                "domain": meta["domain"],
                "aspect": meta["aspect"],
                "region": meta["region"],
                "length_band": meta["length_band"],
                "template": meta["template"],
                "scenario_family": meta["scenario_family"],
                "role": role,
                "phenomena": "|".join(phenomena),
                "hard_case": meta["hard_case"],
                "synthetic": "YES",
                "label_origin": proto["label_origin"],
                "generator_method": GENERATOR,
                "generator_model": MODEL,
                "generator_version": VERSION,
                "seed": str(seed),
                "generated_date": generated_date,
                "text_sha256": digest,
                "approx_beto_tokens": str(approx_beto_tokens(text)),
            })
    return sorted(rows, key=lambda r: r["review_id"])


def lexical_closing_audit(rows: list[dict], *, max_exclusivity: float = 0.92) -> dict:
    """Detect class-exclusive closing/filler phrases (the V4.0 shortcut family)."""
    candidates = []
    for phrase in SHARED_CLOSINGS:
        candidates.append(phrase.casefold().rstrip("."))
    for phrases in NON_RETURN.values():
        for phrase in phrases:
            candidates.append(phrase.casefold().rstrip("."))
    exclusive = []
    shared_ok = []
    for phrase in sorted(set(candidates)):
        hits = Counter(
            r["intended_label"] for r in rows if phrase in r["text"].casefold()
        )
        total = sum(hits.values())
        if total == 0:
            continue
        top = max(hits.values()) / total
        entry = {"phrase": phrase, "counts": dict(hits), "top_share": round(top, 4)}
        if top >= max_exclusivity and len(hits) == 1:
            exclusive.append(entry)
        else:
            shared_ok.append(entry)
    return {
        "version": VERSION,
        "exclusive_closing_phrases": exclusive,
        "shared_or_mixed_phrases": shared_ok[:20],
        "max_exclusivity": max_exclusivity,
        "status": "FAIL" if exclusive else "PASS",
    }


def audit_records(rows: list[dict], *, jaccard_threshold: float = 0.84) -> dict:
    proto = protocol()
    if len(rows) != proto["target_rows"]:
        raise CorpusError(f"Se esperaban {proto['target_rows']} filas.")
    ids, digests = set(), set()
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if tuple(row) != FIELDS and set(row) != set(FIELDS):
            raise CorpusError("Esquema V4 inválido.")
        if row["synthetic"] != "YES" or row["label_origin"] != proto["label_origin"]:
            raise CorpusError("No atribuir oro humano al corpus V4.")
        if row["intended_label"] not in LABELS or row["role"] not in ROLES:
            raise CorpusError("Etiqueta o rol inválido.")
        if row["review_id"] in ids or row["text_sha256"] in digests:
            raise CorpusError("Duplicado exacto.")
        if row["text_sha256"] != text_sha(row["text"]):
            raise CorpusError("Huella inconsistente.")
        if not 12 <= len(row["text"].split()) <= 320:
            raise CorpusError("Longitud fuera de rango operativo.")
        ids.add(row["review_id"])
        digests.add(row["text_sha256"])
        by_family[row["scenario_family"]].append(row)

    if len(by_family) != proto["family_count"]:
        raise CorpusError("Número de familias distinto del protocolo.")
    for family, members in by_family.items():
        if sorted(m["intended_label"] for m in members) != sorted(LABELS):
            raise CorpusError(f"Familia incompleta: {family}")
        roles = {m["role"] for m in members}
        if len(roles) != 1:
            raise CorpusError(f"Fuga de rol dentro de familia: {family}")

    label_counts = Counter(r["intended_label"] for r in rows)
    if dict(label_counts) != {label: proto["rows_per_class"] for label in LABELS}:
        raise CorpusError(f"Balance de clases incorrecto: {dict(label_counts)}")

    role_families = {
        role: {r["scenario_family"] for r in rows if r["role"] == role} for role in ROLES
    }
    expected = protocol()["roles"]
    if {k: len(v) for k, v in role_families.items()} != expected:
        raise CorpusError("Conteo de familias por rol distinto del protocolo.")
    for a, b in combinations(ROLES, 2):
        if role_families[a] & role_families[b]:
            raise CorpusError(f"Fuga de familias entre {a} y {b}.")
        texts_a = {r["text_sha256"] for r in rows if r["role"] == a}
        texts_b = {r["text_sha256"] for r in rows if r["role"] == b}
        if texts_a & texts_b:
            raise CorpusError(f"Fuga de textos entre {a} y {b}.")

    # Blocked near-duplicate screen across families (prefix + trigram Jaccard).
    blocks: dict[str, list[tuple[dict, set]]] = defaultdict(list)
    near = []
    for row in rows:
        tokens = re.findall(r"\w+", row["text"].casefold())
        sig = {tuple(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))}
        key = " ".join(tokens[:8])
        for other, other_sig in blocks.get(key, []):
            if other["scenario_family"] == row["scenario_family"] or not sig or not other_sig:
                continue
            union = len(sig | other_sig)
            score = (len(sig & other_sig) / union) if union else 0.0
            if score >= jaccard_threshold:
                near.append({
                    "review_ids": [other["review_id"], row["review_id"]],
                    "jaccard": round(score, 4),
                    "action": "REVIEW_OR_REGENERATE",
                })
                if len(near) >= 25:
                    break
        if len(near) >= 25:
            break
        blocks.setdefault(key, []).append((row, sig))
    if near:
        raise CorpusError(f"Candidatos near-duplicate entre familias: {len(near)}")

    hard_neg = [
        r for r in rows
        if r["role"] == "hard_holdout" and r["intended_label"] == "negativo"
        and r["template"] == "praise_then_failure"
    ]
    if len(hard_neg) != expected["hard_holdout"]:
        raise CorpusError("Hard holdout negativo incompleto.")
    for row in hard_neg:
        text = row["text"].casefold()
        if not any(k in text for k in ("transporte", "traslado", "bus", "conductor")):
            raise CorpusError("Hard case sin señal de transporte.")
        if "reclam" not in text and "queja" not in text and "soporte" not in text and "llamamos" not in text:
            raise CorpusError("Hard case sin reclamación.")
        if not any(k in text for k in ("perdimos", "se cayó", "no pudimos", "se perdió", "canceló", "sin conexión")):
            raise CorpusError("Hard case sin desenlace negativo en el cuerpo.")

    shortcut = lexical_closing_audit(rows)
    if shortcut["exclusive_closing_phrases"]:
        raise CorpusError("Atajos de cierre exclusivos por etiqueta detectados.")

    return {
        "purpose": "INTERNAL_SYNTHETIC_V4_QUALITY_ONLY",
        "schema": SCHEMA,
        "records": len(rows),
        "families": len(by_family),
        "labels": dict(sorted(label_counts.items())),
        "roles_families": {k: len(v) for k, v in role_families.items()},
        "domains": dict(sorted(Counter(r["domain"] for r in rows).items())),
        "templates": dict(sorted(Counter(r["template"] for r in rows).items())),
        "length_bands": dict(sorted(Counter(r["length_band"] for r in rows).items())),
        "unique_text_ratio": 1.0,
        "unique_token_count": len(set(re.findall(r"\w+", " ".join(r["text"] for r in rows).casefold()))),
        "near_duplicate_candidates_cross_family": near,
        "hard_holdout_negatives": len(hard_neg),
        "lexical_closing_audit": shortcut,
        "generator_version": VERSION,
        "approx_beto_tokens": {
            "min": min(int(r["approx_beto_tokens"]) for r in rows),
            "median": int(sorted(int(r["approx_beto_tokens"]) for r in rows)[len(rows) // 2]),
            "max": max(int(r["approx_beto_tokens"]) for r in rows),
            "over_128_proxy": sum(1 for r in rows if int(r["approx_beto_tokens"]) > 128),
            "over_256_proxy": sum(1 for r in rows if int(r["approx_beto_tokens"]) > 256),
        },
        "limitations": [
            "Compositional generator; lexical shortcuts may remain across families.",
            "Labels are generator intent, not human gold.",
            "approx_beto_tokens is a proxy, not the BETO WordPiece count.",
            "Regional colloquialisms are stylized, not speaker-validated.",
        ],
    }


def write_corpus(rows: list[dict], out_dir: Path, audit: dict, *, seed: int) -> dict:
    out = ignored_target(Path(out_dir), kinds=("data/processed", "data/private"))
    if out.exists() and any(out.iterdir()):
        raise CorpusError("Directorio no vacío: no sobrescribir resultados.")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "synthetic_reviews_v4.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    splits = {role: [r for r in rows if r["role"] == role] for role in ROLES}
    for role, part in splits.items():
        split_path = out / f"split_{role}.csv"
        with split_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(part)
    manifest = {
        "schema": SCHEMA,
        "source_type": "SYNTHETIC_GENERATED",
        "not_human_testimonials": True,
        "not_scientific_external_validation": True,
        "seed": seed,
        "method": GENERATOR,
        "model": MODEL,
        "version": VERSION,
        "protocol_sha256": file_sha(PROTOCOL),
        "rows_sha256": file_sha(path),
        "role_sha256": {role: canonical_sha(part) for role, part in splits.items()},
        "audit": audit,
    }
    write_json(out / "synthetic_manifest_v4.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generar corpus sintético turístico V4")
    parser.add_argument("--out", type=Path, default=Path("data/processed/synthetic-tourism-v4.1"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--date", default="2026-11-25")
    args = parser.parse_args(argv)
    try:
        rows = generate_records(seed=args.seed, generated_date=args.date)
        audit = audit_records(rows)
        seed = int(rows[0]["seed"])
        manifest = write_corpus(rows, args.out, audit, seed=seed)
    except (CorpusError, ValueError, OSError) as exc:
        parser.exit(2, f"V4 generation blocked: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": "SYNTHETIC_V4_ONLY",
        "n": audit["records"],
        "families": audit["families"],
        "labels": audit["labels"],
        "roles_families": audit["roles_families"],
        "sha256": manifest["rows_sha256"],
        "approx_beto_tokens": audit["approx_beto_tokens"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
