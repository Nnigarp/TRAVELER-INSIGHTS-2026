"""Independent authored-domain challenge, separate from synthetic_tourism generator.

A new standalone inventory of 60 tourism situations, 3 human-authored
INTENDED polarities each, and 7 different narration frames. Result: 1260
artificial records grouped into 180 semantic nuclei. Not real testimonials,
not independent human sentiment annotations, not out-of-domain validation.
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
from pathlib import Path

from .corpus import CorpusError, file_sha, text_sha
from .corpus_automation import privacy_flags

VERSION = "independent_authored_tourism_situations_v1"
SOURCE = "ORIGINAL_SYNTHETIC_RULE_BASED_NO_EXTERNAL_LLM"
FIELDS = ("review_id", "text", "intended_label", "domain", "nucleus_id",
          "variation_id", "phenomenon", "region", "synthetic", "label_origin",
          "generator", "version", "seed", "date", "text_sha256")
LABELS = ("negativo", "neutral", "positivo")

# Each triplet is authored separately. This inventory is deliberately separate
# from the 2026-11-05 generator's CONTEXT, ASPECT_TEXT, ENDING and hard suffixes.
# Per-domain: ten concrete tourism situations; every nucleus can be written in
# three intended polarities. Presence of "neutral" is descriptive, not gold.
SITUATIONS = {
    "hotel": [
        ("El check-in tomó menos de lo anunciado y nos dieron la habitación apenas llegamos",
         "En recepción anotaron la hora a la que nos correspondía ingresar",
         "El registro de entrada quedó enredado y perdimos gran parte de la mañana"),
        ("El ventilador funcionó toda la noche y descansamos a gusto",
         "La habitación contaba con ventilador de techo",
         "El ventilador no funcionó y no pudimos dormir por el calor"),
        ("La ducha mantenía buena presión y el agua salía a temperatura agradable",
         "En el baño había una ducha con regulador de temperatura",
         "La ducha se cortaba cada rato y apenas salía agua"),
        ("Las sábanas olían a recién lavadas y la cama estaba cuidada",
         "En la cama había dos almohadas y una manta",
         "Encontramos manchas viejas en las sábanas al acostarnos"),
        ("La señora de recepción nos ayudó a ubicar una farmacia cercana",
         "La persona de recepción nos entregó el mapa del sector",
         "Pedimos orientación por un problema y nos dejaron hablando solos"),
        ("El ascensor nos llevó sin dificultad a la planta donde nos alojamos",
         "El alojamiento contaba con ascensor para los pisos superiores",
         "El ascensor se averió y nadie ofreció solución a los huéspedes"),
        ("La ventana aislaba el ruido y pudimos descansar hasta tarde",
         "La ventana daba hacia una calle con tránsito",
         "Los vehículos se escuchaban como si no hubiera pared de por medio"),
        ("Nos guardaron el equipaje con cuidado mientras recorríamos la ciudad",
         "Había una consigna de maletas junto a la entrada",
         "Perdieron la etiqueta de nuestra maleta y tardaron horas en encontrarla"),
        ("Los enchufes junto a la cama funcionaban y cargamos nuestros equipos",
         "Había dos tomacorrientes cerca del escritorio",
         "Los enchufes estaban sueltos y era peligroso conectar cualquier cosa"),
        ("La reserva correspondió exactamente a las fotos que habíamos revisado",
         "La página mostraba fotografías y un listado de servicios",
         "Las fotos prometían un cuarto amplio y nos dieron uno diminuto"),
    ],
    "restaurante": [
        ("El pescado llegó jugoso y con el punto de cocción que pedimos",
         "La carta incluía pescado como opción de almuerzo",
         "El pescado estaba crudo por dentro y nadie quiso cambiarlo"),
        ("Nos trajeron las bebidas frías en el momento oportuno",
         "Las bebidas figuraban en la carta de la mesa",
         "Trajeron las bebidas cuando ya habíamos acabado de comer"),
        ("Había alternativas vegetales bien preparadas para todo el grupo",
         "El menú contenía un apartado para platos vegetarianos",
         "Prometieron una alternativa sin carne pero terminaron sirviendo pollo"),
        ("La mesera explicó los ingredientes con paciencia antes de tomar el pedido",
         "En el menú se indicaban los ingredientes del plato",
         "No quisieron aclararnos qué llevaba la comida pese a la alergia"),
        ("El arroz con mariscos estaba sabroso y recién hecho",
         "Ofrecían arroz con mariscos en una sección del menú",
         "El arroz tenía sabor rancio y el plato llegó helado"),
        ("El salón estaba ventilado y era agradable conversar sin gritar",
         "El salón tenía mesas para cuatro personas",
         "La música estaba tan fuerte que no podíamos conversar"),
        ("Nos entregaron la cuenta detallada sin cobros que no reconociéramos",
         "La cuenta separaba los platos y los impuestos",
         "Nos agregaron consumos que nunca hicimos y discutieron al reclamarlos"),
        ("Atendieron una restricción alimentaria y cambiaron la guarnición",
         "El local indicaba que podía adaptar ciertos platos",
         "Ignoraron la advertencia sobre nuestra intolerancia alimentaria"),
        ("La comida llegó en porciones suficientes y compartimos sin quedarnos cortos",
         "La carta señalaba el tamaño aproximado de las porciones",
         "La porción era minúscula para lo que habían ofrecido"),
        ("La sobremesa se disfrutó con una vista tranquila a la plaza",
         "Desde la mesa se veía una plaza al otro lado de la ventana",
         "El olor de la cocina se acumulaba en el comedor sin ventilación"),
    ],
    "playa": [
        ("La marea estaba tranquila y el salvavidas orientaba a los bañistas",
         "En la arena había un letrero con información de mareas",
         "Nadie advertía del oleaje fuerte y varias personas entraron sin saberlo"),
        ("El acceso por la pasarela resultó cómodo para entrar con el coche del bebé",
         "Había una pasarela desde la vía hasta la arena",
         "La única entrada tenía escalones rotos y era imposible bajar con ruedas"),
        ("Encontramos sombra natural y pudimos estar sin sufrir tanto el sol",
         "La zona tenía varias palmeras al borde de la arena",
         "No había refugio del sol y el alquiler de sombrillas era abusivo"),
        ("El agua tenía buena visibilidad y fue agradable nadar allí",
         "La bahía cambiaba de color según la hora del día",
         "Había residuos flotando y no quisimos meternos al mar"),
        ("Las duchas públicas tenían agua y ayudaron después del baño",
         "Cerca de la salida había un módulo de duchas",
         "Las duchas no servían y el lugar estaba lleno de charcos sucios"),
        ("Un vecino nos indicó dónde dejar la basura y mantuvimos limpia la zona",
         "Había recipientes de residuos distribuidos por el malecón",
         "Los tachos desbordaban y la basura llegaba hasta el agua"),
        ("Pudimos caminar por la orilla sin vendedores insistentes",
         "Había vendedores que recorrían la orilla durante el día",
         "Nos perseguían vendiendo cosas aunque repetimos que no queríamos nada"),
        ("Las piedras se veían claramente y el sendero estaba marcado",
         "Una señal indicaba un sendero junto al acantilado",
         "El sendero tenía bordes resbalosos sin advertencias visibles"),
        ("El bote regresó puntual y desembarcamos sin contratiempos",
         "La información del paseo incluía una hora de retorno en bote",
         "El bote se demoró tanto que quedamos varados sin explicación"),
        ("Nos prestaron chalecos del tamaño correcto para ir al islote",
         "Antes de salir se repartían chalecos para el trayecto",
         "Nos mandaron al islote con chalecos dañados y sin ajuste"),
    ],
    "museo": [
        ("La audioguía explicaba las piezas con relatos comprensibles",
         "Había códigos para escuchar una audioguía de las piezas",
         "La audioguía saltaba partes y no se entendía casi nada"),
        ("Los paneles incluían información que ayudó a comprender la época",
         "Los paneles estaban ordenados cronológicamente",
         "Los paneles tenían errores notorios y nadie pudo aclararlos"),
        ("La rampa permitía llegar sin dificultad a las salas superiores",
         "Se indicaba una rampa para subir a otra sala",
         "La rampa estaba clausurada y no existía otra entrada accesible"),
        ("Las piezas estaban iluminadas sin reflejos molestos",
         "Las vitrinas contaban con iluminación interior",
         "Las luces apuntaban a los ojos e impedían observar las obras"),
        ("El taller hizo que los niños participaran activamente y se divirtieran",
         "Había un taller educativo programado para visitantes",
         "Cancelaron el taller a última hora sin avisar a nadie"),
        ("El personal vigilaba las salas con discreción y dio indicaciones útiles",
         "Algunas salas tenían personal de vigilancia",
         "Nadie controlaba el ingreso y vimos que tocaban piezas delicadas"),
        ("El folleto tenía un plano claro y no nos perdimos durante el circuito",
         "En la entrada repartían un folleto del recorrido",
         "El plano estaba desactualizado y terminamos dando vueltas"),
        ("La explicación sobre las comunidades locales fue respetuosa y detallada",
         "La exposición incluía una sección de culturas regionales",
         "Los textos repetían estereotipos ofensivos sobre la población local"),
        ("Había bancos para descansar entre las diferentes exhibiciones",
         "Una sala contaba con una pequeña zona para sentarse",
         "No había dónde sentarse durante un trayecto demasiado largo"),
        ("Las obras tenían etiquetas legibles y pudimos compararlas con calma",
         "Las obras estaban acompañadas de etiquetas descriptivas",
         "Las etiquetas se habían desprendido y no sabíamos qué estábamos viendo"),
    ],
    "transporte": [
        ("El conductor manejó con cuidado incluso en las curvas de la carretera",
         "El recorrido incluía varios tramos de carretera con curvas",
         "El conductor aceleró demasiado cerca de una zona escolar"),
        ("El bus llegó a la terminal cuando lo indicaba el boleto",
         "El pasaje indicaba la hora de llegada a la terminal",
         "El bus apareció mucho después y perdimos la conexión"),
        ("Las maletas quedaron sujetas y llegaron sin golpes",
         "El equipaje se colocaba debajo del asiento asignado",
         "Nuestras maletas viajaron sueltas y una terminó rota"),
        ("La ruta estaba bien anunciada y bajamos cerca del alojamiento",
         "El vehículo tenía un aviso con la ruta prevista",
         "Nadie informó del desvío y acabamos muy lejos del destino"),
        ("El aire acondicionado funcionaba y el viaje fue llevadero",
         "El vehículo contaba con un equipo de aire acondicionado",
         "El aire no funcionaba y el vehículo estaba sofocante"),
        ("El chofer aguardó a que una persona mayor estuviera sentada",
         "Subió una persona mayor antes de que saliéramos",
         "Arrancaron cuando una persona mayor todavía intentaba subir"),
        ("La aplicación mostró una tarifa que se respetó al terminar el trayecto",
         "La tarifa estaba disponible antes de confirmar el trayecto",
         "Nos exigieron pagar el doble de la tarifa anunciada"),
        ("Los asientos tenían espacio suficiente para las piernas",
         "Los asientos estaban distribuidos en dos columnas",
         "Los asientos estaban tan apretados que no podíamos movernos"),
        ("El embarque se organizó con turnos claros y sin empujones",
         "Los pasajeros esperaban por turnos antes de embarcar",
         "La fila se convirtió en empujones porque no había organización"),
        ("El bus tenía indicaciones visibles para salir en emergencia",
         "Había señales de salida de emergencia cerca de las ventanas",
         "Las salidas estaban obstruidas con equipaje durante todo el viaje"),
    ],
    "experiencia": [
        ("El guía narró anécdotas locales sin convertirlas en caricaturas",
         "El itinerario incluía un tramo con narración histórica",
         "El guía inventó datos y se burló de las costumbres del lugar"),
        ("El paseo a pie permitió conocer rincones poco visitados del barrio",
         "La caminata estaba prevista por varias calles del barrio",
         "Nos llevaron por calles cerradas y el recorrido quedó incompleto"),
        ("Pudimos ajustar el ritmo del grupo y nadie quedó rezagado",
         "La caminata avanzó con el grupo por una ruta señalada",
         "Avanzaron sin mirar atrás y dejaron a dos personas perdidas"),
        ("Nos explicaron el origen de las recetas tradicionales con respeto",
         "La actividad incluía una explicación sobre cocina regional",
         "La actividad redujo la cultura local a bromas y comentarios irrespetuosos"),
        ("El bote pequeño permitió observar aves sin acercarse demasiado",
         "El trayecto en bote incluía una zona de observación de aves",
         "El bote se acercaba bruscamente a los animales y los espantaba"),
        ("Las paradas para fotografías fueron suficientes y bien ubicadas",
         "El recorrido preveía varias paradas para fotografías",
         "No permitieron detenernos ni siquiera en los puntos anunciados"),
        ("El taller artesanal explicó técnicas sin presionar para comprar",
         "El programa incluía una visita a un taller artesanal",
         "Insistieron tanto en vendernos recuerdos que apenas hubo explicación"),
        ("El grupo recibió instrucciones claras antes de entrar a la zona natural",
         "Antes de ingresar se entregaban normas del parque",
         "Nos llevaron a la zona natural sin explicar los riesgos del terreno"),
        ("Pudimos conversar con residentes sin interrumpir su trabajo cotidiano",
         "El programa proponía un diálogo con residentes de la zona",
         "Nos animaron a fotografiar personas sin preguntarles y fue incómodo"),
        ("La visita terminaba a una hora razonable y alcanzamos el regreso",
         "La duración anunciada del paseo era de medio día",
         "Alargaron la visita sin avisar y perdimos el transporte de retorno"),
    ],
}
assert len(SITUATIONS) == 6 and all(len(x) == 10 for x in SITUATIONS.values())

# Seven explicitly different discourse constructions; *the same nucleus*
# stays grouped so these forms never become independent observations.
FRAMES = (
    lambda t, c, i, region: f"{t}. Lo vi durante {c}; dejo constancia de ello.",
    lambda t, c, i, region: f"Si alguien pregunta cómo nos fue en {c}, esto diría: {t[0].lower()+t[1:]}.",
    lambda t, c, i, region: f"Fue durante {c}. Primero pensé otra cosa; al final comprobé que {t[0].lower()+t[1:]}.",
    lambda t, c, i, region: f"No quiero exagerar lo que ocurrió en {c}: {t[0].lower()+t[1:]}.",
    lambda t, c, i, region: f"Pues en {c}, para ser preciso, {t[0].lower()+t[1:]}.",
    lambda t, c, i, region: f"¿Cómo resumir lo que pasó en {c}? {t}. Eso fue lo que presencié.",
    lambda t, c, i, region: f"En {c} hubo varios detalles. Entre ellos, {t[0].lower()+t[1:]}.",
)
CONTEXTS = (
    "un viaje con amistades", "una salida del fin de semana",
    "unas vacaciones breves", "una visita familiar", "un paseo por el centro",
    "un recorrido por la costa", "una escapada al interior",
)
REGION_MARKERS = {
    "colombia": ("Oiga, ", "Parce, ", "Ve, "),
    "peru": ("Oye, ", "Pucha, ", "Mira, "),
    "general": ("", "En fin, ", "Por cierto, "),
}


def generate_challenge(*, seed: int = 913, generated_date: str = "2026-11-05") -> list[dict]:
    try:
        date.fromisoformat(generated_date)
    except ValueError as exc:
        raise CorpusError("Fecha ISO 8601 inválida.") from exc
    if not isinstance(seed, int) or seed < 0:
        raise CorpusError("Semilla inválida.")
    rng = random.Random(seed)
    rows = []
    for domain, situations in SITUATIONS.items():
        for index, triplet in enumerate(situations):
            for label, original in zip(LABELS, (triplet[2], triplet[1], triplet[0])):
                group = f"challenge:{domain}:{index:02d}:{label}"
                for variant, frame in enumerate(FRAMES):
                    region = ("colombia", "peru", "general")[(index + variant) % 3]
                    raw = frame(original, CONTEXTS[(index + variant) % len(CONTEXTS)],
                                index, region)
                    if variant in (1, 4):
                        raw = rng.choice(REGION_MARKERS[region]) + raw
                    if variant == 5:
                        raw = raw.replace("porque", "xq").replace("también", "tambien")
                    if variant == 6:
                        raw = raw.replace("estábamos", "estabamos").replace("habíamos", "habiamos")
                    rid = hashlib.sha256(
                        f"{VERSION}|{seed}|{group}|{variant}".encode()
                    ).hexdigest()[:24]
                    text = raw.strip()
                    rows.append({
                        "review_id": "chal_" + rid, "text": text,
                        "intended_label": label, "domain": domain,
                        "nucleus_id": group, "variation_id": str(variant),
                        "phenomenon": ("contrast" if variant == 2 else
                                      "negation" if variant == 3 else
                                      "regional" if variant in (1, 4) else
                                      "misspelling" if variant in (5, 6) else
                                      "plain"),
                        "region": region, "synthetic": "YES",
                        "label_origin": "AUTHORED_INTENT_SYNTHETIC_NOT_HUMAN_GOLD",
                        "generator": SOURCE, "version": VERSION,
                        "seed": str(seed), "date": generated_date,
                        "text_sha256": text_sha(text),
                    })
    return rows


def audit_challenge(rows: list[dict], original_texts: list[str] | None = None) -> dict:
    if len(rows) != 1260:
        raise CorpusError("El desafío v1 espera 1260 reseñas sintéticas.")
    if any(set(r) != set(FIELDS) for r in rows):
        raise CorpusError("Esquema de procedencia inválido.")
    ids = set()
    hashes = set()
    groups = set()
    label_counts = Counter()
    for row in rows:
        if row["synthetic"] != "YES" or row["label_origin"] != "AUTHORED_INTENT_SYNTHETIC_NOT_HUMAN_GOLD":
            raise CorpusError("No declarar etiquetas humanas independientes.")
        if row["review_id"] in ids or row["text_sha256"] in hashes:
            raise CorpusError("Duplicados exactos en el desafío.")
        if row["text_sha256"] != text_sha(row["text"]):
            raise CorpusError("Texto alterado respecto de su hash.")
        if privacy_flags(row["text"]):
            raise CorpusError("Posible dato personal en ejemplo.")
        if not 55 <= len(row["text"]) <= 1000 or not row["text"].strip().endswith("."):
            raise CorpusError("Longitud o puntuación inválida.")
        ids.add(row["review_id"]); hashes.add(row["text_sha256"])
        groups.add(row["nucleus_id"]); label_counts[row["intended_label"]] += 1
    if set(label_counts) != set(LABELS) or len(groups) != 180:
        raise CorpusError("Faltan grupos o clases previstos.")
    overlap = []
    if original_texts is not None:
        old_hashes = {text_sha(t) for t in original_texts}
        overlap = sorted(old_hashes & hashes)
        if overlap:
            raise CorpusError("Contaminación por duplicados exactos del corpus inicial.")
    return {
        "type": "SYNTHETIC_CHALLENGE_NOT_REAL",
        "rows": len(rows), "independent_nuclei": len(groups),
        "labels": dict(sorted(label_counts.items())),
        "domains": dict(sorted(Counter(r["domain"] for r in rows).items())),
        "phenomena": dict(sorted(Counter(r["phenomenon"] for r in rows).items())),
        "exact_overlap_count": len(overlap),
        "limits": [
            "Independent authored situations, but frames repeat within a nucleus.",
            "Intended polarity not verified by independent human annotators.",
            "Limited irony/complex mixed polarity; no gold linguistic annotation.",
            "No inference about performance on authentic tourism reviews.",
        ],
    }


def save_challenge(rows: list[dict], audit: dict, dest: Path, *, seed: int) -> dict:
    dest = Path(dest)
    if "processed" not in dest.parts and "private" not in dest.parts:
        raise CorpusError("Guardar solo en data/processed o data/private.")
    if dest.exists() and any(dest.iterdir()):
        raise CorpusError("No sobrescribir desafío cerrado.")
    dest.mkdir(parents=True, exist_ok=True)
    datafile = dest / "challenge_synthetic.csv"
    with datafile.open("w", encoding="utf-8", newline="") as file:
        w = csv.DictWriter(file, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    manifest = {
        "type": "SYNTHETIC_CHALLENGE_NOT_REAL",
        "generation": {"method": SOURCE, "version": VERSION, "seed": seed,
                       "date": rows[0]["date"]},
        "data_sha256": file_sha(datafile), "audit": audit,
    }
    (dest / "challenge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description="Independent authored SYNTHETIC challenge, NOT human gold")
    p.add_argument("--out", type=Path, default=Path("data/processed/out-of-generator-challenge-v1"))
    p.add_argument("--seed", type=int, default=913)
    args = p.parse_args(argv)
    try:
        rows = generate_challenge(seed=args.seed)
        report = audit_challenge(rows)
        manifest = save_challenge(rows, report, args.out, seed=args.seed)
    except (CorpusError, OSError, ValueError) as exc:
        p.exit(2, "Challenge generation blocked: " + str(exc) + "\n")
    print(json.dumps({
        "type": report["type"], "rows": report["rows"],
        "nuclei": report["independent_nuclei"],
        "sha256": manifest["data_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
