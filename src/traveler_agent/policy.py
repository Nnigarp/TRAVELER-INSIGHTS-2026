"""Política transparente: sugerencias locales sustentadas; sin acciones externas."""
from __future__ import annotations

ACTIONS = {
    "seguridad": "Revisar el incidente de seguridad descrito y escalarlo a un responsable humano.",
    "higiene": "Solicitar verificación humana del estado de limpieza e higiene relatado.",
    "precio": "Contrastar el cobro relatado con la tarifa anunciada antes de proponer medidas.",
    "espera": "Revisar los tiempos de espera y la información ofrecida al visitante.",
    "acceso": "Verificar las condiciones de acceso, traslado o señalización descritas.",
    "servicio": "Revisar el proceso de atención y las observaciones del visitante.",
    "ruido": "Comprobar las condiciones acústicas reportadas sin atribuir aún su origen.",
    "alimentacion": "Revisar los comentarios sobre alimentos o servicio de restauración.",
}
PRIORITY = ("seguridad", "higiene", "precio", "espera", "acceso", "servicio", "ruido", "alimentacion")


def _incident_line(incidents: list) -> str:
    open_ = [i for i in incidents if i.get("status") == "no_resuelto"]
    resolved = [i for i in incidents if i.get("status") == "resuelto"]
    bits = []
    if open_:
        sample = open_[0].get("clause") or "incidente abierto"
        bits.append(f"Incidente abierto: «{sample[:160]}».")
    if resolved:
        sample = resolved[0].get("clause") or "incidente resuelto"
        bits.append(f"Incidente resuelto en el relato: «{sample[:120]}».")
    return " ".join(bits)


def recommend(label: str, aspects: dict, needs_review: bool, *, context: dict | None = None) -> tuple[str, str]:
    context = context or {}
    incidents = list(context.get("incidents") or [])
    assessment = dict(context.get("aspect_assessment") or {})
    ambiguity = dict(context.get("ambiguity") or {})
    incident_note = _incident_line(incidents)

    if needs_review:
        detail = incident_note or "Confianza limitada, contraste, ambigüedad o verificación contextual."
        return (
            "revisión_humana",
            "Revisar manualmente antes de actuar. " + detail
            + " No emitir una acción automática vinculante.",
        )

    if ambiguity.get("ambiguous") or ambiguity.get("doubt"):
        return (
            "revisión_humana",
            "La reseña expresa duda o valoración no concluyente. Solicitar aclaración humana; "
            "no cerrar un incidente ni una recomendación operativa.",
        )

    open_incidents = [i for i in incidents if i.get("status") == "no_resuelto"]
    resolved_only = incidents and all(i.get("status") == "resuelto" for i in incidents) and not open_incidents

    if label == "negativo":
        unfavorable = [a for a in PRIORITY if assessment.get(a, {}).get("polarity") == "desfavorable"
                       or a in (aspects or {})]
        # Prefer aspects tied to open incidents.
        for inc in open_incidents:
            for a in inc.get("aspects") or []:
                if a in PRIORITY and a not in unfavorable:
                    unfavorable.insert(0, a)
        detected = [a for a in PRIORITY if a in unfavorable or a in (aspects or {})]
        detected = list(dict.fromkeys(detected))
        if len(detected) == 1:
            msg = ACTIONS[detected[0]]
            if incident_note:
                msg = f"{msg} {incident_note}"
            return "sugerencia", msg
        if detected:
            items = "; ".join(f"{a}: {ACTIONS[a]}" for a in detected)
            return "sugerencia", (
                "Verificar por separado los aspectos mencionados (polaridad por aspecto "
                f"no determinada): {items}"
                + (f" {incident_note}" if incident_note else "")
            )
        if incident_note:
            return "sugerencia", f"Revisar la reseña negativa con el responsable. {incident_note}"
        return "sugerencia", "Revisar la reseña negativa y determinar una acción con el responsable."

    if label == "neutral":
        if open_incidents:
            return "sugerencia", (
                "Aunque la etiqueta global es neutral, hay indicios de incidente abierto. "
                f"{incident_note} Derivar a revisión humana antes de archivar."
            )
        return "informativo", "Registrar la reseña descriptiva y observar tendencias antes de actuar."

    # positivo
    if open_incidents:
        return "sugerencia", (
            "La etiqueta global es positiva, pero el relato incluye un incidente no resuelto. "
            f"{incident_note} Contrastar con una persona antes de cerrar el caso."
        )
    if resolved_only:
        fav = [a for a in PRIORITY if assessment.get(a, {}).get("polarity") == "favorable"]
        note = f" Aspectos favorables: {', '.join(fav)}." if fav else ""
        return "informativo", (
            "Registrar la experiencia favorable tras un incidente que el relato marca como resuelto."
            + note + (f" {incident_note}" if incident_note else "")
        )
    fav = [a for a in PRIORITY if assessment.get(a, {}).get("polarity") == "favorable"]
    if fav:
        return "informativo", (
            "Registrar la experiencia favorable; no se requiere intervención automática. "
            f"Aspectos favorables señalados: {', '.join(fav)}."
        )
    return "informativo", "Registrar la experiencia favorable; no se requiere intervención automática."
