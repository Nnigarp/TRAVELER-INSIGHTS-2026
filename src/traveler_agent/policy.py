"""Política transparente: sugerencias, no decisiones automatizadas vinculantes."""
ACTIONS = {
 "seguridad": "Revisar el incidente de seguridad y escalarlo al responsable humano.",
 "higiene": "Solicitar verificación de limpieza e higiene del servicio.",
 "precio": "Revisar claridad y comunicación de tarifas y cobros.",
 "espera": "Revisar tiempos de espera y capacidad de atención.",
 "acceso": "Revisar las condiciones de acceso y accesibilidad.",
 "servicio": "Revisar el proceso de atención y las observaciones del visitante.",
 "ruido": "Verificar condiciones acústicas y reportes de ruido.",
 "alimentacion": "Revisar los comentarios sobre alimentos y desayunos.",
}
PRIORITY = ("seguridad", "higiene", "precio", "espera", "acceso", "servicio", "ruido", "alimentacion")

def recommend(label: str, aspects: dict, needs_review: bool) -> tuple[str, str]:
    if needs_review:
        return "revisión_humana", "Revisar manualmente: confianza insuficiente o texto ambiguo. No emitir una acción automática."
    if label == "negativo":
        detected = [aspect for aspect in PRIORITY if aspect in aspects]
        if len(detected) == 1:
            return "sugerencia", ACTIONS[detected[0]]
        if detected:
            # La etiqueta global negativa no implica polaridad negativa en cada aspecto.
            # No se prioriza una sola dimensión ni se afirma una queja específica sin ABSA.
            items = "; ".join(f"{a}: {ACTIONS[a]}" for a in detected)
            return "sugerencia", (
                "Verificar por separado los aspectos mencionados (polaridad por aspecto "
                f"no determinada): {items}"
            )
        return "sugerencia", "Revisar la reseña negativa y determinar una acción con el responsable."
    if label == "neutral":
        return "informativo", "Registrar la reseña y observar tendencias antes de actuar."
    return "informativo", "Registrar la experiencia favorable; no se requiere intervención automática."
