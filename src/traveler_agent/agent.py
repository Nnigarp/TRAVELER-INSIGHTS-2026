"""Integración de percepción, clasificación, evidencia descriptiva y reglas."""
import re
from .aspects import find_aspects,evidence_spans
from .policy import recommend

def inspect_review(text: str, model, *, review_id: str = "", threshold: float = 0.56) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("La reseña no puede estar vacía.")
    if len(text) > 5000:
        raise ValueError("La reseña excede el límite de 5000 caracteres.")
    content = text.strip()
    probs = model.predict_proba([content])[0]
    classes = list(model.classes_)
    i = int(probs.argmax())
    label = classes[i]
    confidence = float(probs[i])
    aspects = find_aspects(content)
    mixed = bool(re.search(r"\b(pero|aunque|sin embargo)\b",content,re.I))
    # Puntajes sin calibrar: umbral demostrativo, no fiabilidad probabilística validada.
    needs_review = confidence < threshold or (mixed and confidence < 0.80)
    state, action = recommend(label, aspects, needs_review)
    return {
        "id": str(review_id), "sentimiento": label,
        "puntuaciones_modelo_no_calibradas": {c:round(float(p),4) for c,p in zip(classes, probs)},
        "confianza_no_calibrada": round(confidence,4),
        "aspectos_exploratorios": list(aspects),
        "fragmentos_literales_no_explicativos": evidence_spans(content,aspects),
        "revision_humana": bool(needs_review), "tipo_salida": state,
        "recomendacion": action,
        "advertencia": "Prototipo académico: las puntuaciones no están calibradas; los aspectos se extraen mediante reglas. No inferir causalidad de los fragmentos.",
    }
