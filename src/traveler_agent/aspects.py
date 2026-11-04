"""Extracción de aspectos por diccionario: EXPLORATORIA; no es ABSA entrenado."""
import re
ASPECTS = {
  "servicio": ["personal", "atencion", "atención", "mesero", "guia", "guía", "recepcion", "recepción", "amable", "amables", "trato", "empleado", "empleados", "empleada", "empleadas", "servicio"],
  "precio": ["precio", "tarifa", "costoso", "caro", "barato", "costo", "cobro", "pagar"],
  "higiene": ["limpio", "limpia", "sucio", "sucia", "higiene", "basura", "aseo", "limpieza"],
  "acceso": ["acceso", "accesibilidad", "rampa", "transporte", "movilidad", "entrada"],
  "espera": ["espera", "esperamos", "demora", "tardaron", "tardó", "fila", "retraso"],
  "seguridad": ["seguro", "inseguro", "robo", "peligro", "seguridad"],
  "ruido": ["ruido", "ruidos", "ruidoso", "ruidosa", "silencio"],
  "alimentacion": ["desayuno", "comida", "restaurante", "almuerzo", "cena", "alimento"],
}

def find_aspects(text: str) -> dict[str, list[str]]:
    found = {}
    for name, terms in ASPECTS.items():
        matched = []
        for term in terms:
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, flags=re.I):
                matched.append(term)
        if matched:
            found[name] = matched
    return found

def evidence_spans(text: str, aspects: dict[str, list[str]], max_spans: int = 3) -> list[str]:
    """Frases literales que mencionan términos: NO equivalen a explicación causal."""
    segments = [p.strip() for p in re.split(r"(?<=[.!?;])\s+|\s*,\s*(?=(?:pero|aunque)\b)", text) if p.strip()]
    result=[]
    for seg in segments:
        if any(re.search(r"(?<!\w)"+re.escape(term)+r"(?!\w)", seg, re.I)
               for terms in aspects.values() for term in terms):
            result.append(seg[:260])
        if len(result) >= max_spans:
            break
    return result
