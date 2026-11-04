"""Etiquetas semánticas; NO se asume el mapeo numérico de Kaggle."""
LABELS = ("negativo", "neutral", "positivo")
ALIASES = {
    "negative": "negativo", "negativo": "negativo", "negativa": "negativo",
    "neutral": "neutral", "neutro": "neutral", "neutra": "neutral",
    "positive": "positivo", "positivo": "positivo", "positiva": "positivo",
}

def normalize_label(value: str) -> str:
    key = str(value).strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Etiqueta no admitida: {value!r}. Utilice positivo/neutral/negativo; no números sin verificar el CSV.")
    return ALIASES[key]
