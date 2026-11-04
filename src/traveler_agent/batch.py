"""Deterministic demo batch analysis and explicit human corrections.

Not a scientific benchmark: no ground-truth metrics are calculated here.
"""
from __future__ import annotations
import csv
import io
import json
from collections import Counter
from typing import Any

VALID_LABELS = frozenset({"positivo", "neutral", "negativo"})
MAX_ROWS = 200
MAX_TEXT_LENGTH = 5000
MAX_FILE_BYTES = 1024 * 1024


def parse_review_lines(raw: str, *, max_rows: int = MAX_ROWS) -> list[str]:
    if not isinstance(raw, str):
        raise ValueError("La entrada debe ser texto.")
    reviews = [line.strip() for line in raw.splitlines() if line.strip()]
    if not reviews:
        raise ValueError("Introduce al menos una reseña, una por línea.")
    if len(reviews) > max_rows:
        raise ValueError(f"Límite de {max_rows} reseñas por lote.")
    if any(len(text) > MAX_TEXT_LENGTH for text in reviews):
        raise ValueError(f"Cada reseña debe tener como máximo {MAX_TEXT_LENGTH} caracteres.")
    return reviews


def parse_review_csv(content: bytes, *, text_col: str = "reseña") -> list[str]:
    if not isinstance(content, bytes) or len(content) > MAX_FILE_BYTES:
        raise ValueError("El CSV debe ser un archivo de hasta 1 MiB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("El CSV debe estar codificado en UTF-8.") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or text_col not in reader.fieldnames:
        raise ValueError(f"Falta la columna obligatoria '{text_col}'.")
    rows = []
    for row in reader:
        value = (row.get(text_col) or "").strip()
        if value:
            rows.append(value)
        if len(rows) > MAX_ROWS:
            raise ValueError(f"Límite de {MAX_ROWS} reseñas por lote.")
    if not rows:
        raise ValueError("El CSV no contiene reseñas válidas.")
    if any(len(text) > MAX_TEXT_LENGTH for text in rows):
        raise ValueError("Una reseña excede 5000 caracteres.")
    return rows


def analyze_batch(reviews: list[str], model: Any, *, prefix: str = "demo") -> list[dict]:
    if not reviews or len(reviews) > MAX_ROWS:
        raise ValueError("Número de reseñas no permitido.")
    from .agent import inspect_review
    result = []
    for index, text in enumerate(reviews, 1):
        record = inspect_review(text, model, review_id=f"{prefix}-{index:03d}")
        record["texto_original"] = text
        record["etiqueta_corregida_humano"] = None
        record["comentario_revision_humana"] = None
        record["estado_revision"] = "pendiente" if record["revision_humana"] else "sin_revisar"
        result.append(record)
    return result


def apply_human_correction(records: list[dict], review_id: str, label: str, note: str = "") -> list[dict]:
    if label not in VALID_LABELS:
        raise ValueError("Etiqueta de corrección inválida.")
    if not isinstance(note, str) or len(note) > 500:
        raise ValueError("Comentario limitado a 500 caracteres.")
    if not isinstance(records, list):
        raise ValueError("No hay resultados para revisar.")
    matches = [i for i, row in enumerate(records) if row.get("id") == review_id]
    if len(matches) != 1:
        raise ValueError("Identificador no encontrado o duplicado.")
    edited = [dict(row) for row in records]
    edited[matches[0]]["etiqueta_corregida_humano"] = label
    edited[matches[0]]["comentario_revision_humana"] = note.strip()
    edited[matches[0]]["estado_revision"] = "corregida"
    edited[matches[0]]["tipo_salida"] = "revisión_humana"
    edited[matches[0]]["recomendacion"] = (
        "Corrección registrada. Revisar acciones con un responsable; "
        "no automatizar a partir de la etiqueta corregida."
    )
    return edited


def demo_trends(records: list[dict]) -> list[dict]:
    """Descriptive counts of demo predictions, never accuracy or scientific metrics."""
    counts = Counter(row["sentimiento"] for row in records)
    return [{"sentimiento_modelo": key, "cantidad": counts[key]}
            for key in ("positivo", "neutral", "negativo")]


def export_json(records: list[dict]) -> str:
    return json.dumps({"tipo": "DEMO_NO_VALIDADO", "n": len(records), "registros": records},
                      ensure_ascii=False, indent=2)


def export_csv(records: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    fields = ["id", "texto_original", "sentimiento", "confianza_no_calibrada",
              "revision_humana", "etiqueta_corregida_humano", "estado_revision", "recomendacion"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()
