"""Private, provenance-aware corpus infrastructure; never an automatic labeler.

Raw reviews and annotations remain local. This module never scrapes websites,
infers legal reuse permission, maps labels numerically or reports model accuracy.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .labels import LABELS

SCHEMA_VERSION = "tourism-corpus-v1"
ASPECTS = frozenset({"servicio", "higiene", "espera", "precio", "seguridad",
                     "acceso", "alimentacion", "instalaciones"})
POLARITIES = frozenset((*LABELS, "desconocido"))
REVIEW_COLUMNS = ("review_id", "text", "source_id", "group_id", "language")
ANNOTATION_COLUMNS = ("review_id", "annotator_id", "sentiment", "aspects_json")
ADJUDICATION_COLUMNS = ("review_id", "sentiment", "aspects_json", "adjudicator_id", "rationale")
SOURCE_COLUMNS = ("source_id", "source_type", "source_reference", "reuse_basis",
                  "permission_reference", "authorization_reviewed")


class CorpusError(ValueError):
    pass


def normalized_key(text: str) -> str:
    # Derived duplicate key only; training text remains unchanged.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_sha(text: str) -> str:
    return hashlib.sha256(normalized_key(text).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path, expected: tuple[str, ...]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise CorpusError("CSV inválido: encabezados ausentes o repetidos.")
        if set(reader.fieldnames) != set(expected):
            raise CorpusError(f"Esquema incorrecto. Columnas exigidas: {expected}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise CorpusError("CSV inválido: número de columnas inconsistente.")
    return rows


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace protects against interrupted writes.
    import os
    import tempfile
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".corpus-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _require_identifier(value: str, field: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}", value):
        raise CorpusError(f"{field} debe ser un identificador seudónimo válido (máx. 80).")
    return value


def validate_sources(path: Path) -> tuple[list[dict], dict]:
    rows = _read_csv(path, SOURCE_COLUMNS)
    if not rows:
        raise CorpusError("No hay fuentes registradas.")
    ids = set()
    for row in rows:
        sid = _require_identifier(row["source_id"], "source_id")
        if sid in ids:
            raise CorpusError("source_id duplicado.")
        ids.add(sid)
        if row["source_type"] not in {"voluntary", "open_corpus", "institutional"}:
            raise CorpusError("source_type inválido.")
        if row["reuse_basis"] not in {"consent", "license", "written_permission"}:
            raise CorpusError("reuse_basis inválido.")
        if not row["source_reference"] or not row["permission_reference"]:
            raise CorpusError("Procedencia y prueba/documentación de permiso obligatorias.")
        if row["authorization_reviewed"] != "yes":
            raise CorpusError("La autorización debe revisarse manualmente antes de admitir la fuente.")
    return rows, {"sources": len(rows), "sha256": file_sha(path)}


def validate_reviews(path: Path, sources_path: Path) -> tuple[list[dict], dict]:
    sources, _ = validate_sources(sources_path)
    approved = {row["source_id"] for row in sources}
    rows = _read_csv(path, REVIEW_COLUMNS)
    if not rows:
        raise CorpusError("No hay reseñas.")
    ids, keys, duplicates = set(), {}, 0
    language_counts = Counter()
    for row in rows:
        rid = _require_identifier(row["review_id"], "review_id")
        if rid in ids:
            raise CorpusError("review_id duplicado.")
        ids.add(rid)
        if row["source_id"] not in approved:
            raise CorpusError("Reseña sin fuente autorizada registrada.")
        _require_identifier(row["group_id"], "group_id")
        text = row["text"]
        if not 3 <= len(text.strip()) <= 5000:
            raise CorpusError("Texto fuera de rango (3–5000).")
        if row["language"] not in {"es", "en", "it", "other", "unknown"}:
            raise CorpusError("Idioma no admitido; utilizar unknown si no está verificado.")
        language_counts[row["language"]] += 1
        key = text_sha(text)
        if key in keys:
            duplicates += 1
        else:
            keys[key] = rid
    return rows, {
        "schema": SCHEMA_VERSION, "rows": len(rows),
        "normalized_exact_duplicate_extra_rows": duplicates,
        "language_counts": dict(sorted(language_counts.items())),
        "sha256": file_sha(path),
        "note": "La revisión de PII y licencias es manual; la validación no la certifica.",
    }


def parse_aspects(raw: str) -> dict[str, str]:
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise CorpusError("aspects_json debe ser JSON válido.") from exc
    if not isinstance(obj, dict):
        raise CorpusError("aspects_json debe ser objeto JSON.")
    if any(key not in ASPECTS or val not in POLARITIES for key, val in obj.items()):
        raise CorpusError("Aspecto o polaridad no admitidos.")
    return obj


def validate_annotations(path: Path, valid_review_ids: set[str]) -> tuple[list[dict], dict]:
    rows = _read_csv(path, ANNOTATION_COLUMNS)
    keys, counts = set(), Counter()
    per_review = Counter()
    for row in rows:
        if row["review_id"] not in valid_review_ids:
            raise CorpusError("Anotación de reseña inexistente.")
        _require_identifier(row["annotator_id"], "annotator_id")
        key = (row["review_id"], row["annotator_id"])
        if key in keys:
            raise CorpusError("Anotación duplicada por anotador.")
        keys.add(key)
        if row["sentiment"] not in POLARITIES:
            raise CorpusError("Sentimiento inválido.")
        parse_aspects(row["aspects_json"])
        counts[row["sentiment"]] += 1
        per_review[row["review_id"]] += 1
    return rows, {
        "annotations": len(rows), "annotated_reviews": len(per_review),
        "reviews_with_two_or_more_annotations": sum(n >= 2 for n in per_review.values()),
        "label_counts": dict(sorted(counts.items())),
        "sha256": file_sha(path),
    }


def record_annotation(path: Path, review_ids: set[str], *, review_id: str,
                      annotator_id: str, sentiment: str, aspects: dict) -> None:
    if review_id not in review_ids:
        raise CorpusError("review_id inexistente.")
    _require_identifier(annotator_id, "annotator_id")
    if sentiment not in POLARITIES:
        raise CorpusError("Sentimiento no admitido.")
    aspects_str = json.dumps(aspects, ensure_ascii=False, sort_keys=True)
    parse_aspects(aspects_str)
    rows = _read_csv(path, ANNOTATION_COLUMNS) if path.exists() else []
    if any(r["review_id"] == review_id and r["annotator_id"] == annotator_id for r in rows):
        raise CorpusError("Ya existe anotación de esa reseña por ese anotador. No sobrescribir silenciosamente.")
    rows.append({"review_id": review_id, "annotator_id": annotator_id,
                 "sentiment": sentiment, "aspects_json": aspects_str})
    _write_csv(path, ANNOTATION_COLUMNS, rows)


def validate_adjudications(path: Path, review_ids: set[str],
                           annotations: list[dict]) -> tuple[list[dict], dict]:
    rows = _read_csv(path, ADJUDICATION_COLUMNS)
    annotated = {r["review_id"] for r in annotations}
    seen, counts = set(), Counter()
    for row in rows:
        rid = row["review_id"]
        if rid not in review_ids or rid not in annotated:
            raise CorpusError("Adjudicación requiere reseña y anotación previa.")
        if rid in seen:
            raise CorpusError("Adjudicación duplicada.")
        seen.add(rid)
        if row["sentiment"] not in POLARITIES:
            raise CorpusError("Sentimiento adjudicado inválido.")
        _require_identifier(row["adjudicator_id"], "adjudicator_id")
        if not row["rationale"].strip() or len(row["rationale"]) > 500:
            raise CorpusError("Justificación de adjudicación obligatoria (máx. 500).")
        parse_aspects(row["aspects_json"])
        counts[row["sentiment"]] += 1
    return rows, {"adjudications": len(rows), "label_counts": dict(sorted(counts.items())),
                  "sha256": file_sha(path)}


def record_adjudication(path: Path, review_id: str, sentiment: str, aspects: dict,
                        adjudicator_id: str, rationale: str,
                        reviews: list[dict], annotations: list[dict]) -> None:
    if sentiment not in POLARITIES:
        raise CorpusError("Sentimiento inválido.")
    _require_identifier(adjudicator_id, "adjudicator_id")
    if not rationale.strip() or len(rationale) > 500:
        raise CorpusError("Justificación obligatoria (1–500 caracteres).")
    if review_id not in {r["review_id"] for r in reviews}:
        raise CorpusError("Reseña inexistente.")
    if review_id not in {r["review_id"] for r in annotations}:
        raise CorpusError("Debe existir anotación previa.")
    aspects_raw = json.dumps(aspects, ensure_ascii=False, sort_keys=True)
    parse_aspects(aspects_raw)
    rows = _read_csv(path, ADJUDICATION_COLUMNS) if path.exists() else []
    if any(r["review_id"] == review_id for r in rows):
        raise CorpusError("Adjudicación previa existente. No sobrescribir.")
    rows.append({"review_id": review_id, "sentiment": sentiment, "aspects_json": aspects_raw,
                 "adjudicator_id": adjudicator_id, "rationale": rationale.strip()})
    _write_csv(path, ADJUDICATION_COLUMNS, rows)
