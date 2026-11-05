"""Human annotation quality tools (private by default).

Span positions are Python Unicode string offsets into the ORIGINAL review.
Near-duplicate candidates are for human review only; never auto-delete records.
"""
from __future__ import annotations

import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .corpus import (
    ASPECTS, POLARITIES, CorpusError, _read_csv, _write_csv, file_sha,
    normalized_key, validate_annotations, validate_reviews
)

SPAN_COLUMNS = ("review_id", "annotator_id", "aspect", "polarity", "start", "end", "text")


def validate_span_entries(review_text: str, spans: list[dict]) -> list[dict]:
    """Literal, exact [start,end) spans. No automatic ABSA is implied."""
    if not isinstance(spans, list):
        raise CorpusError("Las evidencias deben ser una lista JSON.")
    result = []
    for span in spans:
        if not isinstance(span, dict) or set(span) != {"aspect", "polarity", "start", "end"}:
            raise CorpusError("Cada evidencia requiere aspect, polarity, start y end.")
        aspect, polarity = span["aspect"], span["polarity"]
        start, end = span["start"], span["end"]
        if aspect not in ASPECTS or polarity not in POLARITIES:
            raise CorpusError("Aspecto o polaridad de evidencia inválidos.")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(review_text):
            raise CorpusError("Offsets fuera de rango o vacíos.")
        selected = review_text[start:end]
        if not selected.strip():
            raise CorpusError("Evidencia vacía.")
        result.append({
            "aspect": aspect, "polarity": polarity,
            "start": str(start), "end": str(end), "text": selected,
        })
    return result


def record_evidence(path: Path, review_id: str, annotator_id: str,
                    review_text: str, spans: list[dict], annotated_aspects: dict) -> None:
    checked = validate_span_entries(review_text, spans)
    if any(annotated_aspects.get(s["aspect"]) != s["polarity"] for s in checked):
        raise CorpusError("El span no coincide con la polaridad anotada.")
    rows = _read_csv(path, SPAN_COLUMNS) if path.exists() else []
    if any(r["review_id"] == review_id and r["annotator_id"] == annotator_id for r in rows):
        raise CorpusError("Evidencias ya registradas para ese anotador y reseña.")
    rows.extend({"review_id": review_id, "annotator_id": annotator_id, **s} for s in checked)
    _write_csv(path, SPAN_COLUMNS, rows)


def validate_evidence(path: Path, reviews: list[dict], annotations: list[dict]) -> dict:
    if not path.exists():
        return {"evidence_spans": 0, "status": "not_created"}
    by_id = {r["review_id"]: r["text"] for r in reviews}
    by_annotation = {
        (r["review_id"], r["annotator_id"]): json.loads(r["aspects_json"])
        for r in annotations
    }
    rows = _read_csv(path, SPAN_COLUMNS)
    for row in rows:
        key = (row["review_id"], row["annotator_id"])
        if key not in by_annotation or row["review_id"] not in by_id:
            raise CorpusError("Evidencia sin anotación o reseña.")
        try:
            start, end = int(row["start"]), int(row["end"])
        except ValueError as exc:
            raise CorpusError("Offsets inválidos.") from exc
        text = by_id[row["review_id"]]
        if (start < 0 or start >= end or end > len(text)
                or text[start:end] != row["text"]):
            raise CorpusError("Evidencia no coincide con el texto original.")
        if by_annotation[key].get(row["aspect"]) != row["polarity"]:
            raise CorpusError("La evidencia contradice la anotación.")
    return {"evidence_spans": len(rows), "sha256": file_sha(path)}


def annotator_agreement(annotations: list[dict]) -> dict:
    """Pairwise kappa on shared reviews; descriptive, not truth/reliability proof."""
    by_person = defaultdict(dict)
    for row in annotations:
        by_person[row["annotator_id"]][row["review_id"]] = row
    people = sorted(by_person)
    comparisons = []
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            shared = sorted(set(by_person[a]) & set(by_person[b]))
            if len(shared) < 2:
                continue
            a_labels = [by_person[a][rid]["sentiment"] for rid in shared]
            b_labels = [by_person[b][rid]["sentiment"] for rid in shared]
            def safe_kappa(x, y):
                import math
                score = float(cohen_kappa_score(x, y))
                return score if math.isfinite(score) else None
            aspects = {}
            for aspect in sorted(ASPECTS):
                left = [json.loads(by_person[a][rid]["aspects_json"]).get(aspect, "ausente")
                        for rid in shared]
                right = [json.loads(by_person[b][rid]["aspects_json"]).get(aspect, "ausente")
                         for rid in shared]
                if any(l != "ausente" or r != "ausente" for l, r in zip(left, right)):
                    aspects[aspect] = safe_kappa(left, right)
            comparisons.append({
                "annotators": [a, b], "shared_reviews": len(shared),
                "global_kappa": safe_kappa(a_labels, b_labels),
                "aspect_kappa": aspects,
            })
    return {"method": "pairwise_cohen_kappa", "comparisons": comparisons,
            "note": "Kappa null cuando es indefinido; requiere varias anotaciones humanas."}


def near_duplicate_candidates(reviews: list[dict], *,
                              threshold: float = 0.88, max_rows: int = 2000) -> dict:
    if len(reviews) > max_rows:
        raise CorpusError("Auditoría de casi duplicados limitada a 2000 reseñas por ejecución.")
    if not .75 <= threshold <= 1:
        raise CorpusError("Umbral fuera de rango.")
    normalized = [(r["review_id"], normalized_key(r["text"])) for r in reviews]
    candidates = []
    for i, (rid, text) in enumerate(normalized):
        for other_id, other in normalized[i+1:]:
            if not text or not other:
                continue
            length_ratio = min(len(text), len(other)) / max(len(text), len(other))
            if length_ratio < threshold * 0.85:
                continue
            similarity = SequenceMatcher(None, text, other, autojunk=False).ratio()
            if similarity >= threshold:
                candidates.append({
                    "review_ids": sorted([rid, other_id]),
                    "similarity_indicative": round(similarity, 4),
                    "decision": "revision_humana_pendiente",
                })
    return {
        "method": "difflib_sequence_matcher_pairwise_v1",
        "review_count": len(reviews),
        "threshold": threshold,
        "candidates": sorted(candidates, key=lambda r: (-r["similarity_indicative"], r["review_ids"])),
        "limitations": "No detecta todas las paráfrasis; no elimina ninguna reseña automáticamente.",
    }
