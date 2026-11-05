"""Local-only automation for permitted reviews; no scraping, no automatic gold labels.

Source authorization remains a human decision. All outputs stay in gitignored
private/processed directories. Suspicious PII is excluded, not silently edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from .corpus import (
    CorpusError, REVIEW_COLUMNS, _read_csv, _write_csv, text_sha, validate_sources,
    validate_reviews, _require_identifier,
)
from .corpus_review import near_duplicate_candidates

_EMAIL = re.compile(r"(?<!\w)[^@\s]{1,80}@[^@\s.]{1,80}\.[A-Za-z]{2,20}(?!\w)")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_PRELABEL_COLUMNS = (
    "review_id", "suggested_sentiment", "max_score_demo_uncalibrated",
    "needs_human_review", "model_provenance", "context_flags"
)


def privacy_flags(text: str) -> list[str]:
    """Conservative warning rules, not PII anonymization or legal assurance."""
    flags = []
    if _EMAIL.search(text):
        flags.append("possible_email")
    if _PHONE.search(text):
        flags.append("possible_phone")
    if _URL.search(text):
        flags.append("possible_url")
    return flags


def ingest_authorized_csv(
    file: Path, root: Path, source_id: str, *,
    text_column: str, group_column: str,
    id_column: str | None = None, language: str = "unknown",
    max_rows: int = 10000,
) -> dict:
    """Conservatively import texts without guessing grouping or human labels."""
    root = Path(root)
    if "private" not in root.parts:
        raise CorpusError("Importación sólo bajo data/private u otra carpeta privada.")
    sources, _ = validate_sources(root / "sources.csv")
    source = next((s for s in sources if s["source_id"] == source_id), None)
    if source is None:
        raise CorpusError("Fuente no registrada con autorización revisada.")
    if language not in {"es", "en", "it", "other", "unknown"}:
        raise CorpusError("Idioma inválido.")
    path = Path(file)
    if not path.is_file() or path.stat().st_size > 40 * 1024 * 1024:
        raise CorpusError("CSV ausente o demasiado grande (40 MiB).")
    existing = _read_csv(root / "reviews.csv", REVIEW_COLUMNS)
    used_ids = {r["review_id"] for r in existing}
    known_hashes = {text_sha(r["text"]) for r in existing}
    accepted, flagged, duplicates, invalid = [], 0, 0, 0
    # Read incrementally with explicit column mapping. Never auto-infer labels.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise CorpusError("Encabezados CSV inválidos.")
        needed = [text_column, group_column] + ([id_column] if id_column else [])
        if any(col not in reader.fieldnames for col in needed):
            raise CorpusError("Faltan columnas de texto, grupo o identificador declaradas.")
        for index, row in enumerate(reader):
            if index >= max_rows:
                raise CorpusError("Límite de registros excedido: importar por lotes más pequeños.")
            if None in row or any(v is None for v in row.values()):
                invalid += 1
                continue
            text = row[text_column].strip()
            group = row[group_column].strip()
            if not 3 <= len(text) <= 5000:
                invalid += 1
                continue
            try:
                _require_identifier(group, "group_id")
            except CorpusError:
                invalid += 1
                continue
            if privacy_flags(text):
                flagged += 1
                continue  # Never write suspected personal information to outputs.
            fingerprint = text_sha(text)
            if fingerprint in known_hashes:
                duplicates += 1
                continue
            upstream_id = row[id_column].strip() if id_column else str(index)
            # Hash avoids copying upstream IDs or personal identifiers.
            rid = "imp_" + hashlib.sha256(
                f"{source_id}:{upstream_id}".encode("utf-8")
            ).hexdigest()[:24]
            if rid in used_ids:
                duplicates += 1
                continue
            accepted.append({
                "review_id": rid, "text": text, "source_id": source_id,
                "group_id": group, "language": language,
            })
            used_ids.add(rid)
            known_hashes.add(fingerprint)
    if accepted:
        _write_csv(root / "reviews.csv", REVIEW_COLUMNS, existing + accepted)
    return {
        "status": "local_import_only",
        "source_id": source_id, "imported": len(accepted),
        "skipped_possible_pii": flagged,
        "skipped_exact_duplicates_or_ids": duplicates,
        "skipped_invalid": invalid,
        "warning": (
            "Human verification of source license and remaining PII is mandatory. "
            "Names, addresses and indirect identifiers are NOT fully detected. "
            "No labels were inferred from stars or model predictions."
        ),
    }


def propose_demo_annotations(root: Path, *, model_kind: str = "lr") -> dict:
    """Optional demonstration model suggestions, NEVER human-verified labels."""
    from .demo_data import DEMO
    from .model import train
    from .linguistic import contextual_signals
    if model_kind not in {"lr", "svm"}:
        raise CorpusError("Sólo lr o svm disponibles como modelos DEMO.")
    root = Path(root)
    reviews, _ = validate_reviews(root / "reviews.csv", root / "sources.csv")
    model = train([x for x, _ in DEMO], [y for _, y in DEMO], model_kind=model_kind)
    rows = []
    for review in reviews:
        scores = model.predict_proba([review["text"]])[0]
        index = int(scores.argmax())
        label = str(model.classes_[index])
        flags = contextual_signals(review["text"])
        # All suggestions require review: model is trained on fictional examples.
        rows.append({
            "review_id": review["review_id"],
            "suggested_sentiment": label,
            "max_score_demo_uncalibrated": f"{float(scores[index]):.4f}",
            "needs_human_review": "yes",
            "model_provenance": f"TFIDF_{model_kind}_TRAINED_ON_SYNTHETIC_DEMO",
            "context_flags": "contrast" if flags["posible_opinion_mixta"] else "none",
        })
    path = root / f"preannotations_demo_{model_kind}.csv"
    _write_csv(path, _PRELABEL_COLUMNS, rows)
    return {"status": "unverified_suggestions_only", "rows": len(rows),
            "review_required": len(rows), "filename": path.name,
            "warning": "Predicciones NO equivalen a anotaciones humanas ni gold standard."}


def audit_near_duplicates(root: Path, threshold: float = 0.88) -> dict:
    reviews, _ = validate_reviews(root / "reviews.csv", root / "sources.csv")
    result = near_duplicate_candidates(reviews, threshold=threshold)
    # IDs and similarities only; no texts or automatic record deletion.
    (root / "near_duplicate_candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"status": "human_review_queue", "candidates": len(result["candidates"]),
            "threshold": threshold}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Importación y sugerencias locales, sin scraping")
    parser.add_argument("--root", type=Path, default=Path("data/private/tourism-corpus-v1"))
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-csv")
    importer.add_argument("--file", required=True, type=Path)
    importer.add_argument("--source-id", required=True)
    importer.add_argument("--text-column", required=True)
    importer.add_argument("--group-column", required=True)
    importer.add_argument("--id-column")
    importer.add_argument("--language", default="unknown")
    importer.add_argument("--max-rows", type=int, default=10000)
    labeler = commands.add_parser("preannotate-demo")
    labeler.add_argument("--model-kind", default="lr", choices=["lr","svm"])
    audit = commands.add_parser("near-duplicates")
    audit.add_argument("--threshold", type=float, default=.88)
    args = parser.parse_args(argv)
    try:
        if args.command == "import-csv":
            outcome = ingest_authorized_csv(args.file, args.root, args.source_id,
                text_column=args.text_column, group_column=args.group_column,
                id_column=args.id_column, language=args.language, max_rows=args.max_rows)
        elif args.command == "preannotate-demo":
            outcome = propose_demo_annotations(args.root, model_kind=args.model_kind)
        else:
            outcome = audit_near_duplicates(args.root, threshold=args.threshold)
    except (CorpusError, OSError, UnicodeError, ValueError):
        parser.exit(2, "Corpus automation: importación detenida; revisa autorización, esquema y formato.\n")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
