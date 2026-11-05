"""Local corpus management CLI. All raw text stays in gitignored data/private.

Example:
python -m traveler_agent.corpus_cli init
python -m traveler_agent.corpus_cli validate
python -m traveler_agent.corpus_cli annotate --review-id r01 --annotator a01 --sentiment neutral --aspects '{}'
python -m traveler_agent.corpus_cli adjudicate --review-id r01 --adjudicator lead --sentiment neutral --aspects '{}' --reason "..."
python -m traveler_agent.corpus_cli split
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .corpus import (
    CorpusError, REVIEW_COLUMNS, SOURCE_COLUMNS,
    _read_csv, _write_csv, record_annotation, record_adjudication,
    validate_sources, validate_reviews, validate_annotations,
    validate_adjudications, parse_aspects, ANNOTATION_COLUMNS, ADJUDICATION_COLUMNS
)
from .corpus_splits import prepare_splits

DEFAULT_ROOT = Path("data/private/tourism-corpus-v1")


def run(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Corpus académico privado: sin scraping ni publicación")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Crear plantillas CSV locales vacías")
    commands.add_parser("validate", help="Auditar esquemas, procedencia y anotaciones")
    annotation = commands.add_parser("annotate", help="Añadir anotación humana independiente")
    annotation.add_argument("--review-id", required=True)
    annotation.add_argument("--annotator", required=True)
    annotation.add_argument("--sentiment", required=True)
    annotation.add_argument("--aspects", default="{}")
    adjudication = commands.add_parser("adjudicate", help="Registrar etiqueta final justificada")
    adjudication.add_argument("--review-id", required=True)
    adjudication.add_argument("--adjudicator", required=True)
    adjudication.add_argument("--sentiment", required=True)
    adjudication.add_argument("--aspects", default="{}")
    adjudication.add_argument("--reason", required=True)
    contribution = commands.add_parser("contribute", help="Importar un texto ofrecido voluntariamente")
    contribution.add_argument("--text-file", type=Path, required=True)
    contribution.add_argument("--source-id", required=True)
    contribution.add_argument("--group-id", required=True)
    contribution.add_argument("--language", default="unknown")
    contribution.add_argument("--consent-confirmed", action="store_true", help="Confirmación explícita de consentimiento documentado")
    split = commands.add_parser("split", help="Preparar conjuntos privados y manifiesto")
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    root = args.root
    # Fail closed: refuse paths that Git would normally publish.
    if not any(part in root.parts for part in ("private", "processed")):
        raise CorpusError("La raíz debe estar bajo una carpeta private o processed.")
    sources = root / "sources.csv"
    reviews = root / "reviews.csv"
    annotations = root / "annotations.csv"
    adjudications = root / "adjudications.csv"
    if args.command == "init":
        for path, columns in ((sources, SOURCE_COLUMNS), (reviews, REVIEW_COLUMNS),
                              (annotations, ANNOTATION_COLUMNS),
                              (adjudications, ADJUDICATION_COLUMNS)):
            if not path.exists():
                _write_csv(path, columns, [])
        return {"status": "plantillas_creadas", "root": str(root)}

    if args.command == "contribute":
        if not args.consent_confirmed:
            raise CorpusError("Se exige consentimiento voluntario explícito y documentado.")
        registered, _ = validate_sources(sources)
        if not any(s["source_id"] == args.source_id and
                   s["source_type"] == "voluntary" and s["reuse_basis"] == "consent"
                   for s in registered):
            raise CorpusError("La fuente voluntaria no tiene registro de consentimiento revisado.")
        if args.language not in {"es", "en", "it", "other", "unknown"}:
            raise CorpusError("Idioma no admitido.")
        from .corpus import _require_identifier
        _require_identifier(args.group_id, "group_id")
        data = args.text_file.read_bytes()
        if len(data) > 20000:
            raise CorpusError("Archivo de contribución demasiado grande.")
        try:
            text = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise CorpusError("Texto debe ser UTF-8.") from exc
        if not 3 <= len(text) <= 5000:
            raise CorpusError("Reseña fuera de 3–5000 caracteres.")
        rows = _read_csv(reviews, REVIEW_COLUMNS)
        rid = "r_" + uuid.uuid4().hex
        rows.append({"review_id": rid, "text": text, "source_id": args.source_id,
                     "group_id": args.group_id, "language": args.language})
        _write_csv(reviews, REVIEW_COLUMNS, rows)
        return {"status": "contribucion_local", "review_id": rid,
                "warning": "El operador debe revisar PII; consentimiento no se verifica automáticamente."}

    rows, review_stats = validate_reviews(reviews, sources)
    valid_ids = {row["review_id"] for row in rows}
    if args.command == "annotate":
        record_annotation(annotations, valid_ids, review_id=args.review_id,
                          annotator_id=args.annotator, sentiment=args.sentiment,
                          aspects=parse_aspects(args.aspects))
        return {"status": "anotacion_registrada", "review_id": args.review_id}
    existing_annotations, annotations_stats = validate_annotations(annotations, valid_ids)
    if args.command == "adjudicate":
        record_adjudication(adjudications, args.review_id, args.sentiment,
                            parse_aspects(args.aspects), args.adjudicator,
                            args.reason, rows, existing_annotations)
        return {"status": "adjudicacion_registrada", "review_id": args.review_id}
    if args.command == "validate":
        _, ad_stats = validate_adjudications(adjudications, valid_ids, existing_annotations)
        return {"status": "auditoria_local", "reviews": review_stats,
                "annotations": annotations_stats, "adjudications": ad_stats}
    if args.command == "split":
        report = prepare_splits(reviews, sources, annotations, adjudications,
                                Path("data/processed/tourism-corpus-v1"),
                                seed=args.seed)
        return {"status": "particiones_preparadas", "manifest": report}
    raise AssertionError("Comando no implementado")


def main():
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    except (CorpusError, OSError, UnicodeError) as exc:
        # Avoid printing raw review text in error messages.
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
