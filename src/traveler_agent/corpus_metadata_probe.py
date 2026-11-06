"""Local, metadata-only CSV probe for PILOTO-01 (no training or label mapping).

This intentionally does NOT certify rights, human labels, privacy, language,
semantic near-duplicates, or release permissions. Never publishes raw reviews.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import unicodedata


class ProbeError(ValueError):
    """Fail without echoing potentially sensitive data."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def inspect_csv(path: Path, *, text_column: str, label_column: str | None = None,
                id_column: str | None = None, language_column: str | None = None,
                group_columns: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """Return aggregate report + PRIVATE fingerprints; never include raw texts."""
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ProbeError("Se requiere un CSV local existente.")
    needed = {text_column, *(group_columns)}
    if label_column:
        needed.add(label_column)
    if id_column:
        needed.add(id_column)
    if language_column:
        needed.add(language_column)
    label_counts, language_counts, lengths = Counter(), Counter(), []
    id_counts, group_counts, fingerprints = Counter(), Counter(), Counter()
    normalized_id = set()
    group_presence = set()
    text_labels = defaultdict(set)
    total = blank_text = blank_label = 0
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)):
                raise ProbeError("CSV sin encabezados o con encabezados duplicados.")
            if not needed.issubset(fields):
                raise ProbeError("Columna solicitada no existe: corregir argumentos; no se hizo auditoría.")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ProbeError("CSV con filas malformadas o columnas desalineadas.")
                total += 1
                raw = row[text_column]
                if not raw.strip():
                    blank_text += 1
                    continue
                token = hashlib.sha256(_normalize(raw).encode("utf-8")).hexdigest()
                fingerprints[token] += 1
                lengths.append(len(raw))
                if label_column:
                    label = row[label_column].strip()
                    if not label:
                        blank_label += 1
                    else:
                        label_counts[label] += 1
                        text_labels[token].add(label)
                if language_column:
                    # Source-provided codes are NOT a verified language detector.
                    language_counts[row[language_column].strip() or "<empty>"] += 1
                if id_column:
                    rid = row[id_column].strip()
                    if rid:
                        id_counts[rid] += 1
                        normalized_id.add(rid)
                if group_columns:
                    group_tuple = tuple(row[k].strip() for k in group_columns)
                    if all(group_tuple):
                        group_hash = hashlib.sha256(
                            json.dumps(group_tuple, ensure_ascii=False).encode("utf-8")
                        ).hexdigest()
                        group_counts[group_hash] += 1
                        group_presence.add(group_hash)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ProbeError("CSV no legible como UTF-8 o con formato incorrecto.") from exc
    report = {
        "file_sha256": _sha(path),
        "columns": fields,
        "rows": total,
        "blank_text_rows": blank_text,
        "blank_label_rows": blank_label if label_column else None,
        "label_values_raw_UNMAPPED": dict(sorted(label_counts.items())) if label_column else None,
        "language_values_source_claims_UNVERIFIED": dict(sorted(language_counts.items())) if language_column else None,
        "id_duplicates_extra_rows": sum(n - 1 for n in id_counts.values() if n > 1) if id_column else None,
        "normalized_exact_text_duplicate_extra_rows": sum(n - 1 for n in fingerprints.values() if n > 1),
        "conflicting_raw_labels_same_normalized_text": sum(len(v) > 1 for v in text_labels.values()),
        "distinct_group_keys": len(group_presence) if group_columns else None,
        "mean_text_length_chars": round(sum(lengths) / len(lengths), 2) if lengths else None,
        "mapping": "UNRESOLVED_NEEDS_ORIGINAL_DATA_DICTIONARY",
        "annotation_origin": "UNVERIFIED",
        "license_and_redistribution": "UNVERIFIED_NO_AUTHORIZATION_INFERRED",
        "near_duplicate_semantic_audit": "NOT_RUN",
        "privacy_review": "MANUAL_PENDING",
    }
    # Not serialized to stdout; includes hashes and source IDs solely for overlap.
    private = {"fingerprints": set(fingerprints), "ids": normalized_id, "groups": group_presence}
    return report, private


def probe(train: Path, *, test: Path | None, text_column: str,
          label_column: str | None, id_column: str | None,
          language_column: str | None, group_columns: tuple[str, ...]) -> dict:
    args = dict(text_column=text_column, id_column=id_column,
                language_column=language_column, group_columns=group_columns)
    training, train_private = inspect_csv(train, label_column=label_column, **args)
    testing = None
    overlap = None
    if test:
        testing, test_private = inspect_csv(test, label_column=None, **args)
        overlap = {
            "normalized_exact_text_shared_count": len(
                train_private["fingerprints"] & test_private["fingerprints"]
            ),
            "shared_nonempty_ids_count": len(train_private["ids"] & test_private["ids"])
                if id_column else None,
            "shared_group_keys_count": len(train_private["groups"] & test_private["groups"])
                if group_columns else None,
        }
    return {
        "report_type": "PILOTO01_METADATA_PROBE_NOT_ADMISSION",
        "train": training, "test": testing, "overlap": overlap,
        "admission": "BLOCKED_UNTIL_LICENSE_LABEL_ORIGIN_PRIVACY_AND_MAPPING_VERIFIED",
        "warning": (
            "Statistics describe only locally provided files. No automatic numeric label "
            "mapping, language certification, source consent, human gold certification, "
            "semantic near-duplicate detection or scientific model evaluation."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audita metadatos locales sin publicar reseñas.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column")
    parser.add_argument("--id-column")
    parser.add_argument("--language-column")
    parser.add_argument("--group-column", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = probe(
            args.train, test=args.test, text_column=args.text_column,
            label_column=args.label_column, id_column=args.id_column,
            language_column=args.language_column,
            group_columns=tuple(args.group_column),
        )
    except (ProbeError, OSError) as exc:
        parser.exit(2, f"Auditoría detenida: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
