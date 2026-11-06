"""Local, metadata-only CSV probe for PILOTO-01 (no training or label mapping).

This intentionally does NOT certify rights, human labels, privacy, language,
semantic near-duplicates, or release permissions. Category values are redacted
unless explicitly allowlisted. Headers remain private, untrusted metadata.
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


def _approved_counts(counts: Counter, allowed: tuple[str, ...]) -> dict:
    """Do not echo arbitrary cells mistaken for labels or language codes."""
    return {value: counts[value] for value in sorted(set(allowed)) if value in counts}


def inspect_csv(path: Path, *, text_column: str, label_column: str | None = None,
                id_column: str | None = None, language_column: str | None = None,
                group_columns: tuple[str, ...] = (),
                allowed_label_values: tuple[str, ...] = (),
                allowed_language_values: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """Return local metadata + PRIVATE fingerprints, not a publishable dataset.

    Repeated group columns mean independent leakage dimensions (hotel OR
    author), not a composite key (hotel AND author). Category allowlists only
    control disclosure; they never map, validate or establish label meaning.
    """
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ProbeError("Se requiere un CSV local existente.")
    category_columns = [c for c in (label_column, language_column) if c is not None]
    if text_column in category_columns or len(category_columns) != len(set(category_columns)):
        raise ProbeError("Texto, etiquetas e idioma requieren columnas distintas.")
    if len(group_columns) != len(set(group_columns)):
        raise ProbeError("Las columnas de agrupación no deben repetirse.")
    needed = {text_column, *(group_columns)}
    if label_column:
        needed.add(label_column)
    if id_column:
        needed.add(id_column)
    if language_column:
        needed.add(language_column)
    label_counts, language_counts = Counter(), Counter()
    id_counts, fingerprints = Counter(), Counter()
    normalized_id = set()
    group_presence = {column: set() for column in group_columns}
    blank_group_values = Counter({column: 0 for column in group_columns})
    text_labels = defaultdict(set)
    total = blank_text = blank_label = blank_id = total_length = 0
    original_sha = _sha(path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = reader.fieldnames
            if (not fields or any(not field.strip() for field in fields)
                    or len(fields) != len(set(fields))):
                raise ProbeError("CSV sin encabezados o con encabezados duplicados.")
            if not needed.issubset(fields):
                raise ProbeError("Columna solicitada no existe: corregir argumentos; no se hizo auditoría.")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ProbeError("CSV con filas malformadas o columnas desalineadas.")
                total += 1
                raw = row[text_column]
                label = ""
                if label_column:
                    label = row[label_column].strip()
                    if not label:
                        blank_label += 1
                    else:
                        label_counts[label] += 1
                if language_column:
                    # Source-provided codes are NOT a verified language detector.
                    language_counts[row[language_column].strip() or "<empty>"] += 1
                if id_column:
                    rid = row[id_column].strip()
                    if rid:
                        id_counts[rid] += 1
                        normalized_id.add(rid)
                    else:
                        blank_id += 1
                for column in group_columns:
                    value = row[column].strip()
                    if value:
                        # Namespace by column; the same code in two unrelated
                        # dimensions must not create a spurious overlap.
                        token = hashlib.sha256(json.dumps(
                            [column, value], ensure_ascii=False
                        ).encode("utf-8")).hexdigest()
                        group_presence[column].add(token)
                    else:
                        blank_group_values[column] += 1
                # Metadata above covers EVERY record, including unusable text.
                if not raw.strip():
                    blank_text += 1
                    continue
                token = hashlib.sha256(_normalize(raw).encode("utf-8")).hexdigest()
                fingerprints[token] += 1
                total_length += len(raw)
                if label:
                    text_labels[token].add(label)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ProbeError("CSV no legible como UTF-8 o con formato incorrecto.") from exc
    if _sha(path) != original_sha:
        raise ProbeError("El archivo cambió durante la auditoría; repetir con una copia estable.")
    group_keys = set().union(*group_presence.values())
    report = {
        "file_sha256": original_sha,
        "columns": fields,
        "headers_privacy": "PRIVATE_UNTRUSTED_METADATA_REVIEW_BEFORE_SHARING",
        "rows": total,
        "blank_text_rows": blank_text,
        "blank_label_rows": blank_label if label_column else None,
        "label_values_raw_UNMAPPED": _approved_counts(label_counts, allowed_label_values) if label_column else None,
        "redacted_label_value_counts": sorted(
            (n for value, n in label_counts.items() if value not in allowed_label_values),
            reverse=True,
        ) if label_column else None,
        "language_values_source_claims_UNVERIFIED": _approved_counts(language_counts, allowed_language_values) if language_column else None,
        "redacted_language_value_counts": sorted(
            (n for value, n in language_counts.items() if value not in allowed_language_values),
            reverse=True,
        ) if language_column else None,
        "blank_id_rows": blank_id if id_column else None,
        "id_duplicates_extra_rows": sum(n - 1 for n in id_counts.values() if n > 1) if id_column else None,
        "normalized_exact_text_duplicate_extra_rows": sum(n - 1 for n in fingerprints.values() if n > 1),
        "conflicting_raw_labels_same_normalized_text": sum(len(v) > 1 for v in text_labels.values()),
        "distinct_group_keys": len(group_keys) if group_columns else None,
        "distinct_group_keys_by_column": {
            column: len(values) for column, values in group_presence.items()
        } if group_columns else None,
        "blank_group_values_by_column": dict(blank_group_values) if group_columns else None,
        "group_key_semantics": "ANY_NONEMPTY_VALUE_PER_COLUMN_NOT_COMPOSITE",
        "mean_text_length_chars": round(total_length / (total - blank_text), 2)
            if total > blank_text else None,
        "mapping": "UNRESOLVED_NEEDS_ORIGINAL_DATA_DICTIONARY",
        "annotation_origin": "UNVERIFIED",
        "license_and_redistribution": "UNVERIFIED_NO_AUTHORIZATION_INFERRED",
        "near_duplicate_semantic_audit": "NOT_RUN",
        "privacy_review": "MANUAL_PENDING",
    }
    # Not serialized to stdout; includes hashes and source IDs solely for overlap.
    private = {"fingerprints": set(fingerprints), "ids": normalized_id,
               "groups": group_keys, "groups_by_column": group_presence}
    return report, private


def probe(train: Path, *, test: Path | None, text_column: str,
          label_column: str | None, id_column: str | None,
          language_column: str | None, group_columns: tuple[str, ...],
          allowed_label_values: tuple[str, ...] = (),
          allowed_language_values: tuple[str, ...] = ()) -> dict:
    args = dict(text_column=text_column, id_column=id_column,
                language_column=language_column, group_columns=group_columns,
                allowed_language_values=allowed_language_values)
    training, train_private = inspect_csv(
        train, label_column=label_column, allowed_label_values=allowed_label_values, **args
    )
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
            "shared_group_keys_by_column": {
                column: len(train_private["groups_by_column"][column]
                            & test_private["groups_by_column"][column])
                for column in group_columns
            } if group_columns else None,
        }
    return {
        "report_type": "PILOTO01_METADATA_PROBE_NOT_ADMISSION",
        "schema_version": "piloto01_metadata_probe_v2",
        "train": training, "test": testing, "overlap": overlap,
        "admission": "BLOCKED_UNTIL_LICENSE_LABEL_ORIGIN_PRIVACY_AND_MAPPING_VERIFIED",
        "warning": (
            "Statistics describe only locally provided files. No automatic numeric label "
            "mapping, language certification, source consent, human gold certification, "
            "semantic near-duplicate detection or scientific model evaluation. "
            "Headers and explicitly allowlisted category values remain private "
            "metadata requiring review before sharing. Group keys are not connected components."
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
    parser.add_argument("--allow-label-value", action="append", default=[],
                        help="Código original revisado que puede mostrarse; no define su significado.")
    parser.add_argument("--allow-language-value", action="append", default=[],
                        help="Código de idioma revisado que puede mostrarse; no certifica el idioma.")
    args = parser.parse_args(argv)
    try:
        result = probe(
            args.train, test=args.test, text_column=args.text_column,
            label_column=args.label_column, id_column=args.id_column,
            language_column=args.language_column,
            group_columns=tuple(args.group_column),
            allowed_label_values=tuple(args.allow_label_value),
            allowed_language_values=tuple(args.allow_language_value),
        )
    except (ProbeError, OSError) as exc:
        # OSError may contain a private full path, account name or mount name.
        parser.exit(2, f"Auditoría detenida: {type(exc).__name__}; revise archivo y opciones localmente.\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
