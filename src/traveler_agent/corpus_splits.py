"""Leakage-aware deterministic train/validation/test preparation.

Splits only manually adjudicated, non-unknown labels. Near duplicates require
separate human review; no near-duplicate detection is claimed here.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .corpus import (
    CorpusError, REVIEW_COLUMNS, ADJUDICATION_COLUMNS, _write_csv,
    file_sha, text_sha, validate_reviews, validate_annotations,
    validate_adjudications,
)

TRAIN_COLUMNS = REVIEW_COLUMNS + ("sentiment", "aspects_json", "split")


def prepare_splits(
    reviews_path: Path, sources_path: Path, annotations_path: Path,
    adjudications_path: Path, out_dir: Path, *, seed: int = 42
) -> dict:
    """Create CSVs for model development, plus non-text metadata manifest.

    Every identical text and every group_id stays in a single split, even if
    they originate in different sources. Never fit vectorizers or classifiers.
    """
    reviews, review_report = validate_reviews(reviews_path, sources_path)
    annotations, annotation_report = validate_annotations(
        annotations_path, {r["review_id"] for r in reviews}
    )
    adjudications, adjudication_report = validate_adjudications(
        adjudications_path, {r["review_id"] for r in reviews}, annotations
    )
    reviewed = {r["review_id"]: r for r in adjudications
                if r["sentiment"] in ("negativo", "neutral", "positivo")}
    selected = [r for r in reviews if r["review_id"] in reviewed]
    if not selected:
        raise CorpusError("Sin adjudicaciones completas; no se generan particiones.")
    if set(reviewed[r["review_id"]]["sentiment"] for r in selected) != {
        "negativo", "neutral", "positivo"
    }:
        raise CorpusError("El conjunto adjudicado debe contener las tres clases.")

    # Union-find: connect records that share group_id OR normalized exact text.
    ids = [r["review_id"] for r in selected]
    parent = {rid: rid for rid in ids}

    def root(rid: str) -> str:
        while parent[rid] != rid:
            parent[rid] = parent[parent[rid]]
            rid = parent[rid]
        return rid

    def union(a: str, b: str) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    fingerprints = {}
    for row in selected:
        rid = row["review_id"]
        group = row["group_id"]
        fingerprint = text_sha(row["text"])
        if group in groups:
            union(rid, groups[group])
        groups[group] = rid
        if fingerprint in fingerprints:
            other = fingerprints[fingerprint]
            if reviewed[rid]["sentiment"] != reviewed[other]["sentiment"]:
                raise CorpusError("Duplicados exactos con etiquetas adjudicadas contradictorias.")
            union(rid, other)
        fingerprints[fingerprint] = rid

    components = defaultdict(list)
    for row in selected:
        components[root(row["review_id"])].append(row)
    if len(components) < 9:
        raise CorpusError("Se exigen al menos 9 grupos independientes para tres particiones.")

    # Stable ordering independent of input row order. Do not hash raw text.
    ordered = sorted(
        components.values(),
        key=lambda rows: hashlib.sha256(
            f"{seed}:{min(r['review_id'] for r in rows)}".encode("utf-8")
        ).hexdigest(),
    )
    # Whole components; no group split even if sizes vary.
    n = len(ordered)
    n_test = max(1, round(n * .2))
    n_val = max(1, round(n * .2))
    assignments = (
        [("test", group) for group in ordered[:n_test]]
        + [("validation", group) for group in ordered[n_test:n_test+n_val]]
        + [("train", group) for group in ordered[n_test+n_val:]]
    )
    prepared = {"train": [], "validation": [], "test": []}
    for split, cluster in assignments:
        for row in cluster:
            label = reviewed[row["review_id"]]
            prepared[split].append({
                **row, "sentiment": label["sentiment"],
                "aspects_json": label["aspects_json"], "split": split,
            })

    # Fail closed if one split lacks a class. Larger or stratified corpora needed.
    per_split = {}
    for name, rows in prepared.items():
        counts = Counter(row["sentiment"] for row in rows)
        if any(counts[label] == 0 for label in ("negativo", "neutral", "positivo")):
            raise CorpusError(
                f"La partición {name} no contiene las tres clases. "
                "Amplía corpus o revisa el balance antes de entrenar."
            )
        per_split[name] = {"rows": len(rows), "labels": dict(sorted(counts.items())),
                           "groups": len({root(r["review_id"]) for r in rows})}

    output_dir = Path(out_dir)
    if not any(name in output_dir.parts for name in ("private", "processed")):
        raise CorpusError("Los conjuntos deben guardarse dentro de data/private o data/processed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in prepared.items():
        _write_csv(output_dir / f"{name}.csv", TRAIN_COLUMNS,
                   sorted(rows, key=lambda r: r["review_id"]))

    manifest = {
        "purpose": "PREPARACION_DATOS; NO_EVALUACION_CIENTIFICA",
        "seed": seed, "strategy": "union_group_and_exact_duplicate_then_hash_split_60_20_20",
        "input_hashes": {
            "reviews": file_sha(reviews_path),
            "sources": file_sha(sources_path),
            "annotations": file_sha(annotations_path),
            "adjudications": file_sha(adjudications_path),
        },
        "input_summary": {
            "total_reviews": review_report["rows"],
            "annotated_reviews": annotation_report["annotated_reviews"],
            "adjudicated_reviews": adjudication_report["adjudications"],
            "eligible_reviews": len(selected),
        },
        "partitions": per_split,
        "output_hashes": {name: file_sha(output_dir / f"{name}.csv")
                          for name in prepared},
        "limitations": [
            "No train/validation/test leakage for shared group_id or exact normalized text.",
            "Near duplicates and real-world author overlap are NOT automatically detected.",
            "The 60/20/20 allocation counts connected components, not exact row percentages.",
            "Training and hyperparameter tuning must use train/validation only; test is held out.",
            "Evidence of consent and license requires independent human review.",
        ],
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
