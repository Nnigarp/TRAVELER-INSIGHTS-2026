"""Reproducible, group-aware research benchmark on an adjudicated PRIVATE corpus.

No scraping. No fitting on test data. No metrics on demo fixtures are presented
as scientific evidence. User controls consent, licensing and data access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib.metadata import version as package_version, PackageNotFoundError
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support
)

from .corpus import (
    CorpusError, text_sha, file_sha, validate_sources,
    validate_reviews, validate_annotations, validate_adjudications
)
from .corpus_splits import TRAIN_COLUMNS
from .labels import LABELS
from .model import build_pipeline

MODEL_GRID = {
    "dummy": [{}],
    "lr": [{"C": 0.5}, {"C": 2.0}],
    "svm": [{"C": 0.5}, {"C": 2.0}],
}
SPLITS = ("train", "validation", "test")


def _read_split(path: Path, expected_split: str) -> list[dict]:
    if not path.is_file():
        raise CorpusError("Falta un archivo de partición.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(TRAIN_COLUMNS):
            raise CorpusError("Esquema de partición inválido.")
        rows = list(reader)
    if not rows:
        raise CorpusError("Partición vacía.")
    ids = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise CorpusError("Fila de partición malformada.")
        if row["split"] != expected_split:
            raise CorpusError("Campo split no coincide con archivo.")
        if row["sentiment"] not in LABELS:
            raise CorpusError("Solo se admiten etiquetas adjudicadas de las tres clases.")
        if not row["text"].strip() or not row["group_id"]:
            raise CorpusError("Faltan texto o grupo.")
        if row["review_id"] in ids:
            raise CorpusError("ID duplicado dentro de una partición.")
        ids.add(row["review_id"])
    return rows


def verified_splits(corpus_dir: Path, split_dir: Path) -> tuple[dict, dict]:
    """Check file hashes, source authorization and leakage BEFORE any model fit."""
    corpus_dir, split_dir = Path(corpus_dir), Path(split_dir)
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise CorpusError("Falta el manifiesto de particiones.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise CorpusError("Manifiesto ilegible.") from exc
    if manifest.get("purpose") != "PREPARACION_DATOS; NO_EVALUACION_CIENTIFICA":
        raise CorpusError("Manifiesto de procedencia inesperado.")

    inputs = {
        "sources": corpus_dir / "sources.csv",
        "reviews": corpus_dir / "reviews.csv",
        "annotations": corpus_dir / "annotations.csv",
        "adjudications": corpus_dir / "adjudications.csv",
    }
    for name, path in inputs.items():
        if not path.is_file() or file_sha(path) != manifest.get("input_hashes", {}).get(name):
            raise CorpusError("Archivo privado de procedencia ausente o modificado; regenerar splits.")
    sources, _ = validate_sources(inputs["sources"])
    reviews, _ = validate_reviews(inputs["reviews"], inputs["sources"])
    annotations, _ = validate_annotations(
        inputs["annotations"], {r["review_id"] for r in reviews}
    )
    adjudications, _ = validate_adjudications(
        inputs["adjudications"], {r["review_id"] for r in reviews}, annotations
    )
    if not sources or not adjudications:
        raise CorpusError("Faltan fuentes aprobadas y etiquetas adjudicadas.")

    splits = {}
    for name in SPLITS:
        path = split_dir / f"{name}.csv"
        if not path.is_file() or file_sha(path) != manifest.get("output_hashes", {}).get(name):
            raise CorpusError("Partición alterada: regenerar conjuntos y manifiesto.")
        splits[name] = _read_split(path, name)
        stats = manifest.get("partitions", {}).get(name, {})
        if stats.get("rows") != len(splits[name]):
            raise CorpusError("Recuento de filas del manifiesto inconsistente.")
        counts = Counter(r["sentiment"] for r in splits[name])
        if set(counts) != set(LABELS):
            raise CorpusError("Falta al menos una clase por partición.")

    # Verify EXACT correspondence to adjudicated rows; no injected samples or labels.
    source_by_id = {r["review_id"]: r for r in reviews}
    accepted = {r["review_id"]: r for r in adjudications if r["sentiment"] in LABELS}
    all_ids, groups, normalized_texts = set(), {}, {}
    for split, rows in splits.items():
        for row in rows:
            rid = row["review_id"]
            if rid in all_ids:
                raise CorpusError("ID repetido entre particiones.")
            all_ids.add(rid)
            raw, final = source_by_id.get(rid), accepted.get(rid)
            if raw is None or final is None:
                raise CorpusError("Partición con reseña no adjudicada.")
            if (any(row[k] != raw[k] for k in ("text","group_id","source_id","language"))
                    or row["sentiment"] != final["sentiment"]
                    or json.loads(row["aspects_json"]) != json.loads(final["aspects_json"])):
                raise CorpusError("Partición no coincide con fuente o adjudicación.")
            group = row["group_id"]
            token = text_sha(row["text"])
            if group in groups and groups[group] != split:
                raise CorpusError("Fuga entre particiones: group_id compartido.")
            if token in normalized_texts and normalized_texts[token] != split:
                raise CorpusError("Fuga entre particiones: texto duplicado.")
            groups[group] = split
            normalized_texts[token] = split
    if all_ids != set(accepted):
        raise CorpusError("Se omitieron reseñas adjudicadas en las particiones.")
    return splits, manifest



def _training_calibration_folds(rows: list[dict], seed: int):
    """Group-safe inner folds for SVM calibration, including exact duplicates."""
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_group, by_text = {}, {}
    for i, row in enumerate(rows):
        group, fingerprint = row["group_id"], text_sha(row["text"])
        if group in by_group:
            union(i, by_group[group])
        if fingerprint in by_text:
            union(i, by_text[fingerprint])
        by_group[group] = i
        by_text[fingerprint] = i

    groups = [root(i) for i in range(len(rows))]
    labels = [r["sentiment"] for r in rows]
    if len(set(groups)) < 3:
        raise CorpusError("SVM necesita al menos 3 grupos independientes en train.")
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(rows)), labels, groups))
    for fit_ids, cal_ids in folds:
        if set(labels[i] for i in fit_ids) != set(LABELS):
            raise CorpusError("No hay tres clases en cada fold de calibración.")
        if set(labels[i] for i in cal_ids) != set(LABELS):
            raise CorpusError("Un fold de calibración carece de una clase.")
        if set(groups[i] for i in fit_ids) & set(groups[i] for i in cal_ids):
            raise CorpusError("Fuga de grupos en calibración SVM.")
    return folds


def _new_estimator(kind: str, params: dict, seed: int):
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent")
    estimator = build_pipeline(kind)
    if kind == "lr":
        estimator.set_params(classifier__C=params["C"], classifier__random_state=seed)
    elif kind == "svm":
        estimator.set_params(estimator__classifier__C=params["C"],
                             estimator__classifier__random_state=seed)
    else:
        raise CorpusError("Clasificador no admitido.")
    return estimator


def _score(model, rows: list[dict]) -> dict:
    truth = [r["sentiment"] for r in rows]
    prediction = list(model.predict([r["text"] for r in rows]))
    if len(prediction) != len(truth):
        raise CorpusError("Predicciones incompletas.")
    result = {
        "n": len(rows),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS,
                                   average="macro", zero_division=0)),
        "classification_report": classification_report(
            truth, prediction, labels=LABELS, output_dict=True, zero_division=0,
        ),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=LABELS).tolist(),
        "by_language": {},
    }
    for lang in sorted(set(r["language"] for r in rows)):
        indices = [i for i, r in enumerate(rows) if r["language"] == lang]
        gt = [truth[i] for i in indices]
        pred = [prediction[i] for i in indices]
        result["by_language"][lang] = {
            "n": len(indices),
            "macro_f1": float(f1_score(gt, pred, labels=LABELS,
                                       average="macro", zero_division=0)),
        }
    return result


def run_benchmark(corpus_dir: Path, split_dir: Path, output_dir: Path,
                  *, seed: int = 42, demo_only: bool = False) -> dict:
    splits, manifest = verified_splits(corpus_dir, split_dir)
    if seed != manifest.get("seed"):
        raise CorpusError("La semilla no coincide con la preparación de particiones.")
    out = Path(output_dir)
    if not any(part in out.parts for part in ("private", "processed", "artifacts", "reports")):
        raise CorpusError("La salida debe mantenerse en carpetas privadas o ignoradas por Git.")
    if out.exists() and any(out.iterdir()):
        raise CorpusError("Directorio de resultados no vacío: usar uno nuevo para evitar sobrescritura.")
    train_rows, valid_rows, test_rows = (splits[s] for s in SPLITS)
    train_labels = Counter(r["sentiment"] for r in train_rows)
    if min(train_labels[label] for label in LABELS) < 3:
        raise CorpusError("SVM con calibración CV=3 requiere tres ejemplos por clase en train.")

    comparison = {}
    selected = {}
    # Group-aware CV keeps related reviews out of SVM calibration folds.
    calibration_folds = _training_calibration_folds(train_rows, seed)
    # Selection sees train+validation ONLY. Test never enters fitting or tuning.
    for kind in MODEL_GRID:
        attempts = []
        for params in MODEL_GRID[kind]:
            clf = _new_estimator(kind, params, seed)
            if kind == "svm":
                clf.set_params(cv=calibration_folds)
            clf.fit([r["text"] for r in train_rows],
                    [r["sentiment"] for r in train_rows])
            validation = _score(clf, valid_rows)
            attempts.append((validation["macro_f1"], params, clf, validation))
        # Stable tie break: first candidate in explicit grid wins.
        best = max(enumerate(attempts), key=lambda item: (item[1][0], -item[0]))[1]
        score, params, clf, val_metrics = best
        selected[kind] = clf
        comparison[kind] = {
            "selected_parameters": params,
            "validation": val_metrics,
            "candidate_validation_macro_f1": [
                {"parameters": item[1], "macro_f1": item[0]} for item in attempts
            ],
        }

    # Single held-out evaluation per selected model. Never use this result to
    # change hyperparameters, refit models or select "the winner" scientifically.
    for kind, clf in selected.items():
        comparison[kind]["test"] = _score(clf, test_rows)

    out.mkdir(parents=True, exist_ok=True)
    # Forbid untrusted joblib: these files must not be loaded from shared sources.
    for kind, model in selected.items():
        joblib.dump(model, out / f"{kind}.joblib")
    # Version manifest records the actually installed environment, not a guessed
    # frozen lockfile; compatibility tests still run with Python 3.12.
    packages = ("scikit-learn", "numpy", "scipy", "pandas", "joblib",
                "threadpoolctl", "gradio", "faster-whisper", "ctranslate2", "av")
    installed_versions = {}
    for package in packages:
        try:
            installed_versions[package] = package_version(package)
        except PackageNotFoundError:
            installed_versions[package] = "NOT_INSTALLED"
    report = {
        "report_type": "SOFTWARE_DEMO_ONLY" if demo_only else "RESEARCH_CANDIDATE_NOT_INDEPENDENTLY_AUDITED",
        "scientific_caveat": (
            "Dataset provenance and annotation quality require external audit. "
            "Never present synthetic fixtures as tourism findings. "
            "Calibration reliability has not been established on an external cohort."
        ),
        "method": "single_group_aware_train_validation_test_split",
        "seed": seed,
        "hashes": {
            "split_manifest_sha256": file_sha(split_dir / "split_manifest.json"),
            "train": file_sha(split_dir / "train.csv"),
            "validation": file_sha(split_dir / "validation.csv"),
            "test": file_sha(split_dir / "test.csv"),
        },
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "installed_package_versions": installed_versions,
            "platform": platform.platform(),
        },
        "splits": {name: {"n": len(rows)} for name, rows in splits.items()},
        "models": comparison,
        "limits": [
            "Test evaluated exactly once per chosen model; comparison across models is descriptive.",
            "Not a nested-CV or cross-source validation.",
            "Near-duplicate and PII checks still require manual audit.",
            "SVM uses stratified group-disjoint calibration within TRAIN, not validation/test.",
            "Machine learning models are serialized with joblib; only load files created locally.",
        ],
    }
    (out / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_manifest = {
        "selection": "best_validation_macro_f1_per_model",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": report["report_type"],
        "provenance": report["hashes"],
        "model_metadata": {
            kind: {
                "sha256": file_sha(out / f"{kind}.joblib"),
                "parameters": comparison[kind]["selected_parameters"],
                "classes": list(map(str, selected[kind].classes_)),
            } for kind in selected
        },
    }
    (out / "model_manifest.json").write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark privado: dummy/LR/SVM sobre corpus adjudicado, sin Kaggle."
    )
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path("data/private/tourism-corpus-v1"))
    parser.add_argument("--split-dir", type=Path,
                        default=Path("data/processed/tourism-corpus-v1"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/corpus-benchmark-v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-only", action="store_true",
                        help="Indicar datos sintéticos: resultados NO científicos")
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.corpus_dir, args.split_dir, args.output_dir,
                               seed=args.seed, demo_only=args.demo_only)
    except (CorpusError, FileNotFoundError, OSError, ValueError) as exc:
        parser.exit(status=2, message=f"Benchmark detenido: {type(exc).__name__}\n")
    print(json.dumps({
        "status": "ok", "report_type": report["report_type"],
        "models": list(report["models"]),
        "note": "Los resultados se guardan localmente; no publicar datos privados.",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
