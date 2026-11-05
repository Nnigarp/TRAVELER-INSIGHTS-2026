"""One-time external-to-generator SYNTHETIC evaluation of frozen baseline setup.

Baseline run 42006977003 did NOT archive joblib files, only original synthetic
CSV, manifest and report. We therefore reconstruct each original fitted model
ONCE from its original TRAIN split with originally selected hyperparameters,
without using challenge samples or changing any baseline source. This is not
literal loading of previously serialized baseline weights, a material caveat.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import CorpusError, file_sha, text_sha
from .scientific_benchmark import (
    MODEL_GRID, _new_estimator, _score, _training_calibration_folds
)
from .synthetic_tourism import (
    FIELDS as BASE_FIELDS, GENERATOR, MODEL, VERSION, audit_records
)
from .synthetic_evaluation import group_split
from .out_of_generator_challenge import (
    FIELDS as CHALLENGE_FIELDS, VERSION as CHALLENGE_VERSION,
    SOURCE as CHALLENGE_SOURCE, audit_challenge
)

BASELINE_CSV_SHA = "315012f9d553c51906e49bac617ef7ed9107ee2fc3cfd4dfd5a1368c7a4be059"
BASELINE_RUN = 42006977003
ORIGINAL_SEED = 42
CHALLENGE_SEED = 913
MODEL_NAMES = ("dummy", "lr", "svm")


def _csv(path: Path, fields: tuple[str, ...]):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or []) != fields:
            raise CorpusError("Esquema inesperado, no importar.")
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise CorpusError("Fila malformada.")
    return rows


def load_frozen_baseline(root: Path):
    root = Path(root)
    datafile = root / "synthetic_reviews.csv"
    manifest_path = root / "synthetic_manifest.json"
    report_path = root / "synthetic_experiment_report.json"
    if not all(p.is_file() for p in (datafile, manifest_path, report_path)):
        raise CorpusError("Faltan los 3 archivos originales del run congelado.")
    if file_sha(datafile) != BASELINE_CSV_SHA:
        raise CorpusError("SHA-256 base congelado NO coincide.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not (manifest.get("seed") == ORIGINAL_SEED
            and manifest.get("rows_sha256") == BASELINE_CSV_SHA
            and manifest.get("method") == GENERATOR
            and manifest.get("model") == MODEL
            and manifest.get("version") == VERSION):
        raise CorpusError("El lote original no coincide con manifiesto verificado.")
    if not (report.get("report_type") == "SYNTHETIC_INTERNAL_EVALUATION_ONLY"
            and report.get("input_sha256") == BASELINE_CSV_SHA
            and report.get("generator", {}).get("seed") == ORIGINAL_SEED):
        raise CorpusError("Informe original no coincide.")
    rows = _csv(datafile, BASE_FIELDS)
    original_audit = audit_records(rows)
    splits = group_split(rows, seed=ORIGINAL_SEED)
    if len(rows) != 336 or original_audit["families"] != 56:
        raise CorpusError("El corpus original fue alterado.")
    for name, records in splits.items():
        expected = report["counts"][name]
        if expected["rows"] != len(records) or expected["families"] != len({r["scenario_family"] for r in records}):
            raise CorpusError("Particiones diferentes de la evaluación original.")
    params = {}
    for kind in MODEL_NAMES:
        declared = report["models"][kind]["parameters"]
        if declared not in MODEL_GRID[kind]:
            raise CorpusError("Hiperparámetro no presente en parrilla original.")
        params[kind] = declared
    return splits, report, params, {"data_sha256": file_sha(datafile),
                                    "manifest_sha256": file_sha(manifest_path),
                                    "report_sha256": file_sha(report_path)}


def load_challenge(root: Path, *, baseline_texts: list[str]):
    root = Path(root)
    csv_path = root / "challenge_synthetic.csv"
    manifest_path = root / "challenge_manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise CorpusError("Falta desafío sintético con manifiesto.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("type") != "SYNTHETIC_CHALLENGE_NOT_REAL"
            or manifest.get("generation", {}).get("seed") != CHALLENGE_SEED
            or manifest.get("generation", {}).get("version") != CHALLENGE_VERSION
            or manifest.get("generation", {}).get("method") != CHALLENGE_SOURCE
            or manifest.get("data_sha256") != file_sha(csv_path)):
        raise CorpusError("Desafío no fijado o hash alterado.")
    rows = _csv(csv_path, CHALLENGE_FIELDS)
    audit = audit_challenge(rows, original_texts=baseline_texts)
    return rows, audit, manifest


def near_duplicate_audit(baseline_texts: list[str], challenge_texts: list[str], *,
                         threshold: float = .96):
    """Char-TF-IDF nearest-neighbor cross-set check; detects candidates only."""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                 min_df=1, sublinear_tf=True)
    x = vectorizer.fit_transform(baseline_texts + challenge_texts)
    old = x[:len(baseline_texts)]
    fresh = x[len(baseline_texts):]
    sim = cosine_similarity(fresh, old)
    highest = sim.max(axis=1)
    return {
        "method": "char_tfidf_cosine_3to5_cross_corpus",
        "threshold": threshold,
        "candidate_count": int((highest >= threshold).sum()),
        "highest_similarity": float(highest.max()),
        "note": "Approximate nearest-neighbor screen; cannot prove semantic independence.",
    }


def run_challenge(baseline_dir: Path, challenge_dir: Path,
                  output_dir: Path, *, seed: int = CHALLENGE_SEED):
    if seed != CHALLENGE_SEED:
        raise CorpusError("La semilla del desafío está congelada en 913.")
    old_splits, original, params, originals = load_frozen_baseline(baseline_dir)
    old_train = old_splits["train"]
    old_all = [r["text"] for group in old_splits.values() for r in group]
    new_rows, audit, new_manifest = load_challenge(
        challenge_dir, baseline_texts=old_all,
    )
    near = near_duplicate_audit(old_all, [r["text"] for r in new_rows])
    if near["candidate_count"] != 0:
        raise CorpusError("Revisión de casi duplicados pendiente; bloquear evaluación.")

    output = Path(output_dir)
    if not any(part in output.parts for part in ("artifacts", "reports")):
        raise CorpusError("Salida restringida a artifacts/reports privados/ignorados.")
    if output.exists() and any(output.iterdir()):
        raise CorpusError("No sobreescribir evaluación de desafío ya ejecutada.")

    # No feature/label/test text is passed to fitting or model selection.
    x = [r["text"] for r in old_train]
    y = [r["target_label"] for r in old_train]
    calib_rows = [{
        "text": r["text"], "sentiment": r["target_label"],
        "group_id": r["scenario_family"]
    } for r in old_train]
    folds = _training_calibration_folds(calib_rows, ORIGINAL_SEED)
    scored = [{
        "text": r["text"], "sentiment": r["intended_label"], "language": "es"
    } for r in new_rows]
    validation_rows = [{
        "text": r["text"], "sentiment": r["target_label"], "language": "es"
    } for r in old_splits["validation"]]

    results = {}
    models = {}
    for kind in MODEL_NAMES:
        model = _new_estimator(kind, params[kind], ORIGINAL_SEED)
        if kind == "svm":
            model.set_params(cv=folds)
        model.fit(x, y)  # baseline TRAIN ONLY, no challenge samples
        # Compare with archived validation values: reconstruct frozen setup,
        # not a new hyperparameter search or rerun of the original test.
        original_val = original["models"][kind]["validation"]["macro_f1"]
        reconstructed_val = _score(model, validation_rows)["macro_f1"]
        if abs(original_val - reconstructed_val) > 1e-10:
            raise CorpusError("Modelo reconstruido no reproduce la validación base.")
        challenge_metrics = _score(model, scored)
        guesses = list(model.predict([r["text"] for r in new_rows]))
        errors = [
            {"review_id": r["review_id"], "domain": r["domain"],
             "phenomenon": r["phenomenon"], "intended": r["intended_label"],
             "predicted": str(guess)}
            for r, guess in zip(new_rows, guesses)
            if r["intended_label"] != guess
        ]
        results[kind] = {
            "original_synthetic_test_macro_f1": original["models"][kind]["test"]["macro_f1"],
            "original_synthetic_test_accuracy": original["models"][kind]["test"]["accuracy"],
            "reconstructed_original_validation_macro_f1": reconstructed_val,
            "challenge": challenge_metrics,
            "errors": {
                "total": len(errors),
                "first_24_ids_only": errors[:24],
                "by_phenomenon": dict(sorted(Counter(e["phenomenon"] for e in errors).items())),
                "by_domain": dict(sorted(Counter(e["domain"] for e in errors).items())),
            },
            "fixed_parameters": params[kind],
        }
        models[kind] = model

    output.mkdir(parents=True, exist_ok=True)
    model_hashes = {}
    for kind, estimator in models.items():
        p = output / f"reconstructed_frozen_{kind}.joblib"
        joblib.dump(estimator, p)
        model_hashes[kind] = file_sha(p)
    packages = {}
    for p in ("scikit-learn", "numpy", "scipy", "joblib"):
        try:
            packages[p] = version(p)
        except PackageNotFoundError:
            packages[p] = "NOT_INSTALLED"
    report = {
        "report_type": "OUT_OF_GENERATOR_SYNTHETIC_CHALLENGE_ONLY",
        "not_real_tourism": True, "not_human_gold": True,
        "baseline_run": BASELINE_RUN,
        "baseline_reconstruction_caveat": (
            "Original 2026-11-05 CI artifact did not include model joblibs. "
            "Each model was reproduced once from frozen original TRAIN, fixed "
            "hyperparameters, original seed, matching original VALIDATION "
            "Macro-F1. Previously serialized weights were not literally loaded."
        ),
        "baseline_hashes": originals,
        "challenge_sha256": new_manifest["data_sha256"],
        "challenge_manifest_sha256": file_sha(Path(challenge_dir) / "challenge_manifest.json"),
        "challenge_seed": seed, "size": len(new_rows),
        "groups": audit["independent_nuclei"], "distribution": audit,
        "cross_corpus_similarity": near,
        "model_hashes": model_hashes,
        "models": results,
        "environment": {"python": platform.python_version(), "packages": packages},
        "limits": [
            "One-time synthetic challenge; do not reuse this test to tune generator.",
            "Synthetic target labels not independent human sentiment annotations.",
            "Same seven narrative frames reused in the challenge; 1260 rows comprise 180 nuclei.",
            "Original test and new challenge have different data distributions.",
            "Similarity checks cannot certify absence of semantic contamination.",
            "No real Spanish tourist validation or human agreement evidence.",
            "Model joblibs are for local trusted loading only.",
        ],
    }
    (output / "challenge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="One-time external-to-generator synthetic challenge")
    p.add_argument("--baseline", type=Path, default=Path("data/processed/frozen-original"))
    p.add_argument("--challenge", type=Path, default=Path("data/processed/out-of-generator-challenge-v1"))
    p.add_argument("--out", type=Path, default=Path("artifacts/out-of-generator-v1"))
    a = p.parse_args(argv)
    try:
        result = run_challenge(a.baseline, a.challenge, a.out)
    except (CorpusError, ValueError, OSError) as exc:
        p.exit(2, f"Challenge blocked: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "report_type": result["report_type"],
        "size": result["size"],
        "similarity": result["cross_corpus_similarity"],
        "metrics": {kind: {
            "macro_f1": item["challenge"]["macro_f1"],
            "accuracy": item["challenge"]["accuracy"],
            "confusion_matrix": item["challenge"]["confusion_matrix"],
            "f1_by_class": {
                label: item["challenge"]["classification_report"][label]["f1-score"]
                for label in ("negativo", "neutral", "positivo")
            },
            "error_count": item["errors"]["total"]
        } for kind, item in result["models"].items()},
        "warning": "Synthetic challenge, not tourism-world accuracy.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
