"""Train Dummy/LR/SVM on GROUPED synthetic data only.

Never claim human labels or external validity. Uses existing model pipelines,
model grid, group-disjoint SVM calibration and metrics without modifications.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib

from .corpus import CorpusError, file_sha, text_sha
from .labels import LABELS
from .scientific_benchmark import MODEL_GRID, _new_estimator, _score, _training_calibration_folds
from .synthetic_tourism import (
    ASPECT_TEXT, CATEGORIES, FIELDS, GENERATOR, MODEL, VERSION, audit_records,
)

NAMES = ("train", "validation", "test")


def _read_input(input_dir: Path) -> tuple[list[dict], dict, dict]:
    root = Path(input_dir)
    if "processed" not in root.parts and "private" not in root.parts:
        raise CorpusError("Entrada debe proceder de data/processed o data/private.")
    path, info_path = root / "synthetic_reviews.csv", root / "synthetic_manifest.json"
    if not path.is_file() or not info_path.is_file():
        raise CorpusError("Faltan archivos sintéticos originales y manifiesto.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if (info.get("source_type") != "SYNTHETIC_GENERATED"
            or info.get("not_human_testimonials") is not True
            or info.get("method") != GENERATOR
            or info.get("model") != MODEL
            or info.get("version") != VERSION
            or info.get("rows_sha256") != file_sha(path)):
        raise CorpusError("Procedencia sintética o hash no verificable.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or tuple(reader.fieldnames) != FIELDS:
            raise CorpusError("Esquema del generador diferente del esperado.")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise CorpusError("Registro malformado.")
    audit = audit_records(rows)
    if audit["records"] != info.get("audit", {}).get("records"):
        raise CorpusError("Manifiesto de control de calidad inconsistente.")
    if any(row["seed"] != str(info["seed"]) for row in rows):
        raise CorpusError("Semillas mezcladas en archivo.")
    return rows, info, audit


def group_split(rows: list[dict], *, seed: int) -> dict[str, list[dict]]:
    """Deterministic 60/20/20 FAMILY split, never splitting paraphrases."""
    families = sorted({row["scenario_family"] for row in rows},
                      key=lambda fam: hashlib.sha256(
                          f"{seed}:{fam}".encode("utf-8")).hexdigest())
    if len(families) < 15:
        raise CorpusError("Familias insuficientes.")
    n = len(families)
    cut1 = round(n * .60)
    cut2 = cut1 + round(n * .20)
    ownership = {
        fam: "train" if i < cut1 else ("validation" if i < cut2 else "test")
        for i, fam in enumerate(families)
    }
    splits = {name: [] for name in NAMES}
    for row in rows:
        splits[ownership[row["scenario_family"]]].append(row)
    if any(set(row["target_label"] for row in splits[name]) != set(LABELS)
           for name in NAMES):
        raise CorpusError("Alguna partición no contiene las tres clases.")
    keys = [{row["text_sha256"] for row in splits[name]} for name in NAMES]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise CorpusError("Fuga de duplicados exactos entre particiones.")
    if len(set(ownership.values())) != 3:
        raise CorpusError("Particiones insuficientes.")
    return splits


def _convert(rows: list[dict]) -> list[dict]:
    return [{
        "text": r["text"], "sentiment": r["target_label"], "language": "es",
        "group_id": r["scenario_family"], "review_id": r["review_id"],
    } for r in rows]



def _synthetic_shortcut_audit(train: list[dict], test: list[dict]) -> dict:
    """Measure reused authored sentiment phrases across supposedly held-out families.

    This is a known vulnerability of compositional synthetic generators: a
    family-disjoint split does not imply language-pattern independence.
    """
    source_cues = [
        (aspect, label, phrase)
        for aspect, sentiments in ASPECT_TEXT.items()
        for label, phrases in sentiments.items()
        for phrase in phrases
    ]
    train_texts = [r["text"].casefold() for r in train]
    test_texts = [r["text"].casefold() for r in test]
    shared = [
        {"aspect": aspect, "intended_class": label}
        for aspect, label, phrase in source_cues
        if any(phrase.casefold() in t for t in train_texts)
        and any(phrase.casefold() in t for t in test_texts)
    ]
    return {
        "shared_exact_authored_class_cues": len(shared),
        "examples_aspect_and_class_only": shared[:12],
        "risk": ("HIGH_TEMPLATE_SHORTCUT_RISK" if shared else "NOT_DETECTED_BY_THIS_HEURISTIC"),
        "interpretation": (
            "Shared authored sentiment phrases may let models memorize synthetic "
            "lexical patterns across families; 100% test performance is not "
            "evidence of semantic generalization."
        ),
    }

def _report_errors(model, rows: list[dict]) -> dict:
    predicted = model.predict([row["text"] for row in rows])
    mismatch = [
        {"review_id": r["review_id"], "target": r["target_label"],
         "predicted": str(p), "category": r["category"], "hard_case": r["hard_case"]}
        for r, p in zip(rows, predicted) if p != r["target_label"]
    ]
    return {"count": len(mismatch), "examples_by_id_no_private_text": mismatch[:20],
            "by_hard_case": dict(sorted(Counter(r["hard_case"] for r, p in zip(rows, predicted)
                                                if p != r["target_label"]).items()))}


def run_synthetic_experiment(input_dir: Path, out_dir: Path, *, seed: int = 42) -> dict:
    raw, manifest, audit = _read_input(input_dir)
    if seed != manifest["seed"]:
        raise CorpusError("Semilla distinta de la usada al generar el corpus.")
    splits = group_split(raw, seed=seed)

    # Pairwise candidate check is an AUDIT, not proof of semantic independence.
    owner = {row["review_id"]: name for name, rows in splits.items() for row in rows}
    suspects = [
        pair for pair in audit["near_duplicate_candidates_cross_family"]
        if owner[pair["review_ids"][0]] != owner[pair["review_ids"][1]]
    ]
    if suspects:
        raise CorpusError(
            "Posibles casi duplicados entre particiones; revisar familias antes de evaluar."
        )

    output = Path(out_dir)
    if "artifacts" not in output.parts and "reports" not in output.parts:
        raise CorpusError("Informes/modelos deben estar en artifacts o reports, ignorados por Git.")
    if output.exists() and any(output.iterdir()):
        raise CorpusError("La salida ya contiene archivos. No sobrescribir test o resultados.")
    train, validation, test = [splits[k] for k in NAMES]
    cv_rows = _convert(train)
    train_counts = Counter(row["target_label"] for row in train)
    if min(train_counts.values()) < 3:
        raise CorpusError("Insuficientes registros por clase para SVM.")

    # The SVM calibration gets GROUP-DISJOINT folds within training only.
    folds = _training_calibration_folds(cv_rows, seed)
    compared = {}
    selected = {}
    for kind, params_grid in MODEL_GRID.items():
        candidates = []
        for params in params_grid:
            estimator = _new_estimator(kind, params, seed)
            if kind == "svm":
                estimator.set_params(cv=folds)
            estimator.fit([row["text"] for row in train],
                          [row["target_label"] for row in train])
            val_metrics = _score(estimator, _convert(validation))
            candidates.append((val_metrics["macro_f1"], params, estimator, val_metrics))
        # Fixed grid-order tie break. Test has NOT been inspected.
        best = max(enumerate(candidates), key=lambda pair: (pair[1][0], -pair[0]))[1]
        _, params, estimator, val_metrics = best
        selected[kind] = estimator
        compared[kind] = {
            "parameters": params,
            "validation": val_metrics,
            "validation_grid": [{"parameters": candidate[1], "macro_f1": candidate[0]}
                                for candidate in candidates],
        }

    # Exactly one held-out test evaluation for each final chosen estimator.
    for kind, estimator in selected.items():
        compared[kind]["test"] = _score(estimator, _convert(test))
        compared[kind]["test_error_audit"] = _report_errors(estimator, test)

    output.mkdir(parents=True, exist_ok=True)
    model_hashes = {}
    for kind, estimator in selected.items():
        filename = output / (kind + ".joblib")
        joblib.dump(estimator, filename)
        model_hashes[kind] = file_sha(filename)

    packages = {}
    for package in ("scikit-learn", "numpy", "scipy", "joblib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"

    report = {
        "report_type": "SYNTHETIC_INTERNAL_EVALUATION_ONLY",
        "research_claim": "NONE_NO_REAL_TOURIST_GENERALIZATION",
        "not_human_testimonials": True,
        "label_provenance": "GENERATOR_INTENT_NOT_HUMAN_GOLD",
        "generator": {"method": GENERATOR, "model": MODEL, "version": VERSION,
                      "seed": seed},
        "input_sha256": manifest["rows_sha256"],
        "manifest_sha256": file_sha(Path(input_dir) / "synthetic_manifest.json"),
        "strategy": "family_disjoint_hash_split_60_20_20_no_refit_on_test",
        "grouping_unit": "category_plus_aspect_generation_scenario",
        "counts": {
            name: {
                "rows": len(rows),
                "families": len({row["scenario_family"] for row in rows}),
                "labels": dict(sorted(Counter(row["target_label"] for row in rows).items())),
            } for name, rows in splits.items()
        },
        "quality": {
            "unique_text_ratio": audit["unique_text_ratio"],
            "unique_token_count": audit["unique_token_count"],
            "near_duplicate_candidates_cross_family": len(
                audit["near_duplicate_candidates_cross_family"]
            ),
            "hard_cases": audit["hard_cases"],
        },
        "lexical_shortcut_audit": _synthetic_shortcut_audit(train, test),
        "models": compared, "model_hashes": model_hashes,
        "environment": {"python": platform.python_version(), "packages": packages},
        "warnings": [
            "Targets express generator intent and are NOT independent human labels.",
            "Every example is synthetic: no tourism-world accuracy is measured.",
            "Shared authored class cues across groups can inflate synthetic accuracy to 100%.",
            "Template and scenario leakage may persist semantically despite family isolation.",
            "No external independent Spanish tourist test is used.",
            "Test was evaluated once per model AFTER validation tuning.",
            "Calibration probability reliability is not externally evaluated.",
            "Never use the synthetic test to tune prompts/generator and claim unbiased performance.",
            "Joblib files are executable when loaded; trust only your own local artifacts.",
        ],
    }
    (output / "synthetic_experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Dummy/LR/SVM: synthetic internal research only")
    p.add_argument("--input", type=Path, default=Path("data/processed/synthetic-tourism-v1"))
    p.add_argument("--out", type=Path, default=Path("artifacts/synthetic-tourism-v1"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    try:
        report = run_synthetic_experiment(args.input, args.out, seed=args.seed)
    except (CorpusError, ValueError, OSError) as exc:
        p.exit(2, f"Synthetic benchmark blocked: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "type": report["report_type"], "seed": args.seed, "counts": report["counts"],
        "models": {
            kind: {
                "validation_macro_f1": item["validation"]["macro_f1"],
                "test_macro_f1": item["test"]["macro_f1"],
                "test_accuracy": item["test"]["accuracy"],
                "test_confusion_matrix": item["test"]["confusion_matrix"],
                "test_classification_report": item["test"]["classification_report"],
                "test_error_audit": item["test_error_audit"],
            } for kind, item in report["models"].items()
        },
        "warning": "Evaluation of generator intent on fictional reviews, NOT real tourism evidence.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
