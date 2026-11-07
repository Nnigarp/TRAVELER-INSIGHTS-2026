"""Registered A/B/C research comparison, isolated from the DEMO and old tests."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np

from .corpus import CorpusError, file_sha
from .labels import LABELS
from .scientific_benchmark import _new_estimator, _training_calibration_folds
from .synthetic_diversity import (assemble_strategies, canonical_sha,
                                  ignored_target, read_prepared, write_json)
from .research_metrics import (bootstrap_indices, bootstrap_macro, family_confusions,
                               interval, review_mask, score_matrix, slice_metrics,
                               summarize_predictions)


def verify_registration(commit: str, manifest: dict) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CorpusError("Se exige un SHA inmutable de prerregistro.")
    for name, expected in (("synthetic_diversity_v2.json", manifest["catalog_sha256"]),
                           ("synthetic_v2_protocol.json", manifest["protocol_sha256"])):
        path = f"src/traveler_agent/research_data/{name}"
        raw = subprocess.check_output(["git", "show", f"{commit}:{path}"], stderr=subprocess.DEVNULL)
        import hashlib
        if hashlib.sha256(raw).hexdigest() != expected:
            raise CorpusError("Catálogo/protocolo no coincide con el commit de prerregistro.")


def select_models(training: list[dict], validation: list[dict], protocol: dict) -> tuple[dict, dict]:
    """This function never receives challenge rows or their intended labels."""
    if any(r["role"] != "train" for r in training) or any(r["role"] != "validation" for r in validation):
        raise CorpusError("Roles incorrectos en selección; no se ajusta sobre challenge.")
    train_texts = [r["text"] for r in training]
    train_labels = [r["intended_label"] for r in training]
    if set(train_labels) != set(LABELS):
        raise CorpusError("Entrenamiento requiere las tres intenciones canónicas.")
    cv_rows = [{"text": r["text"], "sentiment": r["intended_label"], "group_id": r["family"]}
               for r in training]
    folds = _training_calibration_folds(cv_rows, protocol["seed"])
    selected, reports = {}, {}
    for kind, grid in protocol["model_grid"].items():
        attempts = []
        for params in grid:
            estimator = _new_estimator(kind, params, protocol["seed"])
            if kind == "svm":
                estimator.set_params(cv=folds)
            start = time.perf_counter()
            estimator.fit(train_texts, train_labels)
            fit_seconds = time.perf_counter() - start
            scores = score_matrix(estimator, [r["text"] for r in validation])
            predictions = np.asarray(LABELS)[scores.argmax(axis=1)]
            metrics = summarize_predictions([r["intended_label"] for r in validation], predictions)
            attempts.append((metrics["macro_f1"], estimator,
                             {"parameters": params, "fit_seconds": fit_seconds, "validation": metrics}))
        winner = max(range(len(attempts)), key=lambda i: (attempts[i][0], -i))
        selected[kind] = attempts[winner][1]
        reports[kind] = {"selected_parameters": attempts[winner][2]["parameters"],
                         "validation": attempts[winner][2]["validation"],
                         "candidates": [item[2] for item in attempts]}
    return selected, reports


def latency(model, validation: list[dict], protocol: dict) -> dict:
    inputs = [r["text"] for r in sorted(validation, key=lambda r: r["review_id"])]
    for i in range(protocol["latency"]["warmup"]):
        model.predict_proba([inputs[i % len(inputs)]])
    times = []
    for _ in range(protocol["latency"]["passes"]):
        for text in inputs:
            start = time.perf_counter_ns()
            model.predict_proba([text])
            times.append((time.perf_counter_ns() - start) / 1e6)
    return {"scope": "new synthetic validation, batch 1, model loaded, includes vectorization",
            "sample_rows": len(inputs), "calls": len(times),
            "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def run(data: Path, output: Path, *, registration_commit: str, reproduction: bool = False) -> dict:
    rows, manifest, protocol = read_prepared(data)
    verify_registration(registration_commit, manifest)
    output = ignored_target(output)
    if output.exists() and any(output.iterdir()):
        raise CorpusError("Destino de resultados no vacío; nunca sobrescribir evidencia.")
    journal = Path(data) / "evaluation_journal.jsonl"
    if journal.exists() and not reproduction:
        raise CorpusError("El desafío ya fue abierto; usar --reproduction solo para repetición sin cambios.")
    strategies, diagnostic = assemble_strategies(rows, protocol)
    validation = [r for r in rows if r["role"] == "validation"]
    challenge = [r for r in rows if r["role"] == "challenge"]
    unlabeled = {role: [r for r in rows if r["role"] == role] for role in ("ambiguity", "illustrative")}
    output.mkdir(parents=True, exist_ok=True)
    purpose = "IDENTICAL_FROZEN_REPRODUCTION" if reproduction else "FIRST_REGISTERED_EVALUATION"
    event = {"started_utc": datetime.now(timezone.utc).isoformat(), "purpose": purpose,
             "registration_commit": registration_commit, "catalog_sha256": manifest["catalog_sha256"]}
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    models, results = {}, {}
    selections = {"purpose": "SELECTED_WITH_TRAIN_AND_NEW_VALIDATION_ONLY", "strategies": {}}
    for strategy, training in strategies.items():
        models[strategy], results[strategy] = select_models(training, validation, protocol)
        selections["strategies"][strategy] = {
            "training_rows": len(training),
            "training_families": len({r["family"] for r in training}),
            "training_sources": dict(Counter(r["source"] for r in training)),
            "training_ids_sha256": canonical_sha(sorted(r["review_id"] for r in training)),
            "training_labels": dict(Counter(r["intended_label"] for r in training)),
            "models": {kind: record["selected_parameters"] for kind, record in results[strategy].items()},
        }
    # The final selection is written and hashed BEFORE any challenge inference.
    write_json(output / "selection_manifest.json", selections)
    prediction_records, bootstrap_by_model = [], {}
    group_names = sorted({r["family"] for r in challenge})
    sampling = bootstrap_indices(len(group_names), protocol["bootstrap"]["resamples"], protocol["bootstrap"]["seed"])
    model_dir = output / "models"
    model_dir.mkdir()
    truth = [r["intended_label"] for r in challenge]
    texts = [r["text"] for r in challenge]
    for strategy, by_kind in models.items():
        for kind, model in by_kind.items():
            scores = score_matrix(model, texts)  # ONE challenge forward call per selected estimator.
            prediction = np.asarray(LABELS)[scores.argmax(axis=1)]
            reviewed = review_mask(scores, texts)
            record = results[strategy][kind]
            record["challenge"] = summarize_predictions(truth, prediction, reviewed=reviewed)
            record["slices"] = slice_metrics(challenge, truth, prediction, reviewed)
            groups, cubes = family_confusions(challenge, truth, prediction)
            if groups != group_names:
                raise CorpusError("Remuestreo de modelos no está pareado por familia.")
            key = f"{strategy}:{kind}"
            bootstrap_by_model[key] = bootstrap_macro(cubes, sampling)
            record["macro_f1_family_bootstrap"] = interval(bootstrap_by_model[key])
            record["latency"] = latency(model, validation, protocol)
            record["unlabeled_cohorts"] = {}
            for role, cohort in unlabeled.items():
                cohort_texts = [r["text"] for r in cohort]
                cohort_scores = score_matrix(model, cohort_texts)
                referral = review_mask(cohort_scores, cohort_texts)
                record["unlabeled_cohorts"][role] = {
                    "n": len(cohort), "reviewed": int(referral.sum()),
                    "review_rate": float(referral.mean()), "sentiment_accuracy": None,
                    "label_note": "No sentiment gold; review expectation is author intent only.",
                }
            record["challenge_errors"] = [r["review_id"] for r, gt, p in zip(challenge, truth, prediction) if gt != p]
            filename = model_dir / f"{strategy}-{kind}.joblib"
            joblib.dump(model, filename)
            record["model_sha256"] = file_sha(filename)
            for row, predicted, needs_review, values in zip(challenge, prediction, reviewed, scores):
                prediction_records.append({"strategy": strategy, "model": kind, "review_id": row["review_id"],
                                           "family": row["family"], "target_intent": row["intended_label"],
                                           "prediction": str(predicted), "needs_human_review": bool(needs_review),
                                           "scores_unverified_probability_reliability": values.tolist()})
    comparisons = {}
    for strategy in ("B_new", "C_mixed_matched"):
        for kind in ("lr", "svm"):
            left, right = f"{strategy}:{kind}", f"A_historical_matched:{kind}"
            delta = (results[strategy][kind]["challenge"]["macro_f1"]
                     - results["A_historical_matched"][kind]["challenge"]["macro_f1"])
            comparisons[f"{left}-minus-{right}"] = {
                "macro_f1_difference": delta,
                "paired_family_bootstrap": interval(bootstrap_by_model[left] - bootstrap_by_model[right]),
            }
    report = {
        "report_type": "SYNTHETIC_AUTHOR_INTENT_ONLY_NOT_HUMAN_VALIDATION",
        "purpose": purpose, "registration_commit": registration_commit,
        "catalog_sha256": manifest["catalog_sha256"], "protocol_sha256": manifest["protocol_sha256"],
        "records_sha256": manifest["records_sha256"],
        "selection_manifest_sha256": file_sha(output / "selection_manifest.json"),
        "challenge_rows": len(challenge), "challenge_families": len(group_names),
        "historical_TRAIN_diagnostic": diagnostic, "strategies": results,
        "matched_comparisons": comparisons,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib")}},
        "limits": [
            "All sentiment targets and phenomenon tags are assistant author intent, not independent human gold.",
            "Family IDs and lexical screens do not prove semantic independence; the same assistant authored all splits.",
            "Historical tests previously influenced hypotheses; they are not blind final tests and were not scored here.",
            "Historical A models are refitted using only original TRAIN and new development selection, not original saved weights.",
            "A/B/C matched runs use 90 rows; full historical TRAIN is a separate 204-row sensitivity comparison.",
            "Neutral texts remain mostly descriptive; residual discourse/style shortcuts may remain.",
            "Only 18 challenge families; slice supports and synthetic family intervals limit conclusions.",
            "DEMO policy is measured unchanged, not optimized on challenge; calibration reliability remains unverified.",
            "No authorized human corpus is admitted; D human/synthetic training remains pending.",
            "No BETO sentiment model is evaluated; the separate encoder smoke is not a classifier.",
        ],
    }
    with (output / "challenge_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in prediction_records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(output / "experiment_report.json", report)
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**event, "completed": True,
                                "report_sha256": file_sha(output / "experiment_report.json")}, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--reproduction", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.data, args.out, registration_commit=args.registration_commit,
                     reproduction=args.reproduction)
    except (CorpusError, ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Research benchmark stopped: {type(exc).__name__}.\n")
    print(json.dumps({"report_type": report["report_type"], "macro_f1": {
        strategy: {kind: metrics["challenge"]["macro_f1"] for kind, metrics in values.items()}
        for strategy, values in report["strategies"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
