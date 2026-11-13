"""Analyze frozen V3 saved predictions; never fit, select or read source texts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .corpus import CorpusError, file_sha
from .labels import LABELS
from .research_metrics import (
    bootstrap_indices, bootstrap_macro, family_confusions, interval,
    summarize_predictions,
)
from .synthetic_diversity import ignored_target, write_json


PRIMARY_SEED = 42
REPLICA_SEEDS = (7, 101)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
TARGET_TRAIN_ROWS = 3501
SCALE_DEVELOPMENT_CAP_SECONDS = 3600
ASSESSMENT_SCOPES = {
    "FRESH_SYNTHETIC_PILOT_ASSESSMENT",
    "REPRODUCTION_OF_OBSERVED_SYNTHETIC_PILOT_NOT_A_FRESH_TEST",
}


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"Resultado guardado ausente o ilegible: {path.name}.") from exc


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise CorpusError(f"Conteo guardado inválido: {name}.")
    return value


def _membership(manifest, protocol):
    rows = manifest.get("assessment_membership")
    if not isinstance(rows, list) or not rows:
        raise CorpusError("Falta la membresía assessment congelada en el manifiesto.")
    if any(not isinstance(row, dict) for row in rows):
        raise CorpusError("Membresía assessment inválida.")
    for row in rows:
        if (not isinstance(row.get("review_id"), str) or not row["review_id"]
                or not isinstance(row.get("family"), str) or not row["family"]
                or row.get("intended_label") not in LABELS):
            raise CorpusError("ID, familia o intención inválidos en membresía assessment.")
    if len({row["review_id"] for row in rows}) != len(rows):
        raise CorpusError("ID duplicado en la membresía assessment.")
    counts = manifest.get("counts", {}).get("assessment", {})
    original_n = _positive_int(counts.get("rows"), "assessment.rows")
    admitted_n = _positive_int(
        manifest.get("admitted_role_rows", {}).get("assessment", original_n),
        "admitted_role_rows.assessment",
    )
    if len(rows) != admitted_n or admitted_n > original_n:
        raise CorpusError("Membresía distinta del conteo assessment admitido.")
    family_labels = {}
    for row in rows:
        family_labels.setdefault(row["family"], []).append(row["intended_label"])
    if protocol.get("pilot_labeled_family_rows") != len(LABELS):
        raise CorpusError("Protocolo de familias V3 inesperado.")
    if any(sorted(labels) != sorted(LABELS) for labels in family_labels.values()):
        raise CorpusError("Familia assessment incompleta o sin las tres intenciones.")
    original_families = _positive_int(counts.get("families"), "assessment.families")
    if (len(family_labels) > original_families
            or (admitted_n == original_n and len(family_labels) != original_families)):
        raise CorpusError("Familias distintas del conteo assessment congelado.")
    quarantined = set(manifest.get("quarantined_families", []))
    if quarantined.intersection(family_labels):
        raise CorpusError("El assessment contiene familias en cuarentena.")
    support = dict(Counter(row["intended_label"] for row in rows))
    if admitted_n == original_n and counts.get("classes") != support:
        raise CorpusError("Etiquetas distintas del conteo assessment congelado.")
    return sorted(rows, key=lambda row: row["review_id"])


def _check_classification(reported, calculated):
    if not isinstance(reported, dict):
        raise CorpusError("Faltan métricas de clasificación guardadas.")
    exact = ("n", "support", "confusion_matrix_labels", "confusion_matrix",
             "negative_to_positive_count", "negative_to_neutral_count")
    numeric = ("macro_f1", "accuracy", "negative_to_positive_rate",
               "negative_to_neutral_rate")
    if any(reported.get(key) != calculated[key] for key in exact):
        raise CorpusError("Métricas guardadas no coinciden con las predicciones.")
    for key in numeric:
        value = reported.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(value, calculated[key], rel_tol=0, abs_tol=1e-12)):
            raise CorpusError(f"Métrica guardada inconsistente: {key}.")


def _predictions(saved, report, rows):
    results = report.get("results")
    entries = report.get("model_entries")
    if (not isinstance(results, dict) or not results
            or any(not isinstance(key, str) or not key for key in results)
            or not isinstance(saved, list) or not saved
            or not isinstance(entries, list)
            or any(not isinstance(entry, dict) for entry in entries)):
        raise CorpusError("Resultados o modelos guardados incompletos.")
    keys = [entry.get("key") for entry in entries]
    if (any(not isinstance(key, str) or not key for key in keys)
            or len(keys) != len(set(keys)) or set(keys) != set(results)):
        raise CorpusError("Modelos assessment distintos del inventario congelado.")
    models = {}
    expected = {row["review_id"]: row for row in rows}
    for record in saved:
        if not isinstance(record, dict):
            raise CorpusError("Predicción guardada inválida.")
        key, rid = record.get("model"), record.get("review_id")
        if (not isinstance(key, str) or key not in results
                or not isinstance(rid, str) or rid not in expected):
            raise CorpusError("Modelo o ID ajeno al assessment congelado.")
        if rid in models.setdefault(key, {}):
            raise CorpusError("Predicción duplicada para modelo e ID.")
        member = expected[rid]
        if (record.get("family") != member["family"]
                or record.get("intended_label") != member["intended_label"]
                or record.get("prediction") not in LABELS):
            raise CorpusError("Familia, intención o predicción desalineadas.")
        for field in ("probabilities", "calibrated_probabilities"):
            try:
                values = np.asarray(record.get(field), dtype=float)
            except (TypeError, ValueError) as exc:
                raise CorpusError("Probabilidades guardadas inválidas.") from exc
            if (values.shape != (len(LABELS),) or not np.isfinite(values).all()
                    or (values < 0).any()
                    or not np.isclose(values.sum(), 1., rtol=0, atol=1e-6)):
                raise CorpusError("Probabilidades guardadas inválidas.")
            if field == "probabilities" and LABELS[int(values.argmax())] != record["prediction"]:
                raise CorpusError("Predicción distinta del argmax guardado.")
        models[key][rid] = record
    if set(models) != set(results):
        raise CorpusError("Faltan predicciones de modelos evaluados.")
    truth = [row["intended_label"] for row in rows]
    ordered, metrics = {}, {}
    for key, records in models.items():
        if set(records) != set(expected):
            raise CorpusError("Predicciones assessment incompletas.")
        ordered[key] = [records[row["review_id"]]["prediction"] for row in rows]
        metrics[key] = summarize_predictions(truth, ordered[key])
        value = results[key]
        if not isinstance(value, dict):
            raise CorpusError("Métricas de modelo inválidas.")
        _check_classification(value.get("classification"), metrics[key])
    if "lr" not in models or "svm" not in models:
        raise CorpusError("Faltan los comparadores LR/SVM registrados.")
    return truth, ordered, metrics


def _seed_summary(report, metrics):
    primary = report.get("selected_model_key")
    if primary is not None and primary not in metrics:
        raise CorpusError("Falta el modelo ajustado seleccionado antes del assessment.")
    statuses = report.get("replica_status", [])
    if not isinstance(statuses, list) or any(not isinstance(row, dict) for row in statuses):
        raise CorpusError("Estados de réplicas inválidos.")
    by_seed = {}
    for row in statuses:
        seed = row.get("seed")
        if type(seed) is not int or seed not in REPLICA_SEEDS or seed in by_seed:
            raise CorpusError("Semilla de réplica no registrada o duplicada.")
        by_seed[seed] = row
    planned = [(PRIMARY_SEED, primary), *[(seed, f"beto_seed{seed}") for seed in REPLICA_SEEDS]]
    records = []
    for seed, key in planned:
        status = by_seed.get(seed, {}).get("status") if seed != PRIMARY_SEED else None
        available = key in metrics if key is not None else False
        if seed != PRIMARY_SEED and ((available and status != "EXECUTED")
                                    or (not available and status == "EXECUTED")):
            raise CorpusError("Estado de réplica inconsistente con resultados guardados.")
        if primary is None and available:
            raise CorpusError("Réplica sin configuración ajustada seleccionada.")
        missing_reason = (None if available else
                          "NO_ADJUSTED_CONFIGURATION_SELECTED" if primary is None else
                          status or "NOT_REQUESTED_OR_NOT_REPORTED")
        metric = metrics[key] if available else {}
        records.append({
            "seed": seed, "model_key": key,
            "status": "OBSERVED" if available else "MISSING",
            "reported_execution_status": status,
            "missing_reason": missing_reason,
            "macro_f1": metric.get("macro_f1"),
            "negative_to_positive_count": metric.get("negative_to_positive_count"),
            "negative_to_positive_rate": metric.get("negative_to_positive_rate"),
        })
    observed = [row for row in records if row["status"] == "OBSERVED"]
    scores = [row["macro_f1"] for row in observed]
    dangerous = [row["negative_to_positive_count"] for row in observed]
    return {
        "primary_seed": PRIMARY_SEED, "registered_replica_seeds": list(REPLICA_SEEDS),
        "seed_results": records,
        "observed_seeds": [row["seed"] for row in observed],
        "missing_seeds": [row["seed"] for row in records if row["status"] == "MISSING"],
        "all_fixed_seeds_observed": len(observed) == len(planned),
        "observed_seed_count": len(observed),
        "mean_macro_f1": float(np.mean(scores)) if scores else None,
        "sample_std_macro_f1": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
        "negative_to_positive": dangerous,
        "mean_negative_to_positive_count": float(np.mean(dangerous)) if dangerous else None,
        "sample_std_negative_to_positive_count": float(np.std(dangerous, ddof=1)) if len(dangerous) > 1 else None,
        "summary_scope": "Available fixed seeds only; missing seeds have no imputed score.",
        "best_seed_selection_performed": False, "model_promotion_performed": False,
    }


def _cost_projection(root, report, manifest, protocol):
    source = root / "development-selection.json"
    if not source.exists():
        return {"status": "NOT_AVAILABLE", "reason": "development-selection.json is absent",
                "measured_full_5000_run": False, "quality_inferred_from_assessment_score": False}
    selection = _read(source)
    if (not isinstance(selection, dict)
            or selection.get("schema") != "beto-v3-pilot-selection"
            or selection.get("records_sha256") != manifest["records_sha256"]
            or selection.get("selected_model_key") != report.get("selected_model_key")
            or any(selection.get(key) != report.get(key) for key in ("data_commit", "protocol_commit"))
            or report.get("selection_sha256") != file_sha(source)):
        raise CorpusError("Coste de desarrollo de otra selección o manifiesto.")
    cost = selection.get("cost", {})
    seconds = cost.get("wall_seconds") if isinstance(cost, dict) else None
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0):
        return {"status": "NOT_AVAILABLE", "reason": "No finite positive measured development cost",
                "measured_full_5000_run": False, "quality_inferred_from_assessment_score": False}
    admitted = manifest.get("admitted_role_rows", {})
    train_rows_source = ("admitted_role_rows.train" if "train" in admitted else "counts.train.rows")
    train_n = _positive_int(admitted.get("train", manifest.get("counts", {}).get("train", {}).get("rows")),
                            train_rows_source)
    target = protocol.get("target_5000_plan", {}).get("train")
    cap = protocol.get("compute_budget", {}).get("scale_development_wall_seconds_cap")
    if target != TARGET_TRAIN_ROWS or cap != SCALE_DEVELOPMENT_CAP_SECONDS:
        raise CorpusError("Objetivo o límite de escalado distintos del protocolo registrado.")
    projection = float(seconds * target / train_n)
    return {
        "status": "ROUGH_FIRST_ORDER_ROW_LINEAR_PROJECTION",
        "development_cost_source": "ARCHIVED_ORIGINAL_TUNE_SEARCH" if selection.get("recovery") else "CURRENT_DEVELOPMENT_SEARCH",
        "archived_selection_commit": selection.get("recovery", {}).get("original_selection_commit"),
        "separate_fixed_reconstruction_cost": selection.get("recovery", {}).get("cost"),
        "source_selection_sha256": file_sha(source),
        "measured_pilot_development_cost": cost,
        "admitted_train_rows": train_n, "train_rows_source": train_rows_source,
        "target_train_rows": target, "row_scale_factor": target / train_n,
        "rough_projected_development_wall_seconds": projection,
        "registered_scale_development_wall_seconds_cap": cap,
        "rough_projection_within_registered_cap": projection <= cap,
        "rough_projection_to_cap_ratio": projection / cap,
        "formula": "measured pilot development wall seconds * 3501 / admitted TRAIN rows",
        "measured_full_5000_run": False, "quality_inferred_from_assessment_score": False,
        "limits": [
            "Rough first-order planning estimate, not a measurement of a full 5000-row run.",
            "Development includes fixed overhead, tuning and any executed optional references; row-linear scaling may fail.",
            "Candidate skips, early stopping, hardware, memory and non-TRAIN cohort sizes can change cost.",
            "Cost comparison alone neither admits corpus quality nor authorizes scale-up.",
        ],
    }


def analyze(root):
    """Validate and summarize a completed saved run without source-corpus access."""
    root = Path(root)
    paths = {name: root / name for name in (
        "assessment.json", "assessment-predictions.json", "manifest.json", "protocol.json",
    )}
    report, saved, manifest, protocol = [_read(path) for path in paths.values()]
    if (not isinstance(report, dict) or not isinstance(manifest, dict) or not isinstance(protocol, dict)
            or report.get("schema") != "beto-v3-pilot-assessment"
            or report.get("status") not in ("COMPLETED_WITHIN_REGISTERED_BUDGET", "COMPLETED_WITH_DISCLOSED_FIXED_INFERENCE_CONTINUATION")
            or report.get("scope") not in ASSESSMENT_SCOPES
            or manifest.get("schema") != "beto-v3-pilot-manifest"
            or manifest.get("human_gold") is not False
            or report.get("human_performance_measured") is not False
            or not isinstance(manifest.get("records_sha256"), str)
            or report.get("data_manifest") != manifest):
        raise CorpusError("Informe V3 completo y manifiesto sintético coincidente requeridos.")
    if report['status']=="COMPLETED_WITH_DISCLOSED_FIXED_INFERENCE_CONTINUATION":
        from .beto_training_v3_continuation import validate_ledger
        validate_ledger(root,report,saved)
    if (protocol.get("primary_seed") != PRIMARY_SEED
            or protocol.get("replica_seeds") != list(REPLICA_SEEDS)):
        raise CorpusError("Semillas distintas del protocolo V3 registrado.")
    # protocol_sha256 identifies the original registered resource bytes. The
    # runner serializes its saved snapshot with sorted keys, so its file hash
    # need not equal that resource hash. Record the snapshot hash separately.
    rows = _membership(manifest, protocol)
    truth, models, metrics = _predictions(saved, report, rows)
    samples, paired = {}, {}
    family_names = sorted({row["family"] for row in rows})
    indices = bootstrap_indices(len(family_names), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    for key, predictions in models.items():
        names, cubes = family_confusions(rows, truth, predictions)
        if names != family_names:
            raise CorpusError("Familias de remuestreo no alineadas.")
        samples[key] = bootstrap_macro(cubes, indices)
        paired[key] = {"families": len(names), "classification": metrics[key],
                       "macro_f1_family_bootstrap": interval(samples[key])}
    for key in paired:
        paired[key]["difference_from_lr"] = interval(samples[key] - samples["lr"])
        paired[key]["point_macro_f1_difference_from_lr"] = metrics[key]["macro_f1"] - metrics["lr"]["macro_f1"]
    return {
        "schema": "beto-v3-saved-prediction-analysis",
        "scope": "SAVED_NEW_SYNTHETIC_ASSESSMENT_ONLY_NO_FITTING_OR_SELECTION",
        "assessment_scope": report["scope"], "synthetic": True, "human_gold": False,
        "records_sha256": manifest["records_sha256"],
        "source_sha256": {name: file_sha(path) for name, path in paths.items()},
        "membership_validated_against": "frozen manifest.assessment_membership",
        "assessment_rows": len(rows), "assessment_families": len(family_names),
        "bootstrap": {"unit": "synthetic semantic family", "resamples": BOOTSTRAP_RESAMPLES,
                      "seed": BOOTSTRAP_SEED, "paired_reference": "lr",
                      "same_family_indices_for_all_models": True},
        "selected_model_key": report.get("selected_model_key"),
        "paired_family_bootstrap": paired,
        "selected_configuration_seed_summary": _seed_summary(report, metrics),
        "cost_projection": _cost_projection(root, report, manifest, protocol),
        "best_model_selection_performed": False,
        "limits": [
            "AI author intent and AI review are synthetic labels, not human gold or human validation.",
            "Family bootstrap intervals describe this synthetic cohort, not a human population.",
            "The fixed primary seed remains 42; no assessment score chooses a model or seed or promotes it to the UI.",
            "At most three fixed seeds share this evaluation; they do not establish general optimizer stability.",
            "This cohort is already observed; it cannot serve again as a fresh full-scale final test.",
            "Optional V1/V2 recipe references are retrained on NEW TRAIN, not original historical model binaries; no across-cohort quality claim is made.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = ignored_target(args.run, kinds=("artifacts",))
    output = ignored_target(args.out, kinds=("artifacts",))
    result = analyze(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)  # Exclusive creation preserves frozen source reports.


if __name__ == "__main__":
    main()
