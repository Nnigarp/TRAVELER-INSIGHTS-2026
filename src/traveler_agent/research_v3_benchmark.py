"""Registered quantity/diversity factorial and separate selective-risk experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import joblib
import numpy as np

from .corpus import CorpusError, file_sha
from .labels import LABELS
from .synthetic_diversity import canonical_sha, ignored_target, write_json
from .synthetic_diversity_benchmark import select_models, latency
from .research_metrics import (score_matrix, review_mask, summarize_predictions, slice_metrics,
                               family_confusions, bootstrap_indices, bootstrap_macro, interval)
from .research_v3_data import (DOMAINS, ASPECTS, read_prepared, training_arms,
                               ablation_rows, shortcut_audit)
from .research_v3_selective import (fit_temperature, temperature_transform, probability_metrics,
                                    select_policy, policy_review, fixed_threshold_curve)

SOURCE_FILES = (
    "src/traveler_agent/research_v3_data.py", "src/traveler_agent/research_v3_selective.py",
    "src/traveler_agent/research_v3_benchmark.py", "src/traveler_agent/model.py",
    "src/traveler_agent/scientific_benchmark.py", "src/traveler_agent/research_metrics.py",
    "src/traveler_agent/synthetic_diversity.py", "src/traveler_agent/synthetic_diversity_benchmark.py",
    "src/traveler_agent/research_data/synthetic_diversity_v2.json",
)


def verify_registration(commit: str, manifest: dict) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CorpusError("Se requiere commit inmutable de prerregistro V3.")
    hashes = {}
    for path in (*SOURCE_FILES, "src/traveler_agent/research_data/synthetic_diversity_v3.json",
                 "src/traveler_agent/research_data/synthetic_v3_protocol.json"):
        raw = subprocess.check_output(["git", "show", f"{commit}:{path}"], stderr=subprocess.DEVNULL)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != file_sha(Path(path)):
            raise CorpusError("Código/catálogo/protocolo difiere del prerregistro; no abrir evaluación.")
        hashes[path] = digest
    if hashes["src/traveler_agent/research_data/synthetic_diversity_v3.json"] != manifest["catalog_sha256"] or hashes["src/traveler_agent/research_data/synthetic_v3_protocol.json"] != manifest["protocol_sha256"]:
        raise CorpusError("El manifiesto no corresponde al registro.")
    return hashes


def development_rows(rows):
    """Adapter to reuse V2 selection without changing its frozen code."""
    if not rows or any(r["role"] != "tune" for r in rows):
        raise CorpusError("Solo tune puede adaptarse para selección de C.")
    return [{**r, "role": "validation"} for r in rows]


def dimension_slices(rows, truth, predicted, reviewed) -> dict:
    result = slice_metrics(rows, truth, predicted, reviewed)
    truth, predicted, reviewed = np.asarray(truth), np.asarray(predicted), np.asarray(reviewed)
    for field in ("domain", "aspect"):
        for value in sorted({r[field] for r in rows}):
            indices = [i for i, row in enumerate(rows) if row[field] == value]
            result[f"{field}:{value}"] = summarize_predictions(truth[indices], predicted[indices], reviewed=reviewed[indices])
    return result


def attribution_on_development(rows, tune, protocol, broad_report):
    """Matched-size leave-one-dimension-out contrasts; never receives assessment."""
    if any(r["role"] != "tune" for r in tune) or any(r["role"] != "train" for r in rows):
        raise CorpusError("Ablaciones solo se puntúan en tune.")
    report = {}
    grid = {**protocol, "model_grid": {key: protocol["model_grid"][key] for key in ("lr", "svm")}}
    for field, values in (("domain", DOMAINS), ("aspect", ASPECTS)):
        for omitted in values:
            training = ablation_rows(rows, field, omitted, protocol["ablation"]["training_budget"])
            models, scores = select_models(training, development_rows(tune), grid)
            for kind, estimator in models.items():
                affected = [r for r in tune if r[field] == omitted]
                probabilities = score_matrix(estimator, [r["text"] for r in affected])
                predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
                scores[kind]["omitted_dimension_tune"] = summarize_predictions([r["intended_label"] for r in affected], predicted)
                scores[kind]["overall_tune_delta_vs_broad_120"] = scores[kind]["validation"]["macro_f1"] - broad_report[kind]["validation"]["macro_f1"]
                scores[kind]["omitted_dimension_tune_delta_vs_full"] = (
                    scores[kind]["omitted_dimension_tune"]["macro_f1"]
                    - broad_report[kind]["tune_slices"][f"{field}:{omitted}"]["macro_f1"])
            report[f"{field}:{omitted}"] = {
                "training_rows": len(training), "training_families": len({r["family"] for r in training}),
                "sources": dict(Counter(r["source"] for r in training)),
                "training_ids_sha256": canonical_sha(sorted(r["review_id"] for r in training)),
                "models": scores,
                "scope": "TUNE ONLY, replenished related TRAIN variants; not isolated causal or human contribution",
            }
    return report


def policy_overhead(scores, temperature, specification):
    times = []
    for row in scores:
        start = time.perf_counter_ns()
        policy_review(temperature_transform([row], temperature), specification)
        times.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "scope": "temperature transform and selected policy only; excludes estimator"}


def run(data: Path, output: Path, registration_commit: str, *, reproduction=False) -> dict:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    rows, manifest, protocol = read_prepared(data)
    source_hashes = verify_registration(registration_commit, manifest)
    output = ignored_target(output)
    if output.exists() and any(output.iterdir()):
        raise CorpusError("No sobrescribir resultados experimentales.")
    journal = Path(data) / "v3_evaluation_journal.jsonl"
    if journal.exists() and not reproduction:
        raise CorpusError("Evaluación ya abierta; solo --reproduction de especificación idéntica.")
    roles = {role: [r for r in rows if r["role"] == role] for role in protocol["role_rows"]}
    tune, calibration, policy_dev = roles["tune"], roles["calibration"], roles["policy_dev"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    purpose = "IDENTICAL_FROZEN_REPRODUCTION" if reproduction else "FIRST_REGISTERED_SYNTHETIC_ASSESSMENT"
    event = {"utc": datetime.now(timezone.utc).isoformat(), "purpose": purpose,
             "registration_commit": registration_commit, "catalog_sha256": manifest["catalog_sha256"]}
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    arms = training_arms(rows)
    development = {"purpose": "TRAIN_TUNE_CALIBRATION_POLICY_DEV_ONLY", "arms": {}, "train_shortcut_audits": {}}
    estimators, selections = {}, {"registration_commit": registration_commit, "arms": {}}
    for arm, training in arms.items():
        development["train_shortcut_audits"][arm] = shortcut_audit(training)
        estimators[arm], records = select_models(training, development_rows(tune), protocol)
        development["arms"][arm] = records
        selection = {"training_rows": len(training), "training_families": len({r["family"] for r in training}),
                     "training_sources": dict(Counter(r["source"] for r in training)),
                     "training_class_counts": dict(Counter(r["intended_label"] for r in training)),
                     "training_ids_sha256": canonical_sha(sorted(r["review_id"] for r in training)), "models": {}}
        for kind, model in estimators[arm].items():
            start = time.perf_counter()
            cal_scores = score_matrix(model, [r["text"] for r in calibration])
            temperature = fit_temperature(calibration, cal_scores, protocol)
            chosen_t = temperature["temperature"]
            dev_scores = score_matrix(model, [r["text"] for r in policy_dev])
            amb_scores = score_matrix(model, [r["text"] for r in roles["ambiguity_policy"]])
            adjusted = temperature_transform(dev_scores, chosen_t)
            policy = select_policy(policy_dev, adjusted, roles["ambiguity_policy"],
                                   temperature_transform(amb_scores, chosen_t), protocol)
            record = records[kind]
            tune_scores = score_matrix(model, [r["text"] for r in tune])
            record["tune_slices"] = dimension_slices(tune, [r["intended_label"] for r in tune],
                np.asarray(LABELS)[tune_scores.argmax(axis=1)], review_mask(tune_scores, [r["text"] for r in tune]))
            record["temperature_fit"] = temperature
            record["review_policy_fit"] = policy
            record["calibration_and_policy_seconds"] = time.perf_counter() - start
            record["independent_policy_dev_probability_metrics"] = {
                "raw": probability_metrics(policy_dev, dev_scores), "temperature": probability_metrics(policy_dev, adjusted)}
            record["latency"] = {**latency(model, policy_dev, protocol), "scope": "V3 policy_dev, batch 1, loaded estimator including vectorization"}
            record["policy_overhead"] = policy_overhead(dev_scores, chosen_t, policy["selected"])
            filename = output / "models" / f"{arm}-{kind}.joblib"
            joblib.dump({"research_only": True, "estimator": model, "temperature": chosen_t,
                         "review_policy": policy["selected"], "catalog_sha256": manifest["catalog_sha256"]}, filename)
            record["artifact_bytes"] = filename.stat().st_size
            record["artifact_sha256"] = file_sha(filename)
            selection["models"][kind] = {"parameters": record["selected_parameters"], "temperature": chosen_t,
                                          "review_policy": policy["selected"], "policy_status": policy["status"]}
        selections["arms"][arm] = selection
    development["dimension_ablations"] = attribution_on_development(roles["train"], tune, protocol, development["arms"]["broad_120"])
    selections["development_ids"] = {role: canonical_sha(sorted(r["review_id"] for r in roles[role]))
                                     for role in ("tune", "calibration", "policy_dev", "ambiguity_policy")}
    # All C values, temperatures, policies and diagnostic ablations are fixed here.
    write_json(output / "selection_manifest.json", selections)
    write_json(output / "development_report.json", development)
    assessment, ambiguity = roles["assessment"], roles["ambiguity_assessment"]
    truth, texts = [r["intended_label"] for r in assessment], [r["text"] for r in assessment]
    group_names = sorted({r["family"] for r in assessment})
    sampling = bootstrap_indices(len(group_names), protocol["bootstrap"]["resamples"], protocol["bootstrap"]["seed"])
    results, bootstrap_samples, predictions = {}, {}, []
    for arm, models in estimators.items():
        results[arm] = {}
        for kind, model in models.items():
            chosen = selections["arms"][arm]["models"][kind]
            scores = score_matrix(model, texts)  # one assessment call per selected estimator
            adjusted = temperature_transform(scores, chosen["temperature"])
            guessed = np.asarray(LABELS)[scores.argmax(axis=1)]
            masks = {"DEMO_RAW_UNCHANGED": review_mask(scores, texts),
                     "DEMO_TEMPERATURE": review_mask(adjusted, texts),
                     "SELECTED_ASYMMETRIC_TEMPERATURE": policy_review(adjusted, chosen["review_policy"])}
            amb_raw = score_matrix(model, [r["text"] for r in ambiguity])
            amb_adjusted = temperature_transform(amb_raw, chosen["temperature"])
            amb_masks = {"DEMO_RAW_UNCHANGED": review_mask(amb_raw, [r["text"] for r in ambiguity]),
                         "DEMO_TEMPERATURE": review_mask(amb_adjusted, [r["text"] for r in ambiguity]),
                         "SELECTED_ASYMMETRIC_TEMPERATURE": policy_review(amb_adjusted, chosen["review_policy"])}
            policies = {}
            for name, mask in masks.items():
                metrics = summarize_predictions(truth, guessed, reviewed=mask)["abstention"]
                dangerous_ids = [r["review_id"] for r, guess, referred in zip(assessment, guessed, mask)
                                 if r["intended_label"] == "negativo" and guess == "positivo" and not referred]
                policies[name] = {**metrics, "automatic_negative_to_positive_ids": dangerous_ids,
                                  "ambiguity": {"n": len(ambiguity), "reviewed": int(amb_masks[name].sum()),
                                                "review_rate": float(amb_masks[name].mean()), "sentiment_accuracy": None,
                                                "review_expectation": "author intent, not human gold"}}
            groups, cubes = family_confusions(assessment, truth, guessed)
            if groups != group_names:
                raise CorpusError("Familias no alineadas para comparación pareada.")
            sample = bootstrap_macro(cubes, sampling)
            bootstrap_samples[f"{arm}:{kind}"] = sample
            errors = [{"review_id": r["review_id"], "family": r["family"], "intended_label": r["intended_label"],
                       "prediction": str(guess), "phenomena": r["phenomena"]}
                      for r, guess in zip(assessment, guessed) if r["intended_label"] != guess]
            results[arm][kind] = {
                "classification": summarize_predictions(truth, guessed), "family_bootstrap_macro_f1": interval(sample),
                "slices": dimension_slices(assessment, truth, guessed, masks["SELECTED_ASYMMETRIC_TEMPERATURE"]),
                "probability_metrics": {"raw": probability_metrics(assessment, scores),
                                        "temperature": probability_metrics(assessment, adjusted)},
                "policies": policies, "fixed_confidence_curve_descriptive_only": fixed_threshold_curve(assessment, adjusted, protocol["review_policy"]["global_thresholds"]),
                "errors": errors, "latency": development["arms"][arm][kind]["latency"],
                "artifact_bytes": development["arms"][arm][kind]["artifact_bytes"],
            }
            for i, row in enumerate(assessment):
                predictions.append({"arm": arm, "model": kind, "review_id": row["review_id"], "family": row["family"],
                                    "target_author_intent": row["intended_label"], "prediction": str(guessed[i]),
                                    "raw_scores": scores[i].tolist(), "temperature_scores": adjusted[i].tolist(),
                                    "reviewed": {name: bool(mask[i]) for name, mask in masks.items()}})
    contrasts = {}
    pairs = {"quantity_narrow": ("narrow_120", "narrow_60"), "quantity_broad": ("broad_120", "broad_60"),
             "diversity_at_60": ("broad_60", "narrow_60"), "diversity_at_120": ("broad_120", "narrow_120")}
    for name, (left, right) in pairs.items():
        for kind in ("lr", "svm"):
            contrasts[f"{name}:{kind}"] = {
                "macro_f1_delta": results[left][kind]["classification"]["macro_f1"] - results[right][kind]["classification"]["macro_f1"],
                "paired_family_bootstrap": interval(bootstrap_samples[f"{left}:{kind}"] - bootstrap_samples[f"{right}:{kind}"])}
    for kind in ("lr", "svm"):
        values = (bootstrap_samples[f"broad_120:{kind}"] - bootstrap_samples[f"broad_60:{kind}"]
                  - bootstrap_samples[f"narrow_120:{kind}"] + bootstrap_samples[f"narrow_60:{kind}"])
        contrasts[f"quantity_diversity_interaction:{kind}"] = {"paired_family_bootstrap": interval(values)}
    with (output / "assessment_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    peak = None
    if platform.system() == "Linux":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report = {"report_type": "V3_SYNTHETIC_QUANTITY_DIVERSITY_AND_SELECTIVE_RISK_NOT_HUMAN_VALIDATION",
              "purpose": purpose, "registration_commit": registration_commit,
              "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_hashes": source_hashes,
              **{key: manifest[key] for key in ("catalog_sha256", "protocol_sha256", "records_sha256")},
              "selection_manifest_sha256": file_sha(output / "selection_manifest.json"),
              "development_report_sha256": file_sha(output / "development_report.json"),
              "assessment_rows": len(assessment), "assessment_families": len(group_names),
              "results": results, "factorial_contrasts": contrasts,
              "cost": {"wall_seconds": time.perf_counter() - wall_start, "cpu_seconds": time.process_time() - cpu_start,
                       "peak_process_rss_mib": peak, "rss_method": "Linux getrusage high-water" if peak else None,
                       "logical_cpus_visible": os.cpu_count()},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {name: version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib", "pandas", "threadpoolctl")}},
              "limits": ["All labels, semantic groups and phenomena are assistant author intent; no independent human gold.",
                         "Same assistant authored all batches; family IDs and lexical screens cannot prove semantic or external independence.",
                         "Factorial compares related-variant versus broader-family regimes at fixed budgets, not all possible notions of diversity.",
                         "One catalog, one training-pool choice and only 15 assessment families; uncertainty and style confounding remain.",
                         "Domain/aspect ablations are development-only and replenish variants; not isolated causal contributions.",
                         "Policy eligibility is empirical on 30 synthetic cases and 10 author-ambiguous cases, not a risk guarantee.",
                         "No human generalization, DEMO replacement, V2 re-scoring, BETO sentiment training or GPU benchmark.",
                         "Synthetic neutrality and regional naturalness need independent human review."]}
    write_json(output / "experiment_report.json", report)
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**event, "completed": True, "report_sha256": file_sha(output / "experiment_report.json")}) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--reproduction", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.data, args.out, args.registration_commit, reproduction=args.reproduction)
    except (CorpusError, ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"V3 benchmark stopped: {type(exc).__name__}: {exc}\n")
    print(json.dumps({"report_type": report["report_type"], "macro_f1": {
        arm: {kind: result["classification"]["macro_f1"] for kind, result in models.items()}
        for arm, models in report["results"].items()}, "cost": report["cost"]}, indent=2))


if __name__ == "__main__":
    main()
