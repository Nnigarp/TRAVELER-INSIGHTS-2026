"""Saved fixture validation only: no source corpus, model fitting or pilot results."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from traveler_agent import beto_training_v3_analysis as analysis
from traveler_agent.corpus import CorpusError, file_sha
from traveler_agent.labels import LABELS
from traveler_agent.research_metrics import summarize_predictions


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "artifacts" / "fake-v3"
    root.mkdir(parents=True)
    protocol = {"primary_seed": 42, "replica_seeds": [7, 101], "pilot_labeled_family_rows": 3,
                "target_5000_plan": {"train": 3501},
                "compute_budget": {"scale_development_wall_seconds_cap": 3600}}
    write(root / "protocol.json", protocol)
    members = [{"review_id": f"fake-{i}", "family": f"fake-family-{i // 3}",
                "intended_label": LABELS[i % 3]} for i in range(9)]
    manifest = {"schema": "beto-v3-pilot-manifest", "records_sha256": "c" * 64,
                # Saved protocol JSON can have different serialization from
                # the registered source resource; the corpus SHA is metadata.
                "protocol_sha256": "d" * 64, "human_gold": False,
                "counts": {"assessment": {"rows": 9, "families": 3,
                                          "classes": dict.fromkeys(LABELS, 3)},
                           "train": {"rows": 300}},
                "admitted_role_rows": {"assessment": 9, "train": 297},
                "assessment_membership": members, "quarantined_families": []}
    write(root / "manifest.json", manifest)
    truth = [row["intended_label"] for row in members]
    predictions = {key: list(truth) for key in ("lr", "svm", "beto_top2_256", "beto_seed7")}
    predictions["lr"][0] = "positivo"
    predictions["svm"][4] = "negativo"
    predictions["beto_top2_256"][0:2] = ["positivo", "negativo"]
    predictions["beto_seed7"][3] = "positivo"
    saved = []
    for key, guesses in predictions.items():
        for row, guess in zip(members, guesses):
            probability = [.1, .1, .1]
            probability[LABELS.index(guess)] = .8
            saved.append({**row, "model": key, "prediction": guess,
                          "probabilities": probability, "calibrated_probabilities": probability})
    write(root / "assessment-predictions.json", saved)
    selection = {"schema": "beto-v3-pilot-selection", "records_sha256": manifest["records_sha256"],
                 "data_commit": "a" * 40, "protocol_commit": "b" * 40,
                 "selected_model_key": "beto_top2_256", "cost": {"wall_seconds": 500., "peak_rss_mib": 100}}
    write(root / "development-selection.json", selection)
    report = {"schema": "beto-v3-pilot-assessment", "status": "COMPLETED_WITHIN_REGISTERED_BUDGET",
              "scope": "FRESH_SYNTHETIC_PILOT_ASSESSMENT", "data_manifest": manifest,
              "data_commit": selection["data_commit"], "protocol_commit": selection["protocol_commit"],
              "selection_sha256": file_sha(root / "development-selection.json"),
              "selected_model_key": selection["selected_model_key"], "human_performance_measured": False,
              "results": {key: {"classification": summarize_predictions(truth, guesses)}
                          for key, guesses in predictions.items()},
              "model_entries": [{"key": key, "kind": "classical" if key in ("lr", "svm") else "beto"}
                                for key in predictions],
              "replica_status": [{"seed": 7, "status": "EXECUTED"},
                                 {"seed": 101, "status": "NOT_EXECUTED_BUDGET"}]}
    write(root / "assessment.json", report)
    return root


def test_paired_bootstrap_seed_omission_and_rough_cost_are_explicit(saved_run):
    paths = list(saved_run.iterdir())
    before = {path.name: path.read_bytes() for path in paths}
    result = analysis.analyze(saved_run)
    assert result == analysis.analyze(saved_run)
    assert before == {path.name: path.read_bytes() for path in paths}
    assert result["assessment_rows"] == 9
    assert result["bootstrap"] == {"unit": "synthetic semantic family", "resamples": 2000,
                                   "seed": 42, "paired_reference": "lr",
                                   "same_family_indices_for_all_models": True}
    lr = result["paired_family_bootstrap"]["lr"]
    assert lr["difference_from_lr"]["low_95"] == 0
    assert lr["difference_from_lr"]["high_95"] == 0
    seeds = result["selected_configuration_seed_summary"]
    assert seeds["primary_seed"] == 42
    assert seeds["observed_seeds"] == [42, 7]
    assert seeds["missing_seeds"] == [101]
    assert seeds["seed_results"][2]["missing_reason"] == "NOT_EXECUTED_BUDGET"
    assert seeds["seed_results"][2]["macro_f1"] is None
    values = [row["macro_f1"] for row in seeds["seed_results"][:2]]
    assert seeds["mean_macro_f1"] == pytest.approx(np.mean(values))
    assert seeds["sample_std_macro_f1"] == pytest.approx(np.std(values, ddof=1))
    assert seeds["negative_to_positive"] == [1, 1]
    assert not seeds["best_seed_selection_performed"]
    cost = result["cost_projection"]
    assert cost["admitted_train_rows"] == 297
    assert cost["rough_projected_development_wall_seconds"] == pytest.approx(500 * 3501 / 297)
    assert not cost["rough_projection_within_registered_cap"]
    assert not cost["measured_full_5000_run"]
    assert not cost["quality_inferred_from_assessment_score"]


@pytest.mark.parametrize("mutation", ["unknown_id", "wrong_label", "wrong_family", "duplicate", "missing", "wrong_argmax"])
def test_membership_and_predictions_are_independently_validated(saved_run, mutation):
    path = saved_run / "assessment-predictions.json"
    records = read(path)
    if mutation == "unknown_id":
        records[0]["review_id"] = "not-in-frozen-membership"
    elif mutation == "wrong_label":
        records[0]["intended_label"] = "neutral"
    elif mutation == "wrong_family":
        records[0]["family"] = "some-other-family"
    elif mutation == "duplicate":
        records.append(deepcopy(records[0]))
    elif mutation == "missing":
        records.pop()
    elif mutation == "wrong_argmax":
        records[0]["prediction"] = "neutral"
    write(path, records)
    with pytest.raises(CorpusError):
        analysis.analyze(saved_run)


def test_report_metrics_cannot_override_saved_predictions(saved_run):
    path = saved_run / "assessment.json"
    report = read(path)
    report["results"]["beto_top2_256"]["classification"]["macro_f1"] = .999
    write(path, report)
    with pytest.raises(CorpusError, match="Métrica guardada inconsistente"):
        analysis.analyze(saved_run)


def test_missing_cost_never_invents_a_projection(saved_run):
    (saved_run / "development-selection.json").unlink()
    cost = analysis.analyze(saved_run)["cost_projection"]
    assert cost["status"] == "NOT_AVAILABLE"
    assert "rough_projected_development_wall_seconds" not in cost


def test_missing_assessment_never_uses_interrupted_or_historical_results(saved_run):
    report = saved_run / "assessment.json"
    report.rename(saved_run / "assessment-interrupted-report.json")
    with pytest.raises(CorpusError, match="assessment.json"):
        analysis.analyze(saved_run)


def test_development_cost_requires_the_frozen_selection(saved_run):
    path = saved_run / "development-selection.json"
    selection = read(path)
    selection["cost"]["wall_seconds"] = 1
    write(path, selection)
    with pytest.raises(CorpusError, match="Coste de desarrollo"):
        analysis.analyze(saved_run)


def test_cli_uses_ignored_outputs_and_exclusive_creation(saved_run):
    output = saved_run / "analysis.json"
    analysis.main(["--run", str(saved_run), "--out", str(output)])
    assert read(output)["schema"] == "beto-v3-saved-prediction-analysis"
    with pytest.raises(FileExistsError):
        analysis.main(["--run", str(saved_run), "--out", str(output)])
    with pytest.raises(CorpusError, match="ignorada"):
        analysis.main(["--run", str(saved_run), "--out", "docs/must-not-write.json"])


def test_plot_outputs_static_common_scale_artifacts_without_source_changes(saved_run):
    pytest.importorskip("matplotlib")
    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_beto_v3.py"
    spec = importlib.util.spec_from_file_location("plot_beto_v3_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = {name: file_sha(saved_run / name) for name in ("assessment.json", "assessment-predictions.json")}
    targets = module.plot(saved_run, saved_run / "comparison")
    assert targets[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = targets[1].read_text(encoding="utf-8")
    assert "sin gold humano" in svg
    assert "semilla 42" in svg
    assert all(line == line.rstrip() for line in svg.splitlines())
    assert before == {name: file_sha(saved_run / name) for name in before}
    with pytest.raises(CorpusError, match="Gráfica existente"):
        module.plot(saved_run, saved_run / "comparison")
