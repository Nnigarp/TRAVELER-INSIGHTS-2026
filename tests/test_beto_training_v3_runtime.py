"""V3 lifecycle contracts using fake backends, never pilot predictions/training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from traveler_agent import beto_training_v3 as runner
from traveler_agent.corpus import CorpusError
from traveler_agent.labels import LABELS


DATA_COMMIT = "a" * 40
SELECTION_COMMIT = "b" * 40
RECORDS_SHA = "c" * 64


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(role, count=9):
    return [{"role": role, "text": f"{role}:{i}", "intended_label": LABELS[i % 3],
             "review_id": f"{role}-{i}", "family": f"{role}-{i // 3}",
             "semantic_component": f"{role}-{i // 3}", "region": "fixture",
             "length_band": "short", "phenomena": []} for i in range(count)]


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    protocol_path = Path(__file__).resolve().parents[1] / "src/traveler_agent/research_data/beto_v3_pilot_protocol.json"
    protocol = read(protocol_path)
    protocol["compute_budget"]["max_replica_and_assessment_wall_seconds"] = 10
    protocol["replica_seeds"] = []
    parts = {role: rows(role) for role in protocol["pilot_role_rows"]}
    manifest = {"records_sha256": RECORDS_SHA}
    chosen = {"id": "top2_256", "configuration": deepcopy(protocol["candidates"][2]),
              "status": "EXECUTED", "head_epoch": 20, "selected_epoch": 1,
              "training": {"fit_wall_seconds": 1},
              "tune": {"macro_f1": .5, "negative_to_positive": 1, "cross_entropy": 1.}}
    selection = {"schema": "beto-v3-pilot-selection", "records_sha256": RECORDS_SHA,
                 "data_commit": DATA_COMMIT, "protocol_commit": runner.PROTOCOL_COMMIT,
                 "selected_adjusted": chosen, "best_overall": deepcopy(chosen),
                 "selected_model_key": "beto_top2_256", "candidates": [deepcopy(chosen)],
                 "classical_selection": {"lr": {"selected": {"C": 1.}}},
                 "head_selection": {"128": {"selected": {"epoch": 20}},
                                    "256": {"selected": {"epoch": 20}}},
                 "model_entries": [{"key": "beto_top2_256", "kind": "beto", "artifact": "model",
                                    "manifest_sha256": "original", "configuration": deepcopy(chosen["configuration"])}],
                 "cost": {"wall_seconds": 1.}}
    registered = deepcopy(selection)
    clock = SimpleNamespace(now=0.)
    calls = []
    state = SimpleNamespace(protocol=protocol, parts=parts, manifest=manifest, selection=selection,
                            registered=registered, clock=clock, calls=calls, hook=None)

    class Model:
        classes_ = LABELS

        def predict_proba(self, texts):
            calls.append(list(texts))
            if state.hook:
                state.hook(texts)
            return np.tile([.6, .2, .2], (len(texts), 1))

    def git_show(command, **_kwargs):
        assert command == ["git", "show", f"{SELECTION_COMMIT}:{runner.SELECTION_PATH}"]
        return json.dumps(state.registered).encode()

    def prepare(name="development", value=None):
        out = tmp_path / "artifacts" / name
        out.mkdir(parents=True)
        runner.write_json(out / "development-selection.json", value or selection)
        runner.write_json(out / "selected-configuration.json", value or selection)
        return out

    state.prepare = prepare
    state.model = Model()
    monkeypatch.setattr(runner, "verify", lambda *_: (parts, manifest, protocol))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: int(clock.now * 1e9))
    monkeypatch.setattr(runner.gc, "collect", lambda: 0)
    monkeypatch.setattr(runner, "resource_report", lambda: {})
    monkeypatch.setattr(runner.platform, "platform", lambda: "fixture-platform")
    monkeypatch.setattr(runner, "_load_entry", lambda *_: state.model)
    monkeypatch.setattr(runner, "fit_temperature", lambda *_: {"temperature": 1.})
    monkeypatch.setattr(runner, "select_policy", lambda *_: {"selected": None})
    monkeypatch.setattr(runner.subprocess, "check_output", git_show)
    return state


@pytest.mark.parametrize("count", [3, 9])
def test_feature_encoding_checks_after_every_call_even_the_last(runtime, count):
    calls = []

    def encode(texts):
        calls.append(texts)
        runtime.clock.now = 11
        return np.zeros((len(texts), 768))

    with pytest.raises(runner.BudgetExceeded, match="feature_encoding:batch_0:after"):
        runner.encode(SimpleNamespace(encode=encode), rows("train", count), deadline=10)
    assert len(calls) == 1


@pytest.mark.parametrize("role", ["calibration", "policy_dev", "ambiguity_policy",
                                  "assessment", "ambiguity_assessment", "tune"])
def test_assessment_budget_covers_all_inference_and_retains_truthful_lock(runtime, role):
    out = runtime.prepare()

    def expire(texts):
        if texts[0].startswith("assessment:"):
            assert (out / "assessment-started.json").exists()
            assert runner._observation_marker(RECORDS_SHA).exists()
            for name in ("evaluation-models.json", "replica-status.json", "calibration-policy-selection.json"):
                assert (out / name).exists()
        if texts[0].startswith(role + ":"):
            runtime.clock.now = 11

    runtime.hook = expire
    with pytest.raises(runner.BudgetExceeded) as failure:
        runner.assess(out, Path("registration"), Path("cache"), DATA_COMMIT,
                      replicas=False, selection_commit=SELECTION_COMMIT)
    assert failure.value.stage.endswith(":after")
    interruption = read(out / "assessment-interruption.json")
    locked = role in {"assessment", "ambiguity_assessment", "tune"}
    assert interruption["status"] == "INTERRUPTED_COMPUTE_BUDGET"
    assert interruption["stage"] == failure.value.stage
    assert interruption["assessment_lock_retained"] is locked
    assert interruption["assessment_predictions_attempted"] is locked
    assert interruption["corpus_observation_marker_retained"] is locked
    assert (out / "assessment-started.json").exists() is locked
    assert not (out / "assessment.json").exists()
    assert runtime.calls[-1][0].startswith(role + ":")
    if role == "assessment":
        # Even a returned over-budget first batch is evidence of observation;
        # no second assessment batch or ambiguity/latency call may start.
        assert interruption["assessment_prediction_calls_started"] == 1
        assert interruption["assessment_prediction_calls_returned"] == 1
        assert len([call for call in runtime.calls if call[0].startswith("assessment:")]) == 1


@pytest.mark.parametrize("stage", ["temperature_selection", "policy_selection"])
def test_assessment_checks_deadline_after_noninference_selection_stages(runtime, monkeypatch, stage):
    out = runtime.prepare()

    def expire(*_args):
        runtime.clock.now = 11
        return {"temperature": 1.} if stage == "temperature_selection" else {"selected": None}

    monkeypatch.setattr(runner, "fit_temperature" if stage == "temperature_selection" else "select_policy", expire)
    with pytest.raises(runner.BudgetExceeded, match=stage + ":beto_top2_256:after"):
        runner.assess(out, Path("registration"), Path("cache"), DATA_COMMIT,
                      replicas=False, selection_commit=SELECTION_COMMIT)
    assert not runner._observation_marker(RECORDS_SHA).exists()
    assert not any(call[0].startswith("assessment:") for call in runtime.calls)


def test_lock_is_retained_when_budget_expires_before_first_final_prediction(runtime, monkeypatch):
    out = runtime.prepare()

    def load(*_args):
        if (out / "assessment-started.json").exists():
            runtime.clock.now = 11
        return runtime.model

    monkeypatch.setattr(runner, "_load_entry", load)
    with pytest.raises(runner.BudgetExceeded, match="assessment_load:beto_top2_256:after"):
        runner.assess(out, Path("registration"), Path("cache"), DATA_COMMIT,
                      replicas=False, selection_commit=SELECTION_COMMIT)
    evidence = read(out / "assessment-interruption.json")
    assert evidence["assessment_lock_retained"]
    assert evidence["corpus_observation_marker_retained"]
    assert evidence["assessment_predictions_attempted"] is False
    assert evidence["assessment_prediction_calls_started"] == 0
    assert not any(call[0].startswith("assessment:") for call in runtime.calls)


def test_replica_and_calibration_share_one_absolute_deadline(runtime, monkeypatch):
    runtime.protocol["compute_budget"]["max_replica_and_assessment_wall_seconds"] = 900
    runtime.protocol["replica_seeds"] = [7, 101]
    out = runtime.prepare()
    deadlines = []

    def replay(*_args, deadline=None, **_kwargs):
        deadlines.append(deadline)
        runtime.clock.now = 800

    def expire(texts):
        if texts[0].startswith("calibration:"):
            runtime.clock.now += 101

    runtime.hook = expire
    monkeypatch.setattr(runner, "replay", replay)
    monkeypatch.setattr(runner, "_entry", lambda _out, key, kind, **_kwargs: {"key": key, "kind": kind})
    with pytest.raises(runner.BudgetExceeded, match="calibration:beto_top2_256:batch_0:after"):
        runner.assess(out, Path("registration"), Path("cache"), DATA_COMMIT,
                      selection_commit=SELECTION_COMMIT)
    assert deadlines == [810]
    evidence = read(out / "assessment-interruption.json")
    assert [row["status"] for row in evidence["replica_status"]] == ["EXECUTED", "NOT_EXECUTED_BUDGET"]
    assert not evidence["assessment_predictions_attempted"]


def test_corpus_lock_blocks_another_output_and_development(runtime, monkeypatch):
    first = runtime.prepare("first")
    marker = runner.acquire_assessment_lock(first, {"records_sha256": RECORDS_SHA, "scope": runner.FRESH_SCOPE})
    original = runner._observation_marker(RECORDS_SHA).read_bytes()
    second = runtime.prepare("second")
    with pytest.raises(CorpusError, match="observado"):
        runner.acquire_assessment_lock(second, {"records_sha256": RECORDS_SHA})
    with pytest.raises(CorpusError, match="observado"):
        runner.assess(second, Path("registration"), Path("cache"), DATA_COMMIT,
                      selection_commit=SELECTION_COMMIT)
    for explicit in (False, True):
        with pytest.raises(CorpusError, match="observado"):
            runner.develop(Path("artifacts/new-development"), Path("registration"), Path("cache"), DATA_COMMIT,
                           reproduce_observed=explicit)
    assert marker.exists()
    assert not Path("artifacts/new-development").exists()
    assert runner._observation_marker(RECORDS_SHA).read_bytes() == original
    assert not runtime.calls


@pytest.mark.parametrize("working_copy_exists", [False, True])
def test_tracked_assessment_report_also_blocks_freshness(runtime, monkeypatch, working_copy_exists):
    report = {"data_manifest": {"records_sha256": RECORDS_SHA}}
    if working_copy_exists:
        path = Path(runner.ASSESSMENT_PATH)
        path.parent.mkdir(parents=True)
        runner.write_json(path, report)

    def git(command, **_kwargs):
        assert command in (["git", "ls-files", "--error-unmatch", runner.ASSESSMENT_PATH],
                           ["git", "show", f"HEAD:{runner.ASSESSMENT_PATH}"])
        return SimpleNamespace(returncode=0, stdout=json.dumps(report).encode())

    monkeypatch.setattr(runner.subprocess, "run", git)
    out = runtime.prepare()
    with pytest.raises(CorpusError, match="observado"):
        runner.assess(out, Path("registration"), Path("cache"), DATA_COMMIT, selection_commit=SELECTION_COMMIT)
    assert not runtime.calls


def test_primary_assessment_requires_exact_committed_selection_before_any_inference(runtime):
    out = runtime.prepare()
    with pytest.raises(CorpusError, match="Commit completo"):
        runner.assess(out, Path("registration"), Path("cache"), DATA_COMMIT)
    runtime.registered["cost"]["wall_seconds"] = 99
    with pytest.raises(CorpusError, match="difiere"):
        runner.assess(out, Path("registration"), Path("cache"), DATA_COMMIT, selection_commit=SELECTION_COMMIT)
    assert not runtime.calls
    assert not (out / "assessment-started.json").exists()


def test_explicit_reproduction_permits_artifact_time_differences_and_reports_scope(runtime):
    first = runtime.prepare("first")
    runner.acquire_assessment_lock(first, {"records_sha256": RECORDS_SHA, "scope": runner.FRESH_SCOPE})
    original = runner._observation_marker(RECORDS_SHA).read_bytes()
    runtime.selection["cost"]["wall_seconds"] = 123
    runtime.selection["model_entries"][0]["manifest_sha256"] = "new artifact"
    runtime.selection["model_entries"][0]["artifact"] = "new portable model"
    runtime.selection["selected_adjusted"]["training"]["fit_wall_seconds"] = 9
    out = runtime.prepare("reproduction")
    report = runner.assess(out, Path("registration"), Path("cache"), DATA_COMMIT,
                           replicas=False, selection_commit=SELECTION_COMMIT, reproduce_observed=True)
    assert report["scope"] == runner.REPRODUCTION_SCOPE
    assert report["new_assessment_is_now_observed"] is False
    assert report["prior_observation_evidence"]
    assert read(out / "assessment-started.json")["scope"] == runner.REPRODUCTION_SCOPE
    assert runner._observation_marker(RECORDS_SHA).read_bytes() == original


@pytest.mark.parametrize("decision", ["configuration", "head_epoch", "selected_epoch", "C", "frozen_head", "dataset", "comparison"])
def test_explicit_reproduction_cannot_change_registered_decisions(runtime, decision):
    current = deepcopy(runtime.selection)
    if decision == "configuration":
        current["selected_adjusted"]["configuration"]["hard_weight"] = 1.5
    elif decision in {"head_epoch", "selected_epoch"}:
        current["selected_adjusted"][decision] += 1
    elif decision == "C":
        current["classical_selection"]["lr"]["selected"]["C"] = 10
    elif decision == "frozen_head":
        current["head_selection"]["128"]["selected"]["epoch"] = 40
    elif decision == "dataset":
        current["records_sha256"] = "d" * 64
    else:
        current["model_entries"][0]["key"] = "beto_new_choice"
    with pytest.raises(CorpusError, match="difiere"):
        runner.verify_selection_commit(current, SELECTION_COMMIT, reproduce_observed=True)


@pytest.mark.parametrize("caller_deadline,expected", [(None, 900), (5, 5)])
def test_standalone_replay_has_absolute_cap_and_stops_after_late_encoding(runtime, monkeypatch, caller_deadline, expected):
    runtime.protocol["compute_budget"]["max_replica_and_assessment_wall_seconds"] = 2000
    selection_path = runtime.prepare() / "selected-configuration.json"
    calls = []

    class Encoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts):
            calls.append(texts)
            runtime.clock.now = expected + 1
            return np.zeros((len(texts), 768))

    monkeypatch.setattr(runner, "EncoderV2", Encoder)
    monkeypatch.setattr(runner, "fit_head_v3", lambda *_args, **_kwargs: pytest.fail("Training started after deadline"))
    out = Path("artifacts/replay")
    with pytest.raises(runner.BudgetExceeded, match="feature_encoding:batch_0:after"):
        runner.replay(out, Path("registration"), Path("cache"), DATA_COMMIT, selection_path, 42,
                      deadline=caller_deadline)
    assert len(calls) == 1
    assert read(out / "replay-interruption.json")["wall_seconds_cap"] == 900
    assert read(out / "replay-interruption.json")["effective_wall_seconds_cap"] == expected
    assert not (out / "replay.json").exists()


@pytest.mark.parametrize("reference", ["v1", "v2"])
@pytest.mark.parametrize("failure", ["budget", "backend"])
def test_optional_reference_failure_never_loses_valid_selection(runtime, monkeypatch, reference, failure):
    runtime.protocol["compute_budget"]["max_development_wall_seconds"] = 2000
    runtime.protocol["candidates"] = runtime.protocol["candidates"][:3]
    out = Path("artifacts/development")
    weights = (np.zeros((3, 768)), np.zeros(3))
    tune_score = {"macro_f1": .5, "negative_to_positive": 1, "cross_entropy": 1.}
    failures = []

    class Encoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts):
            return np.zeros((len(texts), 768))

    def fit(_x, train, _dev, tune, _protocol, _seed, *, fixed_epoch=None, deadline=None):
        assert all(row["role"] == "train" for row in train)
        if fixed_epoch is not None:
            # This is deliberately checked INSIDE the optional head fit: the
            # primary decision must already be durable before the failure.
            assert read(out / "development-selection.json")["selected_model_key"] == "beto_top2_256"
            assert (out / "selected-configuration.json").exists()
            if fixed_epoch == (30 if reference == "v1" else 80):
                failures.append(fixed_epoch)
                if failure == "budget":
                    raise runner.BudgetExceeded("optional_reference_head")
                raise RuntimeError("fixture backend failure")
        return weights, {"selected": {"epoch": 20, "tune": tune_score}}

    def train_encoder(*_args, **_kwargs):
        return weights, {"selected": {"epoch": 1, "tune": tune_score},
                         "history": [{"elapsed_seconds": 1}], "fit_wall_seconds": 1}

    def entry(_out, key, kind, **kwargs):
        return {"key": key, "kind": kind, "configuration": kwargs.get("configuration")}

    monkeypatch.setattr(runner, "EncoderV2", Encoder)
    monkeypatch.setattr(runner, "fit_head_v3", fit)
    monkeypatch.setattr(runner, "train_encoder_v3", train_encoder)
    monkeypatch.setattr(runner, "train_classical", lambda *_: ({"lr": object()}, {"lr": {"selected": {"C": 1.}}}))
    monkeypatch.setattr(runner, "_entry", entry)
    monkeypatch.setattr(runner, "export_model", lambda *_args, **_kwargs: None)
    selected = runner.develop(out, Path("registration"), Path("cache"), DATA_COMMIT)
    assert failures
    assert selected["selected_model_key"] == "beto_top2_256"
    assert read(out / "development-selection.json") == json.loads(json.dumps(selected))
    failed = next(row for row in selected["historical_recipe_references"] if row["key"] == f"beto_reference_{reference}")
    assert failed["status"] == ("INTERRUPTED_COMPUTE_BUDGET" if failure == "budget" else "FAILED_OPTIONAL_REFERENCE")
    assert read(out / "progress.json")["stage"] == "DEVELOPMENT_COMPLETED"


def test_assess_cli_requires_selection_commit_before_touching_data(capsys):
    with pytest.raises(SystemExit, match="2"):
        runner.main(["assess", "--output", "artifacts/unused"])
    assert "--selection-commit" in capsys.readouterr().err
