"""Access-order contracts on tiny fixtures, never fitting the research challenge."""
from pathlib import Path

import numpy as np
import pytest

from traveler_agent.corpus import CorpusError
from traveler_agent.labels import LABELS
from traveler_agent import synthetic_diversity_benchmark as benchmark


def test_registration_requires_immutable_sha_before_git_access():
    with pytest.raises(CorpusError, match="SHA"):
        benchmark.verify_registration("main", {})


@pytest.mark.parametrize("training,validation", [
    ([{"role": "challenge"}], [{"role": "validation"}]),
    ([{"role": "train"}], [{"role": "challenge"}]),
])
def test_selection_rejects_reserved_roles_before_fitting(training, validation):
    with pytest.raises(CorpusError, match="Roles"):
        benchmark.select_models(training, validation, {})


def test_all_selection_precedes_single_challenge_call_and_journal_blocks_reuse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/processed/fixture"
    data.mkdir(parents=True)
    output = tmp_path / "artifacts/fixture"
    rows = []
    for role, count in (("validation", 1), ("challenge", 2)):
        for family in range(count):
            for label in LABELS:
                rows.append({"review_id": f"{role}-{family}-{label}", "family": f"{role}-{family}",
                             "role": role, "text": f"{role}|{label}", "intended_label": label,
                             "region": "es", "length_band": "short", "phenomena": ["negation"]})
    for role in ("ambiguity", "illustrative"):
        rows.append({"role": role, "text": f"{role}|unlabeled"})
    training = [{"role": "train", "review_id": f"train-{label}", "source": "FIXTURE",
                 "family": "training-fixture", "text": label, "intended_label": label}
                for label in LABELS]
    names = ("A_historical_matched", "B_new", "C_mixed_matched", "A_historical_full_sensitivity")
    protocol = {"bootstrap": {"resamples": 10, "seed": 42}}
    manifest = {key: "fixture" for key in ("catalog_sha256", "protocol_sha256", "records_sha256")}
    monkeypatch.setattr(benchmark, "read_prepared", lambda _: (rows, manifest, protocol))
    monkeypatch.setattr(benchmark, "verify_registration", lambda *_: None)
    monkeypatch.setattr(benchmark, "assemble_strategies", lambda *_: ({name: training for name in names}, {}))
    selected = []
    calls = []

    class FixtureEstimator:
        classes_ = np.asarray(LABELS)

        def predict_proba(self, texts):
            if texts[0].startswith("challenge|"):
                assert len(selected) == 4
                assert (output / "selection_manifest.json").exists()
                calls.append(self)
                return np.asarray([[float(text.endswith(label)) for label in LABELS] for text in texts])
            return np.full((len(texts), 3), 1 / 3)

    def fixture_selection(*_):
        selected.append(True)
        return ({kind: FixtureEstimator() for kind in ("dummy", "lr", "svm")},
                {kind: {"selected_parameters": {}} for kind in ("dummy", "lr", "svm")})

    monkeypatch.setattr(benchmark, "select_models", fixture_selection)
    monkeypatch.setattr(benchmark, "latency", lambda *_: {"fixture": True})
    monkeypatch.setattr(benchmark.joblib, "dump", lambda _, path: Path(path).write_bytes(b"fixture-only"))
    report = benchmark.run(data, output, registration_commit="0" * 40)
    assert report["challenge_rows"] == 6
    assert len(calls) == len({id(model) for model in calls}) == 12
    assert len((output / "challenge_predictions.jsonl").read_text().splitlines()) == 72
    assert len((data / "evaluation_journal.jsonl").read_text().splitlines()) == 2
    for records in report["strategies"].values():
        for record in records.values():
            assert record["challenge"]["macro_f1"] == 1
            assert record["unlabeled_cohorts"]["ambiguity"]["sentiment_accuracy"] is None
    with pytest.raises(CorpusError, match="abierto"):
        benchmark.run(data, tmp_path / "artifacts/reuse", registration_commit="0" * 40)
