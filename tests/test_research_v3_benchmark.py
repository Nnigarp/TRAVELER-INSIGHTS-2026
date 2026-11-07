"""Contract fixtures assert access order; real assessment never fits these tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from traveler_agent.corpus import CorpusError
from traveler_agent.labels import LABELS
from traveler_agent import research_v3_benchmark as benchmark
from traveler_agent.research_v3_data import PROTOCOL


def test_registration_and_development_adapters_reject_moving_refs_or_reserved_roles():
    with pytest.raises(CorpusError, match="inmutable"):
        benchmark.verify_registration("main", {})
    with pytest.raises(CorpusError, match="tune"):
        benchmark.development_rows([{"role":"assessment"}])
    with pytest.raises(CorpusError, match="tune"):
        benchmark.attribution_on_development([{"role":"assessment"}], [{"role":"tune"}], {}, {})


def test_single_assessment_inference_after_models_calibration_policy_and_ablations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/processed/fixture"
    root.mkdir(parents=True)
    output = tmp_path / "artifacts/fixture"
    protocol = json.loads(PROTOCOL.read_text())
    rows = []
    for role in ("train", "tune", "calibration", "policy_dev", "assessment"):
        for family in range(3 if role != "assessment" else 2):
            for label in LABELS:
                rows.append({"review_id": f"{role}-{family}-{label}", "role": role,
                             "family": f"{role}-{family}", "text": f"{role}|{label}",
                             "intended_label": label, "source": "FIXTURE", "phenomena": ["negation"],
                             "region": "es", "length_band": "short", "domain": "hotel", "aspect": "price"})
    for role in ("ambiguity_policy", "ambiguity_assessment"):
        rows.append({"role":role,"review_id":role,"text":f"{role}|none","intended_label":None})
    manifest = {k: "fixture" for k in ("catalog_sha256", "protocol_sha256", "records_sha256")}
    monkeypatch.setattr(benchmark, "read_prepared", lambda _: (rows,manifest,protocol))
    monkeypatch.setattr(benchmark, "verify_registration", lambda *_: {})
    names = ("narrow_60", "narrow_120", "broad_60", "broad_120")
    monkeypatch.setattr(benchmark, "training_arms", lambda _: {k:[r for r in rows if r["role"]=="train"] for k in names})
    monkeypatch.setattr(benchmark, "shortcut_audit", lambda *_: {"fixture":True})
    selected, assessed, ablated = [], [], []

    class Estimator:
        classes_ = np.asarray(LABELS)

        def predict_proba(self, texts):
            if texts[0].startswith("assessment|"):
                assert len(selected) == 4 and ablated
                assert (output/"selection_manifest.json").exists()
                assert (output/"development_report.json").exists()
                assessed.append(self)
            return np.asarray([[float(t.endswith(label)) for label in LABELS] if not t.endswith("|none") else [1/3]*3 for t in texts])

    def selection(training, tune, _):
        assert all(r["role"]=="train" for r in training)
        assert all(r["role"]=="validation" for r in tune)
        selected.append(True)
        return ({kind:Estimator() for kind in ("dummy","lr","svm")},
                {kind:{"selected_parameters":{},"validation":{"macro_f1":1.}} for kind in ("dummy","lr","svm")})

    def ablation(training,tune,*_):
        assert all(r["role"]=="train" for r in training)
        assert all(r["role"]=="tune" for r in tune)
        ablated.append(True)
        return {"fixture":True}

    monkeypatch.setattr(benchmark,"select_models",selection)
    monkeypatch.setattr(benchmark,"attribution_on_development",ablation)
    monkeypatch.setattr(benchmark,"latency",lambda *_:{"fixture":True})
    monkeypatch.setattr(benchmark.joblib,"dump",lambda _,p:Path(p).write_bytes(b"fixture"))
    monkeypatch.setattr(benchmark.subprocess,"check_output",lambda *_,**__:"0"*40)
    report=benchmark.run(root,output,"0"*40)
    assert len(assessed) == len({id(m) for m in assessed}) == 12
    assert report["assessment_rows"] == 6 and report["assessment_families"] == 2
    assert len((output/"assessment_predictions.jsonl").read_text().splitlines()) == 72
    assert len((root/"v3_evaluation_journal.jsonl").read_text().splitlines()) == 2
    for models in report["results"].values():
        for result in models.values():
            assert result["classification"]["macro_f1"] == 1.
            assert result["policies"]["SELECTED_ASYMMETRIC_TEMPERATURE"]["ambiguity"]["sentiment_accuracy"] is None
    with pytest.raises(CorpusError,match="abierta"):
        benchmark.run(root,tmp_path/"artifacts/reuse","0"*40)
