"""V3 optimization contracts on small synthetic tensors, never pilot outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from traveler_agent import beto_training_v3_model as models
from traveler_agent.corpus import CorpusError, file_sha
from traveler_agent.labels import LABELS


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[1] / "src/traveler_agent/research_data/beto_v3_pilot_protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def torch():
    return pytest.importorskip("torch", reason="dedicated V3 CPU research environment")


@pytest.fixture
def encoder_class(torch):
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Linear(3, 3, bias=False)
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([
                torch.nn.Linear(3, 3, bias=False) for _ in range(12)
            ])
            for parameter in self.parameters():
                torch.nn.init.eye_(parameter)
                parameter.requires_grad_(False)

    class FixtureEncoder:
        def __init__(self):
            self.model = Tiny()
            self.encode_calls = 0
            self.token_calls = 0
            self.token_hook = None

        def tokens(self, texts):
            self.token_calls += 1
            if self.token_hook:
                self.token_hook(self.token_calls)
            return torch.eye(3)[[int(text) for text in texts]]

        def pooled(self, x):
            x = self.model.embeddings(x)
            for layer in self.model.encoder.layer:
                x = x + .1 * layer(x)
            return torch.nn.functional.pad(torch.nn.functional.normalize(x, dim=1), (0, 765))

        def encode(self, texts):
            self.encode_calls += 1
            self.model.eval()
            with torch.no_grad():
                return self.pooled(self.tokens(texts)).numpy()

    return FixtureEncoder


def rows(count=12):
    return [{"role": "train", "text": str(i % 3), "intended_label": LABELS[i % 3],
             "phenomena": ["negation"] if i % 2 == 0 else [],
             "semantic_component": f"fixture-{i // 3}"} for i in range(count)]


def initial_head():
    weight = np.zeros((3, 768), dtype=np.float32)
    weight[:, :3] = np.eye(3)
    return weight, np.zeros(3, dtype=np.float32)


def score(macro=.5, dangerous=1, nll=1.):
    return {"macro_f1": macro, "negative_to_positive": dangerous, "cross_entropy": nll}


def tiny_config(protocol):
    config = deepcopy(next(row for row in protocol["candidates"] if row["id"] == "top2_256"))
    config.update(encoder_lr=.001, head_lr=.001)
    return config


@pytest.mark.parametrize("candidate,expected", [
    (score(.500002, 5, 5.), True),
    (score(.499998, 0, .1), False),
    (score(.5000005, 0, 5.), True),
    (score(.5000005, 2, .1), False),
    (score(.5, 1, .9998), True),
    (score(.5, 1, .99995), False),
    (score(.5, 1, 1.), False),
])
def test_checkpoint_rule_requires_registered_material_improvement(candidate, expected):
    assert models.meaningful_improvement(candidate, score()) is expected
    assert models.meaningful_improvement(candidate, None)


def test_nonfinite_tune_score_cannot_become_a_checkpoint():
    with pytest.raises(CorpusError, match="Non-finite"):
        models.meaningful_improvement(score(nll=float("nan")), None)


@pytest.mark.parametrize("wrong_role", ["assessment", "calibration", "policy_dev", "ambiguity_policy"])
def test_optimization_and_sampling_refuse_nontrain_or_nontune_roles(wrong_role, tmp_path):
    wrong = [{"role": wrong_role}]
    train = [{"role": "train"}]
    tune = [{"role": "tune"}]
    with pytest.raises(CorpusError, match="TRAIN"):
        models.fit_head_v3(None, wrong, None, tune, {}, 42)
    with pytest.raises(CorpusError, match="tune"):
        models.fit_head_v3(None, train, None, wrong, {}, 42)
    with pytest.raises(CorpusError, match="TRAIN"):
        models.train_encoder_v3(None, wrong, tune, None, {}, {}, 42, tmp_path / "bad.safetensors")
    with pytest.raises(CorpusError, match="tune"):
        models.train_encoder_v3(None, train, wrong, None, {}, {}, 42, tmp_path / "bad.safetensors")
    with pytest.raises(CorpusError, match="TRAIN"):
        models.train_only_sampling_weights(wrong, {"hard_tags": []})
    assert not (tmp_path / "bad.safetensors").exists()


def test_hard_weights_have_equal_class_mass_and_capped_within_class_ratio(protocol):
    train = []
    for label, count in zip(LABELS, (2, 4, 6)):
        train.extend({"role": "train", "intended_label": label,
                      "phenomena": ["negation"] if i % 2 == 0 else []} for i in range(count))
    probabilities = models.train_only_sampling_weights(train, protocol, hard_weight=1.5)
    capped = models.train_only_sampling_weights(train, protocol, hard_weight=20.)
    np.testing.assert_array_equal(probabilities, capped)
    for label in LABELS:
        values = probabilities[[row["intended_label"] == label for row in train]]
        assert values.sum() == pytest.approx(1 / 3)
        assert values.max() / values.min() == pytest.approx(1.5)
    with pytest.raises(CorpusError, match="three TRAIN classes"):
        models.train_only_sampling_weights(train[:2], protocol)


def test_hard_sampler_is_seeded_replacement_with_exact_exposure_accounting(protocol, torch):
    train = rows()
    first, log = models.sample_train_epoch(train, protocol, 42, 1, hard_weight=1.5)
    again, _ = models.sample_train_epoch(train, protocol, 42, 1, hard_weight=1.5)
    other, _ = models.sample_train_epoch(train, protocol, 7, 1, hard_weight=1.5)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert len(first) == len(train) and log["draws"] == len(train)
    assert log["replacement"] and log["repeated_draws"] > 0
    assert log["row_exposure_counts"] == np.bincount(first, minlength=len(train)).tolist()
    assert sum(log["class_draws"].values()) == sum(log["row_exposure_counts"]) == len(train)
    assert log["hard_draws"] == sum(bool(train[i]["phenomena"]) for i in first)
    assert sum(log["hard_class_draws"].values()) == log["hard_draws"]
    assert all(value == pytest.approx(1 / 3) for value in log["class_probability_mass"].values())
    assert sum(log["component_draws"].values()) == len(train)
    assert log["scope"] == "TRAIN_TAGS_ONLY_NO_TUNE_OR_MODEL_OUTPUTS"
    standard, standard_log = models.sample_train_epoch(train, protocol, 42, 1)
    assert sorted(standard.tolist()) == list(range(len(train)))
    assert standard_log["row_exposure_counts"] == [1] * len(train)
    assert not standard_log["replacement"]


def test_registered_head_selection_and_fixed_replay_never_use_tune(protocol, torch, monkeypatch):
    train = rows(18)
    tune = [{**row, "role": "tune"} for row in train]
    x = np.zeros((len(train), 768), dtype=np.float32)
    x[np.arange(len(train)), np.arange(len(train)) % 3] = 1.
    calls = []
    def constant_score(*_args):
        calls.append(True)
        return score()
    monkeypatch.setattr(models, "tuning_score", constant_score)
    weights, log = models.fit_head_v3(x, train, x, tune, protocol, 42)
    assert log["planned_epochs"] == log["executed_epochs"] == 80
    assert [row["epoch"] for row in log["history"] if "tune" in row] == [20, 40, 80]
    assert len(calls) == 3 and log["selected"]["epoch"] == 20
    class ForbiddenFeatures:
        def __array__(self, *_args, **_kwargs):
            pytest.fail("Fixed head replay inspected tune features")
    monkeypatch.setattr(models, "tuning_score", lambda *_: pytest.fail("Fixed head replay scored tune"))
    replay, replay_log = models.fit_head_v3(x, train, ForbiddenFeatures(), [{"role": "tune"}],
                                           protocol, 42, fixed_epoch=20)
    np.testing.assert_array_equal(weights[0], replay[0])
    np.testing.assert_array_equal(weights[1], replay[1])
    assert replay_log["executed_epochs"] == 20 and replay_log["planned_epochs"] == 80
    assert not replay_log["tune_inference_executed"] and not replay_log["selection_retuned"]
    assert all("tune" not in row for row in replay_log["history"])


def test_encoder_patience_restores_paired_best_and_fixed_original_schedule(
        protocol, encoder_class, torch, tmp_path, monkeypatch):
    from safetensors.torch import load_file

    protocol["warmup_fraction"] = .2
    config = tiny_config(protocol)
    train = rows()
    tune = [{**row, "role": "tune"} for row in train[:3]]
    encoder = encoder_class()
    before = {name: value.clone() for name, value in encoder.model.state_dict().items()}
    monkeypatch.setattr(models, "tuning_score", lambda *_: score())
    primary_path = tmp_path / "primary.safetensors"
    weights, log = models.train_encoder_v3(encoder, train, tune, initial_head(), config, protocol,
                                          42, primary_path)
    assert log["selected"]["epoch"] == 1 and log["executed_epochs"] == 2
    assert log["stopped_early"] and log["stop_reason"] == "PATIENCE"
    assert encoder.encode_calls == 2
    assert log["first_encoder_gradient_norm"] > 0 and log["probe_tensor_l2_update"] > 0
    assert log["nonzero_encoder_gradient_observed"] and log["frozen_parameter_gradients_absent"]
    assert log["encoder_trainable_parameters"] == 18 and log["head_trainable_parameters"] == 2307
    assert log["adaptation_sha256"] == file_sha(primary_path)
    allowed = models._allowed_keys(encoder.model, "top2")
    assert set(load_file(str(primary_path))) == allowed
    assert all(torch.equal(before[name], value) for name, value in encoder.model.state_dict().items()
               if name not in allowed)
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())

    replay = encoder_class()
    monkeypatch.setattr(replay, "encode", lambda *_: pytest.fail("Fixed encoder replay encoded tune"))
    monkeypatch.setattr(models, "tuning_score", lambda *_: pytest.fail("Fixed encoder replay scored tune"))
    replay_path = tmp_path / "replay.safetensors"
    replay_weights, replay_log = models.train_encoder_v3(
        replay, train, [{"role": "tune"}], initial_head(), config, protocol, 42, replay_path, fixed_epoch=1)
    np.testing.assert_array_equal(weights[0], replay_weights[0])
    np.testing.assert_array_equal(weights[1], replay_weights[1])
    first = load_file(str(primary_path))
    second = load_file(str(replay_path))
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert all(torch.equal(encoder.model.state_dict()[name], first[name]) for name in first)
    assert replay_log["scheduler_total_steps"] == log["scheduler_total_steps"] == 9
    assert replay_log["scheduler_warmup_steps"] == log["scheduler_warmup_steps"] == 1
    assert replay_log["history"][0]["last_encoder_lr"] == log["history"][0]["last_encoder_lr"]
    assert replay_log["optimizer_steps"] == 3 and replay_log["planned_epochs"] == 3
    assert replay_log["stop_reason"] == "FIXED_EPOCHS"
    assert not replay_log["tune_inference_executed"] and not replay_log["selection_retuned"]
    assert all("tune" not in row for row in replay_log["history"])


def test_hard_encoder_fixed_replay_preserves_seeded_draws(protocol, encoder_class, torch, tmp_path):
    from safetensors.torch import load_file

    config = tiny_config(protocol)
    config["hard_weight"] = 1.5
    train = rows()
    paths = [tmp_path / "first.safetensors", tmp_path / "again.safetensors"]
    results = [models.train_encoder_v3(encoder_class(), train, [], initial_head(), config, protocol,
                                       42, path, fixed_epoch=2) for path in paths]
    assert [row["sampling"] for row in results[0][1]["history"]] == [
        row["sampling"] for row in results[1][1]["history"]]
    for epoch in results[0][1]["history"]:
        assert epoch["sampling"]["draws"] == len(train)
        assert epoch["sampling"]["replacement"]
    np.testing.assert_array_equal(results[0][0][0], results[1][0][0])
    first, second = (load_file(str(path)) for path in paths)
    assert all(torch.equal(value, second[name]) for name, value in first.items())


def test_full_single_epoch_trains_and_exports_all_encoder_tensors(
        protocol, encoder_class, torch, tmp_path, monkeypatch):
    from safetensors.torch import load_file

    config = deepcopy(next(row for row in protocol["candidates"] if row["mode"] == "full"))
    config.update(encoder_lr=.001, head_lr=.001)
    encoder = encoder_class()
    monkeypatch.setattr(models, "tuning_score", lambda *_: score())
    path = tmp_path / "full.safetensors"
    _, log = models.train_encoder_v3(encoder, rows(), [{**row, "role": "tune"} for row in rows(3)],
                                    initial_head(), config, protocol, 42, path)
    assert log["executed_epochs"] == log["selected"]["epoch"] == 1
    assert log["stop_reason"] == "MAX_EPOCHS" and not log["stopped_early"]
    assert set(load_file(str(path))) == set(encoder.model.state_dict())
    assert log["encoder_trainable_parameters"] == 117


def test_expired_budget_fails_before_head_work_or_checkpoint_creation(protocol, tmp_path):
    deadline = time.perf_counter() - 1
    with pytest.raises(models.BudgetExceeded, match="head_start"):
        models.fit_head_v3(None, rows(), None, [{"role": "tune"}], protocol, 42, deadline=deadline)
    path = tmp_path / "not-created" / "model.safetensors"
    with pytest.raises(models.BudgetExceeded, match="encoder_start"):
        models.train_encoder_v3(None, rows(), [{"role": "tune"}], None, {}, protocol, 42,
                                path, deadline=deadline)
    assert not path.parent.exists()


def test_midtraining_deadline_removes_staged_best_without_publishing_a_partial_model(
        protocol, encoder_class, tmp_path, monkeypatch):
    import safetensors.torch

    clock = [0.]
    monkeypatch.setattr(models.time, "perf_counter", lambda: clock[0])
    encoder = encoder_class()
    encoder.token_hook = lambda count: clock.__setitem__(0, 11.) if count == 5 else None
    monkeypatch.setattr(models, "tuning_score", lambda *_: score())
    saved = []
    original_save = safetensors.torch.save_file
    def traced_save(tensors, filename, **kwargs):
        original_save(tensors, filename, **kwargs)
        saved.append(Path(filename))
    monkeypatch.setattr(safetensors.torch, "save_file", traced_save)
    path = tmp_path / "deadline.safetensors"
    with pytest.raises(models.BudgetExceeded, match="encoder_batch_finished"):
        models.train_encoder_v3(encoder, rows(), [{**row, "role": "tune"} for row in rows(3)],
                                initial_head(), tiny_config(protocol), protocol, 42, path, deadline=10.)
    assert len(saved) == 1
    assert not path.exists() and all(not temporary.exists() for temporary in saved)
    assert not list(tmp_path.glob("*.pending"))
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())


def test_existing_adaptation_is_never_overwritten(protocol, tmp_path):
    path = tmp_path / "existing.safetensors"
    path.write_bytes(b"existing-checkpoint")
    with pytest.raises(CorpusError, match="already exists"):
        models.train_encoder_v3(None, rows(), [{"role": "tune"}], None, tiny_config(protocol),
                                protocol, 42, path)
    assert path.read_bytes() == b"existing-checkpoint"
