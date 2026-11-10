"""CPU optimization for the registered V3 pilot; V2 remains unchanged.

``deadline`` is an absolute ``time.perf_counter()`` deadline. A budget failure
raises ``BudgetExceeded`` and never publishes a partial adaptation checkpoint.
Fixed-epoch replay uses TRAIN only and retains the original scheduler horizon.
The portable artifact format remains the already supported supervised V2 format.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import time
import uuid

import numpy as np

from .beto_classifier import softmax
from .beto_training_v2_model import (
    EncoderV2,
    _allowed_keys,
    export_model,
    load_model_v2,
    score_rank,
    target_ids,
    tuning_score,
)
from .corpus import CorpusError, file_sha
from .labels import LABELS


class BudgetExceeded(RuntimeError):
    """The caller's absolute training deadline expired; no artifact is valid."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"V3 compute budget exhausted during {stage}.")


def _check_deadline(deadline, stage):
    if deadline is not None:
        if not math.isfinite(deadline):
            raise CorpusError("A finite absolute training deadline is required.")
        if time.perf_counter() >= deadline:
            raise BudgetExceeded(stage)


def _roles(rows, tune, fixed_epoch):
    if not rows or any(row.get("role") != "train" for row in rows):
        raise CorpusError("V3 optimization requires TRAIN rows only.")
    if any(row.get("role") != "tune" for row in tune) or (fixed_epoch is None and not tune):
        raise CorpusError("V3 selection requires independent tune rows.")


def _training_targets(rows):
    try:
        return target_ids(rows)
    except (KeyError, ValueError) as exc:
        raise CorpusError("Canonical TRAIN author-intent labels are required.") from exc


def _epoch_count(planned, fixed_epoch):
    if isinstance(planned, bool) or not isinstance(planned, int) or planned < 1:
        raise CorpusError("A positive planned epoch count is required.")
    if fixed_epoch is not None:
        if (isinstance(fixed_epoch, bool) or not isinstance(fixed_epoch, int)
                or not 1 <= fixed_epoch <= planned):
            raise CorpusError("Fixed replay epochs must be within the original plan.")
        return fixed_epoch
    return planned


def _torch(protocol, seed):
    import torch

    if protocol.get("threads") != 2:
        raise CorpusError("The registered V3 pilot requires two CPU threads.")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return torch


def meaningful_improvement(score, previous):
    """The preregistered checkpoint rule, including material NLL tie breaks."""
    for name in ("macro_f1", "cross_entropy", "negative_to_positive"):
        if not math.isfinite(score[name]):
            raise CorpusError("Non-finite tune selection score.")
    if previous is None:
        return True
    gain = score["macro_f1"] - previous["macro_f1"]
    if gain >= 1e-6:
        return True
    if abs(gain) >= 1e-6:
        return False
    dangerous = score["negative_to_positive"] - previous["negative_to_positive"]
    return dangerous < 0 or (dangerous == 0 and previous["cross_entropy"] - score["cross_entropy"] >= 1e-4)


def _hard_rows(rows, protocol):
    tags = set(protocol["hard_tags"])
    flags = []
    for row in rows:
        observed = row.get("phenomena", [])
        if (not isinstance(observed, (list, tuple, set))
                or any(not isinstance(tag, str) for tag in observed)):
            raise CorpusError("TRAIN phenomena must be a list of tag names.")
        flags.append(bool(tags.intersection(observed)))
    return np.asarray(flags, dtype=bool)


def train_only_sampling_weights(rows, protocol, *, hard_weight=1.5):
    """Return probabilities with equal class mass and a capped TRAIN-tag ratio.

    The 1.5 cap applies before class normalization: within a class a difficult
    row has at most 1.5 times an ordinary row's probability. No tune information
    or fitted model output enters these weights.
    """
    _roles(rows, [], 1)
    if not math.isfinite(hard_weight) or hard_weight < 1:
        raise CorpusError("Hard sampling requires a finite multiplier at least one.")
    labels = _training_targets(rows)
    if set(labels.tolist()) != set(range(len(LABELS))):
        raise CorpusError("Class-normalized sampling requires all three TRAIN classes.")
    weights = np.where(_hard_rows(rows, protocol), min(hard_weight, 1.5), 1.).astype(np.float64)
    for label in range(len(LABELS)):
        mask = labels == label
        weights[mask] /= weights[mask].sum()
    return weights / weights.sum()


def sample_train_epoch(rows, protocol, seed, epoch, *, hard_weight=1.):
    """Return seeded TRAIN indices and auditable exposure counts for one epoch.

    Standard arms shuffle without replacement. The hard arm makes exactly
    ``len(rows)`` replacement draws from class-normalized capped probabilities.
    An epoch-specific local generator keeps replay independent of tune calls.
    """
    import torch

    _roles(rows, [], 1)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise CorpusError("A positive sampling epoch is required.")
    if not math.isfinite(hard_weight) or hard_weight < 1:
        raise CorpusError("Hard sampling requires a finite multiplier at least one.")
    labels = _training_targets(rows)
    hard = _hard_rows(rows, protocol)
    effective = min(hard_weight, 1.5)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch - 1)
    replacement = effective > 1.
    if replacement:
        probability = train_only_sampling_weights(rows, protocol, hard_weight=effective)
        indices = torch.multinomial(torch.from_numpy(probability), len(rows), replacement=True,
                                    generator=generator).numpy()
    else:
        probability = np.full(len(rows), 1. / len(rows))
        indices = torch.randperm(len(rows), generator=generator).numpy()
    exposure = np.bincount(indices, minlength=len(rows))
    report = {
        "scope": "TRAIN_TAGS_ONLY_NO_TUNE_OR_MODEL_OUTPUTS",
        "strategy": "CLASS_NORMALIZED_REPLACEMENT" if replacement else "UNIFORM_SHUFFLE_WITHOUT_REPLACEMENT",
        "epoch": epoch,
        "replacement": replacement,
        "requested_hard_multiplier": hard_weight,
        "effective_hard_multiplier": effective,
        "draws": len(indices),
        "unique_rows": int(np.count_nonzero(exposure)),
        "repeated_draws": int(len(indices) - np.count_nonzero(exposure)),
        "available_hard_rows": int(hard.sum()),
        "hard_draws": int(hard[indices].sum()),
        "class_probability_mass": {label: float(probability[labels == i].sum()) for i, label in enumerate(LABELS)},
        "class_draws": {label: int((labels[indices] == i).sum()) for i, label in enumerate(LABELS)},
        "hard_class_draws": {label: int(((labels[indices] == i) & hard[indices]).sum()) for i, label in enumerate(LABELS)},
        "row_exposure_counts": exposure.tolist(),
    }
    if all("semantic_component" in row for row in rows):
        report["component_draws"] = dict(Counter(rows[i]["semantic_component"] for i in indices))
    return indices, report


def _feature_array(features, row_count):
    values = np.asarray(features, dtype=np.float32)
    if values.shape != (row_count, 768) or not np.isfinite(values).all():
        raise CorpusError("Finite row-aligned 768-dimensional features are required.")
    return values


def _head_weights(linear):
    return (linear.weight.detach().cpu().numpy().copy(),
            linear.bias.detach().cpu().numpy().copy())


def fit_head_v3(features, rows, tune_features, tune, protocol, seed, *, fixed_epoch=None, deadline=None):
    """Fit the registered 20/40/80 head selection plan, or replay TRAIN only."""
    started = time.perf_counter()
    _roles(rows, tune, fixed_epoch)
    _check_deadline(deadline, "head_start")
    config = protocol["head"]
    epochs = _epoch_count(config["epochs"], fixed_epoch)
    selection_epochs = set(config["selection_epochs"])
    if fixed_epoch is None and not any(1 <= epoch <= epochs for epoch in selection_epochs):
        raise CorpusError("The head plan contains no selectable epoch.")
    torch = _torch(protocol, seed)
    x = torch.as_tensor(_feature_array(features, len(rows)))
    y = torch.as_tensor(_training_targets(rows))
    dev = None if fixed_epoch is not None else _feature_array(tune_features, len(tune))
    linear = torch.nn.Linear(768, 3)
    dropout = torch.nn.Dropout(config["dropout"])
    optimizer = torch.optim.AdamW(linear.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = None
    weights = None
    history = []
    steps = 0
    for epoch in range(1, epochs + 1):
        _check_deadline(deadline, "head_epoch")
        total = 0.
        dropout.train()
        for ids in torch.randperm(len(rows), generator=generator).split(config["batch_size"]):
            _check_deadline(deadline, "head_batch")
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(linear(dropout(x[ids])), y[ids])
            if not torch.isfinite(loss):
                raise CorpusError("Non-finite head training loss.")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            steps += 1
            _check_deadline(deadline, "head_batch_finished")
        record = {"epoch": epoch, "train_cross_entropy": total / len(rows)}
        if fixed_epoch is not None:
            if epoch == epochs:
                best, weights = dict(record), _head_weights(linear)
        elif epoch in selection_epochs:
            _check_deadline(deadline, "head_tune")
            current = _head_weights(linear)
            record["tune"] = tuning_score(tune, softmax(dev @ current[0].T + current[1]))
            _check_deadline(deadline, "head_tune_finished")
            if meaningful_improvement(record["tune"], None if best is None else best["tune"]):
                best, weights = dict(record), current
        history.append(record)
    _check_deadline(deadline, "head_finished")
    return weights, {
        "selected": best, "history": history, "seed": seed,
        "scope": "TRAIN_ONLY_FIXED_EPOCH_REPLAY" if fixed_epoch is not None else "TRAIN_OPTIMIZATION_TUNE_SELECTION",
        "selection_retuned": fixed_epoch is None,
        "tune_inference_executed": fixed_epoch is None,
        "planned_epochs": config["epochs"], "executed_epochs": epochs,
        "stop_reason": "FIXED_EPOCHS" if fixed_epoch is not None else "MAX_EPOCHS",
        "stopped_early": False, "optimizer_steps": steps,
        "fit_wall_seconds": time.perf_counter() - started,
        "encoder_trainable_parameters": 0, "head_trainable_parameters": 2307,
        "device": "cpu", "threads": 2,
    }


def train_encoder_v3(encoder, rows, tune, weights, config, protocol, seed, state_path,
                     *, fixed_epoch=None, deadline=None):
    """Train authorized blocks, stop on tune patience, and restore a paired best.

    A fixed replay executes no tune encoding/scoring. Adaptations are staged in
    a temporary safetensors file and published only after successful validation.
    """
    started = time.perf_counter()
    _roles(rows, tune, fixed_epoch)
    _check_deadline(deadline, "encoder_start")
    epochs = _epoch_count(config["epochs"], fixed_epoch)
    patience = config["patience"]
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise CorpusError("Positive registered encoder patience is required.")
    if config["mode"] not in ("top2", "full"):
        raise CorpusError("V3 encoder optimization requires top2 or full mode.")
    state_path = Path(state_path)
    if state_path.exists():
        raise CorpusError("An adaptation checkpoint already exists; do not overwrite.")
    torch = _torch(protocol, seed)
    from safetensors.torch import load_file, save_file
    from transformers import get_linear_schedule_with_warmup

    if any(parameter.device.type != "cpu" for parameter in encoder.model.parameters()):
        raise CorpusError("The registered V3 pilot requires a CPU encoder.")
    allowed = _allowed_keys(encoder.model, config["mode"])
    encoder.model.zero_grad(set_to_none=True)
    for name, parameter in encoder.model.named_parameters():
        parameter.requires_grad_(name in allowed)
    trainable = [parameter for parameter in encoder.model.parameters() if parameter.requires_grad]
    if not trainable:
        raise CorpusError("Encoder optimization has no authorized trainable parameters.")
    first = trainable[0]
    before = first.detach().clone()
    pending = state_path.with_name(f"{state_path.name}.{uuid.uuid4().hex}.pending")
    try:
        linear = torch.nn.Linear(768, 3)
        initial_weight, initial_bias = np.asarray(weights[0]), np.asarray(weights[1])
        if (initial_weight.shape != (3, 768) or initial_bias.shape != (3,)
                or not np.isfinite(initial_weight).all() or not np.isfinite(initial_bias).all()):
            raise CorpusError("Finite canonical head initialization is required.")
        with torch.no_grad():
            linear.weight.copy_(torch.as_tensor(initial_weight))
            linear.bias.copy_(torch.as_tensor(initial_bias))
        dropout = torch.nn.Dropout(config["head_dropout"])
        optimizer = torch.optim.AdamW([
            {"params": trainable, "lr": config["encoder_lr"]},
            {"params": linear.parameters(), "lr": config["head_lr"]},
        ], weight_decay=config["weight_decay"])
        planned_steps = config["epochs"] * math.ceil(len(rows) / protocol["finetune_batch_size"])
        warmup_steps = int(planned_steps * protocol["warmup_fraction"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, planned_steps)
        y = torch.as_tensor(_training_targets(rows))
        best = None
        selected_weights = None
        history = []
        stale_epochs = 0
        first_gradient = None
        nonzero_gradient_observed = False
        steps = 0
        stop_reason = "FIXED_EPOCHS" if fixed_epoch is not None else "MAX_EPOCHS"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs + 1):
            _check_deadline(deadline, "encoder_epoch")
            encoder.model.eval()
            if config["mode"] == "full":
                encoder.model.train()
            else:
                for index in (10, 11):
                    encoder.model.encoder.layer[index].train()
            dropout.train()
            order, sampling = sample_train_epoch(rows, protocol, seed, epoch, hard_weight=config.get("hard_weight", 1.))
            total = 0.
            gradient_norms = []
            for ids in torch.as_tensor(order).split(protocol["finetune_batch_size"]):
                _check_deadline(deadline, "encoder_batch")
                optimizer.zero_grad(set_to_none=True)
                features = encoder.pooled(encoder.tokens([rows[i]["text"] for i in ids.tolist()]))
                loss = torch.nn.functional.cross_entropy(linear(dropout(features)), y[ids],
                                                         label_smoothing=config["label_smoothing"])
                if not torch.isfinite(loss):
                    raise CorpusError("Non-finite encoder training loss.")
                loss.backward()
                if first.grad is None:
                    raise CorpusError("The authorized encoder probe received no gradient.")
                probe_gradient = float(torch.linalg.vector_norm(first.grad.detach()))
                if first_gradient is None:
                    first_gradient = probe_gradient
                nonzero_gradient_observed |= probe_gradient > 0
                norm = torch.nn.utils.clip_grad_norm_(trainable + list(linear.parameters()),
                                                     protocol["gradient_clip_norm"], error_if_nonfinite=True)
                gradient_norms.append(float(norm))
                optimizer.step()
                scheduler.step()
                steps += 1
                total += float(loss.detach()) * len(ids)
                _check_deadline(deadline, "encoder_batch_finished")
            encoder.model.eval()
            dropout.eval()
            current = _head_weights(linear)
            record = {
                "epoch": epoch, "train_cross_entropy_smoothed_if_configured": total / len(rows),
                "sampling": sampling, "gradient_norm_mean_before_clip": float(np.mean(gradient_norms)),
                "last_encoder_lr": optimizer.param_groups[0]["lr"],
                "last_head_lr": optimizer.param_groups[1]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            if fixed_epoch is None:
                _check_deadline(deadline, "encoder_tune")
                tune_features = encoder.encode([row["text"] for row in tune])
                record["tune"] = tuning_score(tune, softmax(tune_features @ current[0].T + current[1]))
                _check_deadline(deadline, "encoder_tune_finished")
                improved = meaningful_improvement(record["tune"], None if best is None else best["tune"])
            else:
                improved = epoch == epochs
            history.append(record)
            if improved:
                best, selected_weights = dict(record), current
                save_file({key: value.detach().cpu().contiguous()
                           for key, value in encoder.model.state_dict().items() if key in allowed}, str(pending))
                _check_deadline(deadline, "encoder_checkpoint_finished")
                stale_epochs = 0
            elif fixed_epoch is None:
                stale_epochs += 1
                if stale_epochs >= patience:
                    stop_reason = "PATIENCE"
                    break
        _check_deadline(deadline, "encoder_restore")
        encoder.model.load_state_dict(load_file(str(pending), device="cpu"), strict=False)
        update = float(torch.linalg.vector_norm(first.detach() - before))
        if not math.isfinite(update) or update <= 0 or not nonzero_gradient_observed:
            raise CorpusError("A real encoder gradient and parameter update were not verified.")
        frozen_gradients_absent = all(parameter.grad is None for name, parameter in encoder.model.named_parameters()
                                      if name not in allowed)
        if not frozen_gradients_absent:
            raise CorpusError("An encoder gradient escaped the authorized blocks.")
        adaptation_sha = file_sha(pending)
        _check_deadline(deadline, "encoder_publish")
        pending.replace(state_path)
        return selected_weights, {
            "selected": best, "history": history, "seed": seed,
            "scope": "TRAIN_ONLY_FIXED_EPOCH_REPLAY" if fixed_epoch is not None else "TRAIN_OPTIMIZATION_TUNE_SELECTION",
            "selection_retuned": fixed_epoch is None,
            "tune_inference_executed": fixed_epoch is None,
            "planned_epochs": config["epochs"], "executed_epochs": len(history),
            "stop_reason": stop_reason, "stopped_early": len(history) < epochs,
            "patience": patience, "optimizer_steps": steps,
            "scheduler_total_steps": planned_steps, "scheduler_warmup_steps": warmup_steps,
            "fit_wall_seconds": time.perf_counter() - started,
            "encoder_trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "head_trainable_parameters": 2307, "first_encoder_gradient_norm": first_gradient,
            "nonzero_encoder_gradient_observed": nonzero_gradient_observed,
            "probe_tensor_l2_update": update,
            "frozen_parameter_gradients_absent": frozen_gradients_absent,
            "adaptation_sha256": adaptation_sha, "device": "cpu", "threads": 2,
        }
    finally:
        pending.unlink(missing_ok=True)
        for parameter in encoder.model.parameters():
            parameter.requires_grad_(False)
