"""BETO V4 training: frozen128/256 heads and optional last-two-block partial FT.

Does not modify V3 artifacts. Uses CUDA when available (RTX 4050 6 GB budget).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .corpus import CorpusError
from .dataset_v4_training import load_v4_corpus, partition
from .labels import LABELS
from .synthetic_diversity import ignored_target, write_json
from .synthetic_tourism_v4 import protocol


def load_plan(plan_path: Path) -> dict:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("schema") != "traveler-beto-v4-training-plan":
        raise CorpusError("Plan BETO V4 inválido.")
    return plan


def _device_report():
    import torch
    cuda = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if cuda else None
    vram = None
    if cuda:
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    return {
        "torch": torch.__version__,
        "cuda_available": cuda,
        "device_name": name,
        "vram_gb": vram,
        "device": "cuda" if cuda else "cpu",
    }


def _load_raw_encoder(checkpoint: Path, max_length: int, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
        use_safetensors=False, weights_only=True, add_pooling_layer=False,
    )
    model.to(device)
    model.eval()
    return tok, model, max_length, device


def _encode_texts(tok, model, texts, *, max_length: int, device: str, batch_size: int = 8):
    import torch
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.detach().cpu().numpy())
    values = np.concatenate(out).astype(np.float32)
    if values.shape != (len(texts), 768) or not np.isfinite(values).all():
        raise CorpusError("Embeddings V4 inválidos.")
    return values


def train_frozen_head(
    *,
    input_dir: Path,
    plan_path: Path,
    checkpoint: Path,
    out_dir: Path,
    max_length: int = 256,
    seed: int | None = None,
    threads: int = 2,
    batch_size: int = 8,
) -> dict:
    """Train a frozen-encoder logistic head on V4 train; score val/test/hard."""
    seed = protocol()["seed"] if seed is None else seed
    started = time.perf_counter()
    hw = _device_report()
    print(f"[beto-v4] device={hw}", flush=True)
    print(f"[beto-v4] loading corpus from {input_dir}", flush=True)
    rows, info = load_v4_corpus(input_dir)
    parts = partition(rows)
    plan = load_plan(plan_path)
    if plan["corpus_rows_sha256"] != info["rows_sha256"]:
        raise CorpusError("Plan desalineado del corpus V4.")
    if max_length not in plan["candidate_max_lengths"]:
        raise CorpusError("max_length no prerregistrado.")

    destination = ignored_target(Path(out_dir), kinds=("artifacts",))
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"beto_v4_frozen{max_length}_report.json"
    artifact = destination / f"beto-v4-frozen{max_length}"
    if report_path.exists() or artifact.exists():
        raise CorpusError("Artefacto/informe BETO V4 ya existe.")

    from .beto_classifier import export_classifier
    from .research_metrics import summarize_predictions
    import torch
    torch.set_num_threads(threads)

    tok, model, max_length, device = _load_raw_encoder(Path(checkpoint), max_length, hw["device"])
    train_texts = [r["text"] for r in parts["train"]]
    train_labels = [r["intended_label"] for r in parts["train"]]
    feature_cache = destination / f"beto_v4_train_features_{max_length}.npz"
    cache_meta = feature_cache.with_suffix(".json")
    encode_t0 = time.perf_counter()
    if feature_cache.is_file() and cache_meta.is_file():
        meta_cache = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta_cache.get("corpus_rows_sha256") != info["rows_sha256"]:
            raise CorpusError("Caché de features de otro corpus.")
        print(f"[beto-v4] reusing cached train features {feature_cache}", flush=True)
        features = np.load(feature_cache)["features"]
        encode_train_s = 0.0
    else:
        print(f"[beto-v4] encoding train n={len(train_texts)} on {device}", flush=True)
        features = _encode_texts(tok, model, train_texts, max_length=max_length, device=device, batch_size=batch_size)
        np.savez(feature_cache, features=features)
        cache_meta.write_text(
            json.dumps({"corpus_rows_sha256": info["rows_sha256"], "max_length": max_length, "device": device}, indent=2) + "\n",
            encoding="utf-8",
        )
        encode_train_s = time.perf_counter() - encode_t0
        print(f"[beto-v4] cached train features -> {feature_cache}", flush=True)

    fit_t0 = time.perf_counter()
    head = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    head.fit(features, train_labels)
    fit_s = time.perf_counter() - fit_t0
    class_index = [list(head.classes_).index(label) for label in LABELS]
    weight = np.asarray(head.coef_[class_index], dtype=np.float32)
    bias = np.asarray(head.intercept_[class_index], dtype=np.float32)

    export_classifier(
        artifact, Path(checkpoint), weight, bias,
        training={
            "supervised": True,
            "experiment": "BETO_V4_FROZEN_LOGREG_HEAD",
            "max_length": max_length,
            "seed": seed,
            "corpus_rows_sha256": info["rows_sha256"],
            "protocol_sha256": info["protocol_sha256"],
            "label_origin": protocol()["label_origin"],
            "head": "sklearn_logistic_regression_on_frozen_mean_l2",
            "device": device,
        },
    )
    meta_path = artifact / "model_manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["max_length"] = max_length
    meta["pooling"] = "masked_mean_including_special_tokens_l2"
    meta["v4_note"] = "max_length set for V4 after truncation audit; schema remains supervised-v1 export."
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    W, b = weight, bias

    def eval_split(name: str, subset: list[dict]) -> dict:
        texts = [r["text"] for r in subset]
        truth = [r["intended_label"] for r in subset]
        print(f"[beto-v4] encoding {name} n={len(texts)}", flush=True)
        t0 = time.perf_counter()
        feats = _encode_texts(tok, model, texts, max_length=max_length, device=device, batch_size=batch_size)
        pred = [LABELS[i] for i in (feats @ W.T + b).argmax(axis=1)]
        latency_ms = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
        return {"split": name, "metrics": summarize_predictions(truth, pred), "mean_latency_ms": latency_ms}

    report = {
        "schema": "traveler-beto-v4-frozen-report",
        "status": "TRAINED_FROZEN_HEAD",
        "max_length": max_length,
        "seed": seed,
        "corpus_rows_sha256": info["rows_sha256"],
        "artifact": str(artifact),
        "hardware": hw,
        "timing": {
            "encode_train_seconds": encode_train_s,
            "fit_head_seconds": fit_s,
            "wall_seconds": time.perf_counter() - started,
        },
        "validation": eval_split("validation", parts["validation"]),
        "test": eval_split("test", parts["test"]),
        "hard_holdout": eval_split("hard_holdout", parts["hard_holdout"]),
        "human_gold": False,
        "historical_v3_preserved": True,
    }
    write_json(report_path, report)
    return report


def train_partial256(
    *,
    input_dir: Path,
    plan_path: Path,
    checkpoint: Path,
    out_dir: Path,
    seed: int | None = None,
    epochs: int = 2,
    batch_size: int = 2,
    threads: int = 2,
) -> dict:
    """Real last-two-block adaptation at max_length=256 under 6 GB VRAM budget."""
    seed = protocol()["seed"] if seed is None else seed
    max_length = 256
    started = time.perf_counter()
    hw = _device_report()
    if not hw["cuda_available"]:
        raise CorpusError("partial256 requiere CUDA viable en este hardware.")
    rows, info = load_v4_corpus(input_dir)
    parts = partition(rows)
    plan = load_plan(plan_path)
    if plan["corpus_rows_sha256"] != info["rows_sha256"]:
        raise CorpusError("Plan desalineado del corpus V4.")

    destination = ignored_target(Path(out_dir), kinds=("artifacts",))
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "beto_v4_partial256_report.json"
    artifact = destination / "beto-v4-partial256"
    if report_path.exists() or artifact.exists():
        raise CorpusError("Artefacto/informe partial256 ya existe.")

    import torch
    from .beto_classifier import export_classifier
    from .research_metrics import summarize_predictions

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    device = "cuda"
    tok, model, max_length, device = _load_raw_encoder(Path(checkpoint), max_length, device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("encoder.layer.10.") or name.startswith("encoder.layer.11."))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise CorpusError("No hay parámetros entrenables en los últimos bloques.")

    # Initialize head from a quick frozen encode of a stratified subsample for speed.
    sample = []
    for label in LABELS:
        pool = [r for r in parts["train"] if r["intended_label"] == label]
        sample.extend(pool[:80])
    init_feats = _encode_texts(tok, model, [r["text"] for r in sample], max_length=max_length, device=device, batch_size=4)
    init_head = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    init_head.fit(init_feats, [r["intended_label"] for r in sample])
    class_index = [list(init_head.classes_).index(label) for label in LABELS]
    linear = torch.nn.Linear(768, 3).to(device)
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(init_head.coef_[class_index], dtype=torch.float32, device=device))
        linear.bias.copy_(torch.as_tensor(init_head.intercept_[class_index], dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(
        [{"params": trainable, "lr": 1e-5}, {"params": linear.parameters(), "lr": 5e-4}],
        weight_decay=0.01,
    )
    train_rows = parts["train"]
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    history = []
    peak_vram = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for index in (10, 11):
            model.encoder.layer[index].train()
        linear.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
        total = 0.0
        steps = 0
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            batch = [train_rows[i] for i in ids]
            enc = tok([r["text"] for r in batch], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            y = torch.as_tensor([label_to_id[r["intended_label"]] for r in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            loss = torch.nn.functional.cross_entropy(linear(pooled), y)
            if not torch.isfinite(loss):
                raise CorpusError("Loss no finita en partial256.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable + list(linear.parameters()), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            steps += 1
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024 ** 2)
        history.append({"epoch": epoch, "train_loss": total / len(train_rows), "steps": steps})
        print(f"[beto-v4] partial256 epoch {epoch} loss={history[-1]['train_loss']:.4f} peak_mib={peak_vram:.1f}", flush=True)

    adaptation = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
                  if k.startswith("encoder.layer.10.") or k.startswith("encoder.layer.11.")}
    weight = linear.weight.detach().cpu().numpy().astype(np.float32)
    bias = linear.bias.detach().cpu().numpy().astype(np.float32)
    export_classifier(
        artifact, Path(checkpoint), weight, bias,
        training={
            "supervised": True,
            "experiment": "BETO_V4_PARTIAL256_LAST_TWO_BLOCKS",
            "max_length": max_length,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "corpus_rows_sha256": info["rows_sha256"],
            "protocol_sha256": info["protocol_sha256"],
            "label_origin": protocol()["label_origin"],
            "device": device,
            "peak_vram_mib": peak_vram,
            "history": history,
        },
        adaptation=adaptation,
    )
    meta_path = artifact / "model_manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["max_length"] = max_length
    meta["mode"] = "partial256"
    meta["adapted_layer_range"] = [10, 11]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    model.eval()
    linear.eval()

    def eval_split(name: str, subset: list[dict]) -> dict:
        texts = [r["text"] for r in subset]
        truth = [r["intended_label"] for r in subset]
        t0 = time.perf_counter()
        # Encode with adapted weights still in `model`.
        feats = _encode_texts(tok, model, texts, max_length=max_length, device=device, batch_size=4)
        pred = [LABELS[i] for i in (feats @ weight.T + bias).argmax(axis=1)]
        return {
            "split": name,
            "metrics": summarize_predictions(truth, pred),
            "mean_latency_ms": (time.perf_counter() - t0) * 1000 / max(len(texts), 1),
        }

    report = {
        "schema": "traveler-beto-v4-partial256-report",
        "status": "TRAINED_PARTIAL256",
        "max_length": max_length,
        "seed": seed,
        "epochs": epochs,
        "corpus_rows_sha256": info["rows_sha256"],
        "artifact": str(artifact),
        "hardware": hw,
        "peak_vram_mib": peak_vram,
        "history": history,
        "timing": {"wall_seconds": time.perf_counter() - started},
        "validation": eval_split("validation", parts["validation"]),
        "test": eval_split("test", parts["test"]),
        "hard_holdout": eval_split("hard_holdout", parts["hard_holdout"]),
        "human_gold": False,
        "historical_v3_preserved": True,
    }
    write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entrenar BETO V4 (frozen / partial)")
    parser.add_argument("--mode", choices=("frozen", "partial256"), default="frozen")
    parser.add_argument("--input", type=Path, default=Path("data/processed/synthetic-tourism-v4.1"))
    parser.add_argument("--plan", type=Path, default=Path("artifacts/synthetic-tourism-v4.1/beto_v4_training_plan.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/beto-v3-pr28-local/checkpoint"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/synthetic-tourism-v4.1"))
    parser.add_argument("--max-length", type=int, default=256, choices=(128, 256))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.mode == "frozen":
            report = train_frozen_head(
                input_dir=args.input, plan_path=args.plan, checkpoint=args.checkpoint,
                out_dir=args.out, max_length=args.max_length, seed=args.seed,
                threads=args.threads, batch_size=args.batch_size,
            )
        else:
            report = train_partial256(
                input_dir=args.input, plan_path=args.plan, checkpoint=args.checkpoint,
                out_dir=args.out, seed=args.seed, epochs=args.epochs,
                batch_size=min(args.batch_size, 2), threads=args.threads,
            )
    except (CorpusError, ValueError, OSError) as exc:
        parser.exit(2, f"BETO V4 blocked: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "max_length": report.get("max_length"),
        "validation_macro_f1": report["validation"]["metrics"]["macro_f1"],
        "test_macro_f1": report["test"]["metrics"]["macro_f1"],
        "hard_macro_f1": report["hard_holdout"]["metrics"]["macro_f1"],
        "hard_neg_to_pos": report["hard_holdout"]["metrics"]["negative_to_positive_count"],
        "hardware": report.get("hardware"),
        "timing": report.get("timing"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
