"""V4.3 classical LR/SVM: grouped RandomizedSearchCV, TF-IDF grids, calibration.

Also trains frozen-BETO embedding + LR/SVM variants as separate artifacts.
Does not overwrite V4.2 BETO partial256. Never fits on independent eval.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .corpus import CorpusError, file_sha
from .dataset_v4_training import load_v4_corpus, partition
from .labels import LABELS
from .synthetic_diversity import ignored_target, write_json


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "ABSENT"


def _full_metrics(truth: list[str], pred: list[str], *, latency_ms: float | None = None) -> dict:
    prec, rec, f1, support = precision_recall_fscore_support(
        truth, pred, labels=list(LABELS), zero_division=0)
    per = {
        label: {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    neg_idx = list(LABELS).index("negativo")
    n_neg = max(sum(1 for t in truth if t == "negativo"), 1)
    neg_to_pos = sum(1 for t, p in zip(truth, pred) if t == "negativo" and p == "positivo")
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(f1_score(truth, pred, labels=list(LABELS), average="macro", zero_division=0)),
        "neg_recall": float(rec[neg_idx]),
        "per_class": per,
        "classification_report": classification_report(
            truth, pred, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(truth, pred, labels=list(LABELS)).tolist(),
        "neg_to_pos": neg_to_pos,
        "neg_to_pos_rate": neg_to_pos / n_neg,
        "mean_latency_ms": latency_ms,
    }


def _goals(m: dict) -> dict:
    return {
        "accuracy>=0.90": m["accuracy"] >= 0.90,
        "macro_f1>=0.90": m["macro_f1"] >= 0.90,
        "macro_precision>=0.90": m["macro_precision"] >= 0.90,
        "macro_recall>=0.90": m["macro_recall"] >= 0.90,
        "neg_recall>=0.95": m["neg_recall"] >= 0.95,
        "each_f1>=0.85": all(m["per_class"][c]["f1"] >= 0.85 for c in LABELS),
        "neg_to_pos_rate<=0.05": m["neg_to_pos_rate"] <= 0.05,
    }


def _tfidf_union():
    return FeatureUnion([
        ("words", TfidfVectorizer(strip_accents="unicode", sublinear_tf=True, min_df=2)),
        ("chars", TfidfVectorizer(analyzer="char_wb", sublinear_tf=True, min_df=2)),
    ])


def _param_distributions(model_kind: str) -> dict:
    base = {
        "tfidf__words__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__words__max_features": [8000, 16000, 32000, None],
        "tfidf__chars__ngram_range": [(3, 4), (3, 5), (4, 6)],
        "tfidf__chars__max_features": [8000, 12000, 20000],
    }
    if model_kind == "lr":
        base["classifier__C"] = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
        base["classifier__solver"] = ["lbfgs", "saga"]
    else:
        base["classifier__C"] = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    return base


def _base_pipeline(model_kind: str) -> Pipeline:
    if model_kind == "lr":
        clf = LogisticRegression(
            max_iter=2500, class_weight="balanced", random_state=42,
        )
    else:
        clf = LinearSVC(class_weight="balanced", random_state=42, dual="auto")
    return Pipeline([("tfidf", _tfidf_union()), ("classifier", clf)])


def train_tfidf_search(
    *,
    input_dir: Path,
    out_dir: Path,
    model_kind: str,
    n_iter: int = 18,
    seed: int = 42,
) -> dict:
    if model_kind not in {"lr", "svm"}:
        raise CorpusError("model_kind debe ser lr o svm.")
    started = time.perf_counter()
    rows, info = load_v4_corpus(input_dir)
    parts = partition(rows)
    destination = ignored_target(Path(out_dir), kinds=("artifacts",))
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / f"{model_kind}.joblib"
    report_path = destination / f"{model_kind}_report.json"
    if marker.exists() or report_path.exists():
        raise CorpusError(f"Artefacto V4.3 {model_kind} ya existe.")

    # Fit search on train only; groups = scenario_family (no family leak across folds).
    train = parts["train"]
    X = [r["text"] for r in train]
    y = [r["intended_label"] for r in train]
    groups = [r["scenario_family"] for r in train]

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    pipe = _base_pipeline(model_kind)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=_param_distributions(model_kind),
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=seed,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(X, y, groups=groups)
    best_estimator = search.best_estimator_

    # Calibration on grouped folds over train (TF-IDF inside estimator via wrapping whole pipeline).
    cal_folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + 1)
                     .split(X, y, groups))
    calibrated = CalibratedClassifierCV(
        estimator=best_estimator, method="sigmoid", cv=cal_folds,
    )
    calibrated.fit(X, y)

    joblib.dump(calibrated, marker)
    train_wall = time.perf_counter() - started

    def evaluate(name: str, subset: list[dict]) -> dict:
        texts = [r["text"] for r in subset]
        labels = [r["intended_label"] for r in subset]
        t0 = time.perf_counter()
        pred = [str(x) for x in calibrated.predict(texts)]
        lat = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
        metrics = _full_metrics(labels, pred, latency_ms=lat)
        return {
            "split": name,
            "n": len(subset),
            "label_counts": dict(Counter(labels)),
            "metrics": metrics,
            "goals": _goals(metrics),
            "goals_met": sum(_goals(metrics).values()),
        }

    report = {
        "schema": "traveler-v4.3-classical-report",
        "model_kind": model_kind,
        "feature_family": "tfidf_words_chars",
        "corpus_rows_sha256": info["rows_sha256"],
        "protocol_sha256": info["protocol_sha256"],
        "train_rows": len(train),
        "train_wall_seconds": round(train_wall, 2),
        "search": {
            "method": "RandomizedSearchCV",
            "n_iter": n_iter,
            "cv": "StratifiedGroupKFold(n_splits=3)",
            "best_params": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in search.best_params_.items()},
            "best_cv_macro_f1": float(search.best_score_),
        },
        "calibration": {
            "method": "sigmoid",
            "cv": "StratifiedGroupKFold(n_splits=3) on train",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit-learn": _pkg("scikit-learn"),
            "numpy": _pkg("numpy"),
            "joblib": _pkg("joblib"),
        },
        "validation": evaluate("validation", parts["validation"]),
        "test": evaluate("test", parts["test"]),
        "hard_holdout": evaluate("hard_holdout", parts["hard_holdout"]),
        "human_gold": False,
        "historical_artifacts_preserved": True,
        "beto_v42_partial256_untouched": True,
        "limitations": [
            "Synthetic generator intent labels only.",
            "Hyperparameters chosen on train grouped CV; validation/test are in-generator.",
            "Independent eval must be scored separately and never used for fitting.",
        ],
    }
    write_json(report_path, report)
    (destination / f"{model_kind}.meta.json").write_text(
        json.dumps({
            "model_kind": model_kind,
            "feature_family": "tfidf_words_chars",
            "corpus_rows_sha256": info["rows_sha256"],
            "report_sha256": file_sha(report_path),
            "labels": list(LABELS),
            "best_params": report["search"]["best_params"],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _encode_frozen_beto(checkpoint: Path, texts: list[str], *, max_length: int = 256, batch_size: int = 8):
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
        use_safetensors=False, weights_only=True, add_pooling_layer=False,
    )
    model.to(device).eval()
    out = []
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
        raise CorpusError("Embeddings BETO congelados inválidos.")
    return values, device


def train_embedding_classical(
    *,
    input_dir: Path,
    out_dir: Path,
    checkpoint: Path,
    model_kind: str,
    max_length: int = 256,
    seed: int = 42,
) -> dict:
    """Frozen BETO mean-pool embeddings + calibrated LR or LinearSVC."""
    if model_kind not in {"lr", "svm"}:
        raise CorpusError("model_kind debe ser lr o svm.")
    started = time.perf_counter()
    rows, info = load_v4_corpus(input_dir)
    parts = partition(rows)
    destination = ignored_target(Path(out_dir), kinds=("artifacts",))
    destination.mkdir(parents=True, exist_ok=True)
    tag = f"beto_emb_{model_kind}"
    marker = destination / f"{tag}.joblib"
    report_path = destination / f"{tag}_report.json"
    emb_path = destination / f"{tag}_train_embeddings.npz"
    if marker.exists() or report_path.exists():
        raise CorpusError(f"Artefacto embedding {tag} ya existe.")

    train = parts["train"]
    X_text = [r["text"] for r in train]
    y = [r["intended_label"] for r in train]
    groups = [r["scenario_family"] for r in train]

    print(f"[v43-emb] encoding {len(X_text)} train texts…", flush=True)
    X, device = _encode_frozen_beto(Path(checkpoint), X_text, max_length=max_length)
    np.savez_compressed(emb_path, X=X, y=np.array(y))

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups))
    if model_kind == "lr":
        # Grid C on grouped CV macro-F1
        best_score, best_C, best_clf = -1.0, 1.0, None
        for C in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
            scores = []
            for tr, va in folds:
                clf = LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed, C=C,
                )
                clf.fit(X[tr], [y[i] for i in tr])
                pred = clf.predict(X[va])
                scores.append(f1_score(
                    [y[i] for i in va], pred, labels=list(LABELS), average="macro", zero_division=0))
            score = float(np.mean(scores))
            if score > best_score:
                best_score, best_C = score, C
        inner = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed, C=best_C,
        )
        selected = {"C": best_C, "cv_macro_f1": best_score}
    else:
        best_score, best_C = -1.0, 1.0
        for C in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
            scores = []
            for tr, va in folds:
                clf = LinearSVC(class_weight="balanced", random_state=seed, C=C, dual="auto")
                clf.fit(X[tr], [y[i] for i in tr])
                pred = clf.predict(X[va])
                scores.append(f1_score(
                    [y[i] for i in va], pred, labels=list(LABELS), average="macro", zero_division=0))
            score = float(np.mean(scores))
            if score > best_score:
                best_score, best_C = score, C
        inner = LinearSVC(class_weight="balanced", random_state=seed, C=best_C, dual="auto")
        selected = {"C": best_C, "cv_macro_f1": best_score}

    calibrated = CalibratedClassifierCV(estimator=inner, method="sigmoid", cv=folds)
    calibrated.fit(X, y)
    # Bundle encoder path + classifier for inference
    bundle = {
        "classifier": calibrated,
        "checkpoint": str(Path(checkpoint).resolve()),
        "max_length": max_length,
        "model_kind": model_kind,
        "feature_family": "frozen_beto_mean_pool",
        "classes": list(LABELS),
    }
    joblib.dump(bundle, marker)
    train_wall = time.perf_counter() - started

    def evaluate(name: str, subset: list[dict]) -> dict:
        texts = [r["text"] for r in subset]
        labels = [r["intended_label"] for r in subset]
        emb, _ = _encode_frozen_beto(Path(checkpoint), texts, max_length=max_length)
        t0 = time.perf_counter()
        pred = [str(x) for x in calibrated.predict(emb)]
        lat = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
        # Note: latency above is classifier-only; add encode separately if needed.
        metrics = _full_metrics(labels, pred, latency_ms=lat)
        return {
            "split": name,
            "n": len(subset),
            "label_counts": dict(Counter(labels)),
            "metrics": metrics,
            "goals": _goals(metrics),
            "goals_met": sum(_goals(metrics).values()),
        }

    report = {
        "schema": "traveler-v4.3-embedding-classical-report",
        "model_kind": model_kind,
        "feature_family": "frozen_beto_mean_pool",
        "checkpoint": str(Path(checkpoint)),
        "max_length": max_length,
        "device_used": device,
        "corpus_rows_sha256": info["rows_sha256"],
        "protocol_sha256": info["protocol_sha256"],
        "train_rows": len(train),
        "train_wall_seconds": round(train_wall, 2),
        "hyperparameter_selection": selected,
        "calibration": {"method": "sigmoid", "cv": "StratifiedGroupKFold(n_splits=3)"},
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": _pkg("scikit-learn"),
            "torch": _pkg("torch"),
            "transformers": _pkg("transformers"),
        },
        "validation": evaluate("validation", parts["validation"]),
        "test": evaluate("test", parts["test"]),
        "hard_holdout": evaluate("hard_holdout", parts["hard_holdout"]),
        "human_gold": False,
        "separate_from_tfidf_classicals": True,
        "beto_v42_partial256_untouched": True,
    }
    write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="V4.3 classical + embedding classical training")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tf = sub.add_parser("train-tfidf")
    p_tf.add_argument("--input", type=Path, default=Path("data/processed/synthetic-tourism-v4.3"))
    p_tf.add_argument("--out", type=Path, default=Path("artifacts/synthetic-tourism-v4.3"))
    p_tf.add_argument("--model-kind", choices=("lr", "svm"), required=True)
    p_tf.add_argument("--n-iter", type=int, default=18)

    p_emb = sub.add_parser("train-embedding")
    p_emb.add_argument("--input", type=Path, default=Path("data/processed/synthetic-tourism-v4.3"))
    p_emb.add_argument("--out", type=Path, default=Path("artifacts/synthetic-tourism-v4.3"))
    p_emb.add_argument("--checkpoint", type=Path, default=Path("artifacts/beto-v3-pr28-local/checkpoint"))
    p_emb.add_argument("--model-kind", choices=("lr", "svm"), required=True)
    p_emb.add_argument("--max-length", type=int, default=256)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "train-tfidf":
            report = train_tfidf_search(
                input_dir=args.input, out_dir=args.out,
                model_kind=args.model_kind, n_iter=args.n_iter,
            )
            print(json.dumps({
                "status": "TRAINED_TFIDF",
                "model_kind": args.model_kind,
                "best_cv_macro_f1": report["search"]["best_cv_macro_f1"],
                "val_macro_f1": report["validation"]["metrics"]["macro_f1"],
                "test_macro_f1": report["test"]["metrics"]["macro_f1"],
                "train_wall_s": report["train_wall_seconds"],
            }, ensure_ascii=False, indent=2))
        else:
            report = train_embedding_classical(
                input_dir=args.input, out_dir=args.out,
                checkpoint=args.checkpoint, model_kind=args.model_kind,
                max_length=args.max_length,
            )
            print(json.dumps({
                "status": "TRAINED_EMB",
                "model_kind": args.model_kind,
                "cv_macro_f1": report["hyperparameter_selection"]["cv_macro_f1"],
                "val_macro_f1": report["validation"]["metrics"]["macro_f1"],
                "test_macro_f1": report["test"]["metrics"]["macro_f1"],
                "train_wall_s": report["train_wall_seconds"],
            }, ensure_ascii=False, indent=2))
    except (CorpusError, ValueError, OSError) as exc:
        parser.exit(2, f"V4.3 training blocked: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
