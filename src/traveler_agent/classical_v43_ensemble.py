"""V4.3 soft-vote ensemble: TF-IDF LR + frozen-BETO emb SVM.

Weight grid selected ONLY on corpus validation. Independent eval never used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

from .classical_v43_training import _encode_frozen_beto, _full_metrics, _goals
from .corpus import CorpusError, file_sha
from .dataset_v4_training import load_v4_corpus, partition
from .labels import LABELS
from .synthetic_diversity import ignored_target, write_json


def _proba(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts), list(model.classes_)
    # Should not happen for calibrated models
    raise CorpusError("Modelo sin predict_proba.")


def _emb_proba(bundle, texts):
    clf = bundle["classifier"]
    emb, _ = _encode_frozen_beto(
        Path(bundle["checkpoint"]), texts, max_length=int(bundle["max_length"]))
    return clf.predict_proba(emb), list(clf.classes_)


def train_ensemble(
    *,
    input_dir: Path,
    art_dir: Path,
    out_dir: Path,
) -> dict:
    started = time.perf_counter()
    rows, info = load_v4_corpus(input_dir)
    parts = partition(rows)
    destination = ignored_target(Path(out_dir), kinds=("artifacts",))
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "ensemble_softvote.joblib"
    report_path = destination / "ensemble_softvote_report.json"
    if marker.exists() or report_path.exists():
        raise CorpusError("Ensemble V4.3 ya existe.")

    lr_path = Path(art_dir) / "lr.joblib"
    emb_path = Path(art_dir) / "beto_emb_svm.joblib"
    if not lr_path.is_file() or not emb_path.is_file():
        raise CorpusError("Faltan lr.joblib o beto_emb_svm.joblib.")

    lr = joblib.load(lr_path)
    emb_bundle = joblib.load(emb_path)

    val_texts = [r["text"] for r in parts["validation"]]
    val_labels = [r["intended_label"] for r in parts["validation"]]

    p_lr, classes_lr = _proba(lr, val_texts)
    p_emb, classes_emb = _emb_proba(emb_bundle, val_texts)
    if list(classes_lr) != list(LABELS) or list(classes_emb) != list(LABELS):
        # Align columns to LABELS order
        def reorder(P, classes):
            idx = [list(classes).index(c) for c in LABELS]
            return P[:, idx]
        p_lr = reorder(p_lr, classes_lr)
        p_emb = reorder(p_emb, classes_emb)

    best_w, best_score = 0.5, -1.0
    for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        blended = w * p_lr + (1 - w) * p_emb
        pred = [LABELS[i] for i in np.argmax(blended, axis=1)]
        score = float(f1_score(val_labels, pred, labels=list(LABELS), average="macro", zero_division=0))
        if score > best_score:
            best_score, best_w = score, w

    bundle = {
        "kind": "softvote_tfidf_lr_plus_beto_emb_svm",
        "tfidf_lr_path": str(lr_path.resolve()),
        "emb_svm_path": str(emb_path.resolve()),
        "weight_tfidf": best_w,
        "weight_emb": 1.0 - best_w,
        "classes": list(LABELS),
        "selection_split": "validation",
        "val_macro_f1": best_score,
    }
    # Keep loaded models for single-file inference convenience
    bundle["tfidf_lr"] = lr
    bundle["emb_bundle"] = emb_bundle
    joblib.dump(bundle, marker)

    def predict_texts(texts):
        pl, _ = _proba(lr, texts)
        pe, _ = _emb_proba(emb_bundle, texts)
        # assume LABELS order after checks above
        if list(lr.classes_) != list(LABELS):
            pl = pl[:, [list(lr.classes_).index(c) for c in LABELS]]
        if list(emb_bundle["classifier"].classes_) != list(LABELS):
            pe = pe[:, [list(emb_bundle["classifier"].classes_).index(c) for c in LABELS]]
        blended = best_w * pl + (1 - best_w) * pe
        return [LABELS[i] for i in np.argmax(blended, axis=1)]

    def evaluate(name, subset):
        texts = [r["text"] for r in subset]
        labels = [r["intended_label"] for r in subset]
        t0 = time.perf_counter()
        pred = predict_texts(texts)
        lat = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
        m = _full_metrics(labels, pred, latency_ms=lat)
        return {"split": name, "n": len(subset), "metrics": m, "goals": _goals(m),
                "goals_met": sum(_goals(m).values())}

    report = {
        "schema": "traveler-v4.3-ensemble-report",
        "corpus_rows_sha256": info["rows_sha256"],
        "protocol_sha256": info["protocol_sha256"],
        "train_wall_seconds": round(time.perf_counter() - started, 2),
        "weight_selection": {
            "split": "validation",
            "weight_tfidf": best_w,
            "weight_emb": 1.0 - best_w,
            "val_macro_f1": best_score,
            "independent_eval_not_used": True,
        },
        "validation": evaluate("validation", parts["validation"]),
        "test": evaluate("test", parts["test"]),
        "hard_holdout": evaluate("hard_holdout", parts["hard_holdout"]),
        "beto_v42_partial256_untouched": True,
    }
    write_json(report_path, report)
    (destination / "ensemble_softvote.meta.json").write_text(
        json.dumps({
            "kind": bundle["kind"],
            "weight_tfidf": best_w,
            "report_sha256": file_sha(report_path),
            "corpus_rows_sha256": info["rows_sha256"],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/processed/synthetic-tourism-v4.3"))
    p.add_argument("--art", type=Path, default=Path("artifacts/synthetic-tourism-v4.3"))
    p.add_argument("--out", type=Path, default=Path("artifacts/synthetic-tourism-v4.3"))
    args = p.parse_args(argv)
    try:
        report = train_ensemble(input_dir=args.input, art_dir=args.art, out_dir=args.out)
        print(json.dumps({
            "status": "TRAINED_ENSEMBLE",
            "weight_tfidf": report["weight_selection"]["weight_tfidf"],
            "val_macro_f1": report["validation"]["metrics"]["macro_f1"],
            "test_macro_f1": report["test"]["metrics"]["macro_f1"],
        }, indent=2))
    except (CorpusError, ValueError, OSError) as exc:
        p.exit(2, f"Ensemble blocked: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
