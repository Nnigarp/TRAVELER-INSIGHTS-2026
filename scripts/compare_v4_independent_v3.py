"""Score frozen independent eval v3 once; never fit on it."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from traveler_agent.corpus import CorpusError, text_sha  # noqa: E402
from traveler_agent.labels import LABELS  # noqa: E402
from traveler_agent.synthetic_diversity import canonical_sha, write_json  # noqa: E402

EVAL = ROOT / "src/traveler_agent/research_data/v4_independent_eval_v3.json"


def _metrics(truth, pred, latency_ms):
    prec, rec, f1, support = precision_recall_fscore_support(
        truth, pred, labels=list(LABELS), zero_division=0)
    per = {label: {"precision": float(prec[i]), "recall": float(rec[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
           for i, label in enumerate(LABELS)}
    n_neg = max(sum(1 for t in truth if t == "negativo"), 1)
    neg_to_pos = sum(1 for t, p in zip(truth, pred) if t == "negativo" and p == "positivo")
    return {
        "n": len(truth),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(f1_score(truth, pred, labels=list(LABELS), average="macro", zero_division=0)),
        "neg_recall": float(rec[list(LABELS).index("negativo")]),
        "per_class": per,
        "classification_report": classification_report(
            truth, pred, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(truth, pred, labels=list(LABELS)).tolist(),
        "neg_to_pos": neg_to_pos,
        "neg_to_pos_rate": neg_to_pos / n_neg,
        "mean_latency_ms": latency_ms,
    }


def _goals(m):
    return {
        "accuracy>=0.90": m["accuracy"] >= 0.90,
        "macro_f1>=0.90": m["macro_f1"] >= 0.90,
        "macro_precision>=0.90": m["macro_precision"] >= 0.90,
        "macro_recall>=0.90": m["macro_recall"] >= 0.90,
        "neg_recall>=0.95": m["neg_recall"] >= 0.95,
        "each_f1>=0.85": all(m["per_class"][c]["f1"] >= 0.85 for c in LABELS),
        "neg_to_pos_rate<=0.05": m["neg_to_pos_rate"] <= 0.05,
    }


def _load_eval():
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    if data.get("schema") != "traveler-v4-independent-eval-v3" or data.get("frozen") is not True:
        raise CorpusError("Eval v3 no congelada.")
    reviews = data["reviews"]
    digest = canonical_sha([{k: r[k] for k in ("review_id", "text", "intended_label")} for r in reviews])
    if data.get("records_sha256") != digest:
        raise CorpusError("Hash de eval v3 inconsistente.")
    for row in reviews:
        if text_sha(row["text"]) != row["text_sha256"]:
            raise CorpusError(f"text_sha mismatch {row['review_id']}")
    return data, reviews


def _beto_predict(artifact: Path, texts, max_length: int):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import load_file
    meta = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(artifact / "checkpoint"), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(artifact / "checkpoint"), local_files_only=True,
        use_safetensors=False, weights_only=True, add_pooling_layer=False).eval()
    if (artifact / "adaptation.safetensors").is_file():
        model.load_state_dict(load_file(str(artifact / "adaptation.safetensors"), device="cpu"), strict=False)
    head = np.load(artifact / "head.npz")
    W, b = head["weight"], head["bias"]
    labels = meta["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    t0 = time.perf_counter()
    preds = []
    with torch.inference_mode():
        for text in texts:
            enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()[0]
            preds.append(labels[int(np.argmax(pooled @ W.T + b))])
    return preds, (time.perf_counter() - t0) * 1000 / max(len(texts), 1)


def _emb_bundle_predict(path: Path, texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    bundle = joblib.load(path)
    clf = bundle["classifier"]
    checkpoint = Path(bundle["checkpoint"])
    max_length = int(bundle["max_length"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(checkpoint), local_files_only=True,
        use_safetensors=False, weights_only=True, add_pooling_layer=False).eval().to(device)
    t0 = time.perf_counter()
    embs = []
    with torch.inference_mode():
        for text in texts:
            enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()[0]
            embs.append(pooled)
    pred = [str(x) for x in clf.predict(np.asarray(embs, dtype=np.float32))]
    return pred, (time.perf_counter() - t0) * 1000 / max(len(texts), 1)


def _errors_by_phenomenon(reviews, pred):
    buckets = defaultdict(lambda: Counter())
    for row, p in zip(reviews, pred):
        ok = p == row["intended_label"]
        for ph in row.get("phenomena", []):
            buckets[ph]["n"] += 1
            buckets[ph]["errors" if not ok else "ok"] += 1
    return {k: dict(v) for k, v in sorted(buckets.items())}


def _score_classical(path: Path, texts, labels):
    model = joblib.load(path)
    t0 = time.perf_counter()
    pred = [str(x) for x in model.predict(texts)]
    lat = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
    return pred, lat


def _ensemble_predict(path: Path, texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    bundle = joblib.load(path)
    lr = bundle["tfidf_lr"]
    emb_bundle = bundle["emb_bundle"]
    w = float(bundle["weight_tfidf"])
    clf = emb_bundle["classifier"]
    checkpoint = Path(emb_bundle["checkpoint"])
    max_length = int(emb_bundle["max_length"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(checkpoint), local_files_only=True,
        use_safetensors=False, weights_only=True, add_pooling_layer=False).eval().to(device)
    t0 = time.perf_counter()
    pl = lr.predict_proba(texts)
    if list(lr.classes_) != list(LABELS):
        pl = pl[:, [list(lr.classes_).index(c) for c in LABELS]]
    embs = []
    with torch.inference_mode():
        for text in texts:
            enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()[0]
            embs.append(pooled)
    pe = clf.predict_proba(np.asarray(embs, dtype=np.float32))
    if list(clf.classes_) != list(LABELS):
        pe = pe[:, [list(clf.classes_).index(c) for c in LABELS]]
    blended = w * pl + (1.0 - w) * pe
    pred = [LABELS[i] for i in np.argmax(blended, axis=1)]
    return pred, (time.perf_counter() - t0) * 1000 / max(len(texts), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/research/v4/independent_v3_comparison_2026-11-25.json")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite comparison JSON (allowed after adding a priori ensemble).")
    args = parser.parse_args()
    data, reviews = _load_eval()
    texts = [r["text"] for r in reviews]
    labels = [r["intended_label"] for r in reviews]

    candidates = []
    art43 = ROOT / "artifacts/synthetic-tourism-v4.3"
    art42 = ROOT / "artifacts/synthetic-tourism-v4.2"
    demo = ROOT / "artifacts/demo_model.joblib"

    if (art43 / "lr.joblib").is_file():
        candidates.append(("v4_3_tfidf_lr", "joblib", art43 / "lr.joblib"))
    if (art43 / "svm.joblib").is_file():
        candidates.append(("v4_3_tfidf_svm", "joblib", art43 / "svm.joblib"))
    if (art43 / "beto_emb_lr.joblib").is_file():
        candidates.append(("v4_3_beto_emb_lr", "emb_bundle", art43 / "beto_emb_lr.joblib"))
    if (art43 / "beto_emb_svm.joblib").is_file():
        candidates.append(("v4_3_beto_emb_svm", "emb_bundle", art43 / "beto_emb_svm.joblib"))
    if (art43 / "ensemble_softvote.joblib").is_file():
        candidates.append(("v4_3_ensemble_softvote", "ensemble", art43 / "ensemble_softvote.joblib"))
    if (art42 / "lr.joblib").is_file():
        candidates.append(("v4_2_lr_ref", "joblib", art42 / "lr.joblib"))
    if (art42 / "svm.joblib").is_file():
        candidates.append(("v4_2_svm_ref", "joblib", art42 / "svm.joblib"))
    if (art42 / "beto-v4-partial256" / "model_manifest.json").is_file():
        candidates.append(("beto_v4_2_partial256", "beto", art42 / "beto-v4-partial256"))
    if demo.is_file():
        candidates.append(("demo_lr", "joblib", demo))

    results = {}
    for name, kind, path in candidates:
        print(f"scoring {name}…", flush=True)
        if kind == "joblib":
            pred, lat = _score_classical(path, texts, labels)
        elif kind == "emb_bundle":
            pred, lat = _emb_bundle_predict(path, texts)
        elif kind == "ensemble":
            pred, lat = _ensemble_predict(path, texts)
        else:
            pred, lat = _beto_predict(path, texts, max_length=256)
        m = _metrics(labels, pred, lat)
        g = _goals(m)
        results[name] = {
            "path": str(path),
            "metrics": m,
            "goals": g,
            "goals_met": sum(g.values()),
            "errors_by_phenomenon": _errors_by_phenomenon(reviews, pred),
            "sample_errors": [
                {"review_id": r["review_id"], "true": r["intended_label"], "pred": p,
                 "phenomena": r.get("phenomena", []), "text": r["text"][:220]}
                for r, p in zip(reviews, pred) if p != r["intended_label"]
            ][:40],
        }

    payload = {
        "schema": "traveler-v4-independent-v3-comparison",
        "eval_records_sha256": data["records_sha256"],
        "eval_n": data["n"],
        "frozen_before_scoring": True,
        "not_used_for_fitting": True,
        "prior_evals_not_tuned": data.get("prior_evals_observed", []),
        "results": results,
        "summary": {
            name: {
                "accuracy": r["metrics"]["accuracy"],
                "macro_f1": r["metrics"]["macro_f1"],
                "macro_precision": r["metrics"]["macro_precision"],
                "macro_recall": r["metrics"]["macro_recall"],
                "neg_recall": r["metrics"]["neg_recall"],
                "neg_to_pos_rate": r["metrics"]["neg_to_pos_rate"],
                "mean_latency_ms": r["metrics"]["mean_latency_ms"],
                "goals_met": r["goals_met"],
                "confusion_matrix": r["metrics"]["confusion_matrix"],
            }
            for name, r in results.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
