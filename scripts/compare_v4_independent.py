"""Compare classical/BETO historical vs V4 models on the frozen independent eval.

Never fits on the independent set. Writes docs/research/v4/independent_comparison_*.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from traveler_agent.corpus import CorpusError, file_sha, text_sha  # noqa: E402
from traveler_agent.labels import LABELS  # noqa: E402
from traveler_agent.synthetic_diversity import canonical_sha, write_json  # noqa: E402

EVAL = ROOT / "src/traveler_agent/research_data/v4_independent_eval.json"


def _metrics(truth, pred, *, latency_ms, train_seconds=None):
    prec, rec, f1, support = precision_recall_fscore_support(
        truth, pred, labels=list(LABELS), zero_division=0
    )
    per = {
        label: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    return {
        "n": len(truth),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=list(LABELS), average="macro", zero_division=0)),
        "per_class": per,
        "classification_report": classification_report(
            truth, pred, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(truth, pred, labels=list(LABELS)).tolist(),
        "neg_to_pos": sum(1 for t, p in zip(truth, pred) if t == "negativo" and p == "positivo"),
        "mean_latency_ms": latency_ms,
        "train_seconds": train_seconds,
    }


def _load_eval():
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    if data.get("schema") != "traveler-v4-independent-eval-v1" or data.get("frozen") is not True:
        raise CorpusError("Eval independiente no congelada.")
    reviews = data["reviews"]
    digest = canonical_sha([{k: r[k] for k in ("review_id", "text", "intended_label")} for r in reviews])
    if data.get("records_sha256") != digest:
        raise CorpusError("records_sha256 de la eval no coincide; abortar.")
    for row in reviews:
        if row.get("text_sha256") != text_sha(row["text"]):
            raise CorpusError(f"Huella inconsistente: {row['review_id']}")
    return data, reviews


def _predict_classical(path: Path, texts):
    model = joblib.load(path)
    t0 = time.perf_counter()
    pred = [str(x) for x in model.predict(texts)]
    latency = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
    return pred, latency


def _predict_beto_v3(artifact: Path, texts, max_length: int):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import load_file

    meta = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(artifact / "checkpoint"), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(artifact / "checkpoint"), local_files_only=True,
        use_safetensors=False, weights_only=True, add_pooling_layer=False,
    ).eval()
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
            probs = pooled @ W.T + b
            preds.append(labels[int(np.argmax(probs))])
    latency = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
    return preds, latency, meta.get("max_length")


def _predict_beto_v4(artifact: Path, texts, max_length: int):
    return _predict_beto_v3(artifact, texts, max_length)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/research/v4/independent_comparison_2026-11-25.json"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Output exists; refuse overwrite.")
    data, reviews = _load_eval()
    texts = [r["text"] for r in reviews]
    truth = [r["intended_label"] for r in reviews]
    results = {}

    classical = {
        "demo_lr": Path("artifacts/demo_model.joblib"),
        "v1_lr": Path("artifacts/synthetic-tourism-v1/lr.joblib"),
        "v1_svm": Path("artifacts/synthetic-tourism-v1/svm.joblib"),
        "v4_0_lr": Path("artifacts/synthetic-tourism-v4/lr.joblib"),
        "v4_0_svm": Path("artifacts/synthetic-tourism-v4/svm.joblib"),
        "v4_1_lr": Path("artifacts/synthetic-tourism-v4.1/lr.joblib"),
        "v4_1_svm": Path("artifacts/synthetic-tourism-v4.1/svm.joblib"),
    }
    for name, path in classical.items():
        if not path.is_file():
            results[name] = {"status": "MISSING", "path": str(path)}
            continue
        pred, latency = _predict_classical(path, texts)
        train_s = None
        report = path.with_name(path.stem + "_report.json")
        if report.is_file():
            train_s = json.loads(report.read_text(encoding="utf-8")).get("environment")
        results[name] = {
            "status": "OK",
            "path": str(path),
            "sha256": file_sha(path),
            "metrics": _metrics(truth, pred, latency_ms=latency),
        }

    beto_specs = [
        ("beto_v3_pr28", Path("artifacts/beto-v3-pr28-local"), 256),
        ("beto_v4_0_frozen256", Path("artifacts/synthetic-tourism-v4/beto-v4-frozen256"), 256),
        ("beto_v4_1_frozen128", Path("artifacts/synthetic-tourism-v4.1/beto-v4-frozen128"), 128),
        ("beto_v4_1_frozen256", Path("artifacts/synthetic-tourism-v4.1/beto-v4-frozen256"), 256),
        ("beto_v4_1_partial256", Path("artifacts/synthetic-tourism-v4.1/beto-v4-partial256"), 256),
    ]
    for name, path, maxlen in beto_specs:
        if not (path / "model_manifest.json").is_file():
            results[name] = {"status": "MISSING", "path": str(path)}
            continue
        pred, latency, art_max = _predict_beto_v4(path, texts, maxlen)
        results[name] = {
            "status": "OK",
            "path": str(path),
            "infer_max_length": maxlen,
            "artifact_max_length": art_max,
            "manifest_sha256": file_sha(path / "model_manifest.json"),
            "metrics": _metrics(truth, pred, latency_ms=latency),
        }

    # Cartagena diagnostic row
    cart = next(r for r in reviews if r["review_id"] == "ie_cartagena_fail")
    cart_preds = {}
    for name, payload in results.items():
        if payload.get("status") != "OK":
            continue
        # re-predict single text for clarity when classical
        if name.startswith("beto"):
            continue
        path = Path(payload["path"])
        pred, _ = _predict_classical(path, [cart["text"]])
        cart_preds[name] = pred[0]
    for name, path, maxlen in beto_specs:
        if results.get(name, {}).get("status") != "OK":
            continue
        pred, _, _ = _predict_beto_v4(path, [cart["text"]], maxlen)
        cart_preds[name] = pred[0]

    report = {
        "schema": "traveler-v4-independent-comparison",
        "eval_path": str(EVAL),
        "eval_records_sha256": data["records_sha256"],
        "eval_n": len(reviews),
        "frozen_before_prediction": True,
        "optimized_on_this_set": False,
        "results": results,
        "cartagena_case": {
            "review_id": cart["review_id"],
            "intended_label": cart["intended_label"],
            "word_count": len(cart["text"].split()),
            "predictions": cart_preds,
        },
        "gradio_bindings_note": {
            "lr_default": "artifacts/demo_model.joblib (DEMO synthetic, not V4)",
            "svm_default": "in-memory train on DEMO via get_demo_model('svm')",
            "beto": "TRAVELER_BETO_ARTIFACT env → typically artifacts/beto-v3-pr28-local",
            "v4_models_not_wired_to_ui": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({
        "status": "WRITTEN",
        "output": str(args.output),
        "sha256": file_sha(args.output),
        "available": {k: v["status"] for k, v in results.items()},
        "cartagena": cart_preds,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
