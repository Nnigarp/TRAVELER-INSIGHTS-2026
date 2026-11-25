"""Score frozen independent eval v2 (≥300) once; never fit on it."""
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

from traveler_agent.corpus import CorpusError, file_sha, text_sha  # noqa: E402
from traveler_agent.labels import LABELS  # noqa: E402
from traveler_agent.synthetic_diversity import canonical_sha, write_json  # noqa: E402

EVAL = ROOT / "src/traveler_agent/research_data/v4_independent_eval_v2.json"


def _metrics(truth, pred, latency_ms):
    prec, rec, f1, support = precision_recall_fscore_support(
        truth, pred, labels=list(LABELS), zero_division=0)
    per = {label: {"precision": float(prec[i]), "recall": float(rec[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
           for i, label in enumerate(LABELS)}
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
        "neg_to_pos": sum(1 for t, p in zip(truth, pred) if t == "negativo" and p == "positivo"),
        "neg_to_pos_rate": sum(1 for t, p in zip(truth, pred) if t == "negativo" and p == "positivo")
            / max(sum(1 for t in truth if t == "negativo"), 1),
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
    if data.get("schema") != "traveler-v4-independent-eval-v2" or data.get("frozen") is not True:
        raise CorpusError("Eval v2 no congelada.")
    reviews = data["reviews"]
    digest = canonical_sha([{k: r[k] for k in ("review_id", "text", "intended_label")} for r in reviews])
    if data.get("records_sha256") != digest:
        raise CorpusError("Hash de eval v2 inconsistente.")
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


def _errors_by_phenomenon(reviews, pred):
    buckets = defaultdict(lambda: Counter())
    for row, p in zip(reviews, pred):
        ok = p == row["intended_label"]
        for ph in row.get("phenomena", []):
            buckets[ph]["n"] += 1
            buckets[ph]["errors" if not ok else "ok"] += 1
    return {k: dict(v) for k, v in sorted(buckets.items())}


def _truncation_audit(reviews, artifact: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(artifact / "checkpoint"), local_files_only=True)
    stats = {"n": len(reviews), "over_128": 0, "over_256": 0, "max_tokens": 0}
    for row in reviews:
        n = len(tok(row["text"], add_special_tokens=True)["input_ids"])
        stats["max_tokens"] = max(stats["max_tokens"], n)
        if n > 128:
            stats["over_128"] += 1
        if n > 256:
            stats["over_256"] += 1
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/research/v4/independent_v2_comparison_2026-11-25.json"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Refuse overwrite of frozen comparison.")
    data, reviews = _load_eval()
    texts = [r["text"] for r in reviews]
    truth = [r["intended_label"] for r in reviews]
    results = {}

    classical = {
        "v4_2_lr": Path("artifacts/synthetic-tourism-v4.2/lr.joblib"),
        "v4_2_svm": Path("artifacts/synthetic-tourism-v4.2/svm.joblib"),
        "v4_1_lr": Path("artifacts/synthetic-tourism-v4.1/lr.joblib"),
        "v4_1_svm": Path("artifacts/synthetic-tourism-v4.1/svm.joblib"),
        "v4_0_lr": Path("artifacts/synthetic-tourism-v4/lr.joblib"),
        "demo_lr": Path("artifacts/demo_model.joblib"),
    }
    for name, path in classical.items():
        if not path.is_file():
            results[name] = {"status": "MISSING", "path": str(path)}
            continue
        model = joblib.load(path)
        t0 = time.perf_counter()
        pred = [str(x) for x in model.predict(texts)]
        latency = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
        m = _metrics(truth, pred, latency)
        results[name] = {
            "status": "OK", "path": str(path), "sha256": file_sha(path),
            "metrics": m, "goals": _goals(m),
            "errors_by_phenomenon": _errors_by_phenomenon(reviews, pred),
        }

    beto = [
        ("beto_v3_pr28", Path("artifacts/beto-v3-pr28-local"), 256),
        ("beto_v4_2_frozen256", Path("artifacts/synthetic-tourism-v4.2/beto-v4-frozen256"), 256),
        ("beto_v4_2_partial256", Path("artifacts/synthetic-tourism-v4.2/beto-v4-partial256"), 256),
        ("beto_v4_1_partial256", Path("artifacts/synthetic-tourism-v4.1/beto-v4-partial256"), 256),
    ]
    trunc = None
    for name, path, maxlen in beto:
        if not (path / "model_manifest.json").is_file():
            results[name] = {"status": "MISSING", "path": str(path)}
            continue
        if trunc is None:
            trunc = _truncation_audit(reviews, path)
        pred, latency = _beto_predict(path, texts, maxlen)
        m = _metrics(truth, pred, latency)
        results[name] = {
            "status": "OK", "path": str(path), "infer_max_length": maxlen,
            "manifest_sha256": file_sha(path / "model_manifest.json"),
            "metrics": m, "goals": _goals(m),
            "errors_by_phenomenon": _errors_by_phenomenon(reviews, pred),
        }

    report = {
        "schema": "traveler-v4-independent-v2-comparison",
        "eval_path": str(EVAL),
        "eval_records_sha256": data["records_sha256"],
        "eval_n": len(reviews),
        "frozen_before_prediction": True,
        "optimized_on_this_set": False,
        "truncation_coverage": trunc,
        "results": results,
        "experimental_targets_note": "Aspirational; reported honestly even if unmet.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    summary = {k: {"status": v.get("status"),
                   "acc": v.get("metrics", {}).get("accuracy"),
                   "macro_f1": v.get("metrics", {}).get("macro_f1"),
                   "neg_recall": v.get("metrics", {}).get("neg_recall"),
                   "neg2pos": v.get("metrics", {}).get("neg_to_pos"),
                   "goals_met": sum(1 for x in v.get("goals", {}).values() if x) if v.get("goals") else None}
               for k, v in results.items()}
    print(json.dumps({"status": "WRITTEN", "output": str(args.output),
                      "sha256": file_sha(args.output), "summary": summary,
                      "truncation": trunc}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
