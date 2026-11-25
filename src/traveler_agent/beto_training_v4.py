"""BETO V4 training entrypoint over the prepared synthetic plan.

Does not modify V3 artifacts. Requires encoder-research extras and a local
BETO checkpoint cache. Default max_length is 256 after the truncation audit.
"""
from __future__ import annotations

import argparse
import json
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
    if plan.get("status") == "TRAINED":
        raise CorpusError("Plan ya entrenado; usar otro directorio.")
    return plan


def train_frozen_head(
    *,
    input_dir: Path,
    plan_path: Path,
    checkpoint: Path,
    out_dir: Path,
    max_length: int = 256,
    seed: int = 20261125,
    threads: int = 2,
) -> dict:
    """Train a frozen-encoder logistic head on V4 train; score val/test/hard."""
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

    print("[beto-v4] importing encoder", flush=True)
    from .beto_classifier import BetoEncoder, export_classifier
    from .research_metrics import summarize_predictions

    print(f"[beto-v4] loading checkpoint {checkpoint} max_length={max_length}", flush=True)
    encoder = BetoEncoder(Path(checkpoint), download=False, threads=threads, max_length=max_length)
    train_texts = [r["text"] for r in parts["train"]]
    train_labels = [r["intended_label"] for r in parts["train"]]
    feature_cache = destination / f"beto_v4_train_features_{max_length}.npz"
    cache_meta = feature_cache.with_suffix(".json")
    if feature_cache.is_file() and cache_meta.is_file():
        meta_cache = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta_cache.get("corpus_rows_sha256") != info["rows_sha256"]:
            raise CorpusError("Caché de features de otro corpus.")
        print(f"[beto-v4] reusing cached train features {feature_cache}", flush=True)
        features = np.load(feature_cache)["features"]
    else:
        print(f"[beto-v4] encoding train n={len(train_texts)}", flush=True)
        features = encoder.encode(train_texts, batch_size=8)
        np.savez(feature_cache, features=features)
        cache_meta.write_text(
            json.dumps({"corpus_rows_sha256": info["rows_sha256"], "max_length": max_length}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[beto-v4] cached train features -> {feature_cache}", flush=True)
    print("[beto-v4] fitting logistic head", flush=True)
    head = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    head.fit(features, train_labels)
    # Align rows to LABELS order for export_classifier.
    class_index = [list(head.classes_).index(label) for label in LABELS]
    weight = np.asarray(head.coef_[class_index], dtype=np.float32)
    bias = np.asarray(head.intercept_[class_index], dtype=np.float32)

    export_classifier(
        artifact,
        Path(checkpoint),
        weight,
        bias,
        training={
            "supervised": True,
            "experiment": "BETO_V4_FROZEN_LOGREG_HEAD",
            "max_length": max_length,
            "seed": seed,
            "corpus_rows_sha256": info["rows_sha256"],
            "protocol_sha256": info["protocol_sha256"],
            "label_origin": protocol()["label_origin"],
            "head": "sklearn_logistic_regression_on_frozen_mean_l2",
        },
    )
    meta_path = artifact / "model_manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["max_length"] = max_length
    meta["pooling"] = "masked_mean_including_special_tokens_l2"
    meta["v4_note"] = "max_length set for V4 after truncation audit; schema remains supervised-v1 export."
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    stored = np.load(artifact / "head.npz")
    W, b = stored["weight"], stored["bias"]

    def eval_split(name: str, subset: list[dict]) -> dict:
        texts = [r["text"] for r in subset]
        truth = [r["intended_label"] for r in subset]
        print(f"[beto-v4] encoding {name} n={len(texts)}", flush=True)
        feats = encoder.encode(texts, batch_size=8)
        pred = [LABELS[i] for i in (feats @ W.T + b).argmax(axis=1)]
        return {"split": name, "metrics": summarize_predictions(truth, pred)}

    report = {
        "schema": "traveler-beto-v4-frozen-report",
        "status": "TRAINED_FROZEN_HEAD",
        "max_length": max_length,
        "seed": seed,
        "corpus_rows_sha256": info["rows_sha256"],
        "artifact": str(artifact),
        "validation": eval_split("validation", parts["validation"]),
        "test": eval_split("test", parts["test"]),
        "hard_holdout": eval_split("hard_holdout", parts["hard_holdout"]),
        "human_gold": False,
        "historical_v3_preserved": True,
    }
    write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entrenar cabeza congelada BETO V4")
    parser.add_argument("--input", type=Path, default=Path("data/processed/synthetic-tourism-v4"))
    parser.add_argument("--plan", type=Path, default=Path("artifacts/synthetic-tourism-v4/beto_v4_training_plan.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/beto-v3-pr28-local/checkpoint"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/synthetic-tourism-v4"))
    parser.add_argument("--max-length", type=int, default=256, choices=(128, 256))
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = train_frozen_head(
            input_dir=args.input,
            plan_path=args.plan,
            checkpoint=args.checkpoint,
            out_dir=args.out,
            max_length=args.max_length,
            seed=args.seed,
            threads=args.threads,
        )
    except (CorpusError, ValueError, OSError) as exc:
        parser.exit(2, f"BETO V4 blocked: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "max_length": report["max_length"],
        "validation_macro_f1": report["validation"]["metrics"]["macro_f1"],
        "test_macro_f1": report["test"]["metrics"]["macro_f1"],
        "hard_macro_f1": report["hard_holdout"]["metrics"]["macro_f1"],
        "hard_neg_to_pos": report["hard_holdout"]["metrics"]["negative_to_positive_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
