"""Reproducible LR / SVM / BETO V4 training preparation on the V4 synthetic corpus.

Historical V1–V3 artifacts remain untouched. This module never writes into
artifacts/synthetic-tourism-v1 or artifacts/beto-v3-*.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .corpus import CorpusError, file_sha
from .labels import LABELS
from .model import train as train_classical
from .synthetic_diversity import ignored_target, write_json, canonical_sha
from .synthetic_tourism_v4 import (
    FIELDS, MODEL, PROTOCOL, SCHEMA, VERSION, audit_records, protocol,
)


def _classical_metrics(truth: list[str], pred: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=list(LABELS), average="macro", zero_division=0)),
        "classification_report": classification_report(
            truth, pred, labels=list(LABELS), output_dict=True, zero_division=0,
        ),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(truth, pred, labels=list(LABELS)).tolist(),
        "neg_to_pos": sum(1 for t, p in zip(truth, pred) if t == "negativo" and p == "positivo"),
    }

COMMON_ROLES = ("train", "validation", "test")


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "ABSENT"


def load_v4_corpus(input_dir: Path) -> tuple[list[dict], dict]:
    root = ignored_target(Path(input_dir), kinds=("data/processed", "data/private"))
    path = root / "synthetic_reviews_v4.csv"
    info_path = root / "synthetic_manifest_v4.json"
    if not path.is_file() or not info_path.is_file():
        raise CorpusError("Falta corpus V4 o manifiesto.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if (info.get("schema") != SCHEMA or info.get("source_type") != "SYNTHETIC_GENERATED"
            or info.get("version") != VERSION or info.get("model") != MODEL
            or info.get("rows_sha256") != file_sha(path)
            or info.get("protocol_sha256") != file_sha(PROTOCOL)):
        raise CorpusError("Procedencia V4 no verificable.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise CorpusError("Esquema CSV V4 inesperado.")
        rows = list(reader)
    audit = audit_records(rows)
    if audit["records"] != info.get("audit", {}).get("records"):
        raise CorpusError("Auditoría del manifiesto inconsistente.")
    return rows, info


def partition(rows: list[dict]) -> dict[str, list[dict]]:
    parts = {role: [r for r in rows if r["role"] == role] for role in (*COMMON_ROLES, "hard_holdout")}
    for role in COMMON_ROLES:
        if set(r["intended_label"] for r in parts[role]) != set(LABELS):
            raise CorpusError(f"Partición {role} sin las tres clases.")
    return parts


def train_classical_v4(
    input_dir: Path,
    out_dir: Path,
    *,
    model_kind: str,
) -> dict:
    if model_kind not in {"lr", "svm"}:
        raise CorpusError("model_kind debe ser lr o svm.")
    rows, info = load_v4_corpus(input_dir)
    parts = partition(rows)
    destination = ignored_target(Path(out_dir), kinds=("artifacts",))
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / f"{model_kind}.joblib"
    report_path = destination / f"{model_kind}_report.json"
    if marker.exists() or report_path.exists():
        raise CorpusError("Artefacto V4 ya existe; no sobrescribir historial.")

    train_texts = [r["text"] for r in parts["train"]]
    train_labels = [r["intended_label"] for r in parts["train"]]
    model = train_classical(train_texts, train_labels, model_kind=model_kind)
    joblib.dump(model, marker)

    def evaluate(name: str, subset: list[dict]) -> dict:
        texts = [r["text"] for r in subset]
        labels = [r["intended_label"] for r in subset]
        pred = [str(x) for x in model.predict(texts)]
        return {
            "split": name,
            "n": len(subset),
            "label_counts": dict(Counter(labels)),
            "metrics": _classical_metrics(labels, pred),
        }

    report = {
        "schema": "traveler-v4-classical-report",
        "model_kind": model_kind,
        "corpus_rows_sha256": info["rows_sha256"],
        "protocol_sha256": info["protocol_sha256"],
        "train_rows": len(parts["train"]),
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
        "limitations": [
            "Synthetic generator intent labels only.",
            "Hard holdout is still in-generator; not population validity.",
        ],
    }
    write_json(report_path, report)
    (destination / f"{model_kind}.meta.json").write_text(
        json.dumps({
            "model_kind": model_kind,
            "corpus_rows_sha256": info["rows_sha256"],
            "report_sha256": file_sha(report_path),
            "labels": list(LABELS),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def prepare_beto_v4_plan(input_dir: Path, out_dir: Path) -> dict:
    """Materialize frozen JSON plan for BETO V4 without training weights yet."""
    rows, info = load_v4_corpus(input_dir)
    parts = partition(rows)
    proto = protocol()
    destination = ignored_target(Path(out_dir), kinds=("artifacts", "data/processed", "docs"))
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination / "beto_v4_training_plan.json"
    if plan_path.exists():
        raise CorpusError("Plan BETO V4 ya existe.")

    def slim(subset: list[dict]) -> list[dict]:
        return [{
            "review_id": r["review_id"],
            "text": r["text"],
            "intended_label": r["intended_label"],
            "scenario_family": r["scenario_family"],
            "template": r["template"],
            "length_band": r["length_band"],
            "phenomena": r["phenomena"],
            "text_sha256": r["text_sha256"],
            "approx_beto_tokens": int(r["approx_beto_tokens"]),
        } for r in subset]

    plan = {
        "schema": "traveler-beto-v4-training-plan",
        "status": "PREPARED_NOT_TRAINED",
        "corpus_rows_sha256": info["rows_sha256"],
        "protocol_sha256": info["protocol_sha256"],
        "base_model": proto["beto"]["base_model"],
        "revision": proto["beto"]["revision"],
        "default_max_length": proto["beto"]["default_train_max_length"],
        "candidate_max_lengths": proto["beto"]["candidate_max_lengths"],
        "truncation_finding": proto["beto"]["truncation_finding"],
        "hardware_budget": proto["hardware_budget"],
        "counts": {role: len(parts[role]) for role in parts},
        "label_counts": {
            role: dict(Counter(r["intended_label"] for r in parts[role])) for role in parts
        },
        "splits": {role: slim(parts[role]) for role in parts},
        "training_recipe": {
            "baselines": ["dummy", "tfidf_lr", "tfidf_svm"],
            "beto_modes": [
                {"id": "frozen128", "max_length": 128, "encoder": "frozen"},
                {"id": "frozen256", "max_length": 256, "encoder": "frozen"},
                {"id": "partial256", "max_length": 256, "encoder": "last_two_blocks"},
            ],
            "selection_split": "validation",
            "assessment_splits": ["test", "hard_holdout"],
            "seed": proto["seed"],
            "notes": [
                "Do not fit on hard_holdout.",
                "Prefer 256 tokens after measured truncation failure at 128.",
                "On RTX 4050 6GB use batch_size 2–4 and gradient checkpointing if needed.",
            ],
        },
        "records_sha256": canonical_sha(rows),
        "human_gold": False,
    }
    write_json(plan_path, plan)
    # Compact index without full texts for docs handoff.
    index = {
        "schema": "traveler-beto-v4-training-index",
        "status": plan["status"],
        "corpus_rows_sha256": info["rows_sha256"],
        "plan_sha256": file_sha(plan_path),
        "counts": plan["counts"],
        "label_counts": plan["label_counts"],
        "default_max_length": plan["default_max_length"],
        "training_recipe": plan["training_recipe"],
    }
    write_json(destination / "beto_v4_training_index.json", index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entrenamiento / plan V4")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train-classical")
    p_train.add_argument("--input", type=Path, default=Path("data/processed/synthetic-tourism-v4"))
    p_train.add_argument("--out", type=Path, default=Path("artifacts/synthetic-tourism-v4"))
    p_train.add_argument("--model-kind", choices=("lr", "svm"), required=True)

    p_plan = sub.add_parser("prepare-beto-plan")
    p_plan.add_argument("--input", type=Path, default=Path("data/processed/synthetic-tourism-v4"))
    p_plan.add_argument("--out", type=Path, default=Path("artifacts/synthetic-tourism-v4"))

    args = parser.parse_args(argv)
    try:
        if args.cmd == "train-classical":
            report = train_classical_v4(args.input, args.out, model_kind=args.model_kind)
            print(json.dumps({
                "status": "TRAINED",
                "model_kind": args.model_kind,
                "validation_macro_f1": report["validation"]["metrics"].get("macro_f1"),
                "test_macro_f1": report["test"]["metrics"].get("macro_f1"),
                "hard_macro_f1": report["hard_holdout"]["metrics"].get("macro_f1"),
            }, ensure_ascii=False, indent=2))
        else:
            index = prepare_beto_v4_plan(args.input, args.out)
            print(json.dumps(index, ensure_ascii=False, indent=2))
    except (CorpusError, ValueError, OSError) as exc:
        parser.exit(2, f"V4 training blocked: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
