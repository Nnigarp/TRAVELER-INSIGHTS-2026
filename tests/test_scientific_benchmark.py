"""Software tests with synthetic records; NEVER publish scores as science."""
import csv
import json
from pathlib import Path

import pytest

from traveler_agent.corpus import (
    CorpusError, SOURCE_COLUMNS, REVIEW_COLUMNS, ANNOTATION_COLUMNS,
    ADJUDICATION_COLUMNS, _write_csv, file_sha
)
from traveler_agent.corpus_splits import prepare_splits
from traveler_agent.scientific_benchmark import (
    verified_splits, run_benchmark, MODEL_GRID, _new_estimator
)


@pytest.fixture(scope="module")
def corpus_paths(tmp_path_factory):
    private = tmp_path_factory.mktemp("synthetic_private")
    split_dir = tmp_path_factory.mktemp("synthetic_processed")
    source = private / "sources.csv"
    reviews = private / "reviews.csv"
    ann = private / "annotations.csv"
    adj = private / "adjudications.csv"
    _write_csv(source, SOURCE_COLUMNS, [
        {"source_id": "demo", "source_type": "voluntary",
         "source_reference": "SYNTHETIC_TEST_ONLY",
         "reuse_basis": "consent", "permission_reference": "SIMULATED_CONSENT",
         "authorization_reviewed": "yes"}
    ])
    labels = ("positivo", "neutral", "negativo")
    words = ("maravilloso", "normal", "terrible")
    review_rows, annotation_rows, final_rows = [], [], []
    for i in range(240):
        label = labels[i % 3]
        rid = f"r{i:04d}"
        review_rows.append({
            "review_id": rid, "text": f"Reseña artificial {words[i % 3]} número {i}",
            "source_id": "demo", "group_id": f"g{i:04d}", "language": "es",
        })
        annotation_rows.append({"review_id": rid, "annotator_id": "a1",
                                "sentiment": label, "aspects_json": "{}"})
        final_rows.append({"review_id": rid, "adjudicator_id": "lead",
                           "sentiment": label, "aspects_json": "{}",
                           "rationale": "Etiqueta fabricada para probar software"})
    _write_csv(reviews, REVIEW_COLUMNS, review_rows)
    _write_csv(ann, ANNOTATION_COLUMNS, annotation_rows)
    _write_csv(adj, ADJUDICATION_COLUMNS, final_rows)
    prepare_splits(reviews, source, ann, adj, split_dir, seed=42)
    return private, split_dir


def test_verified_splits_preserve_provenance_and_class_balance(corpus_paths):
    private, splits = corpus_paths
    rows, manifest = verified_splits(private, splits)
    assert set(rows) == {"train", "validation", "test"}
    assert sum(len(group) for group in rows.values()) == 240
    assert set(manifest["partitions"]) == set(rows)


def test_benchmark_trains_three_models_without_scientific_claim(corpus_paths, tmp_path):
    private, splits = corpus_paths
    out = tmp_path / "artifacts" / "benchmark"
    report = run_benchmark(private, splits, out, demo_only=True)
    assert report["report_type"] == "SOFTWARE_DEMO_ONLY"
    assert set(report["models"]) == {"dummy", "lr", "svm"}
    assert report["models"]["lr"]["selected_parameters"]["C"] in {0.5, 2.0}
    for kind in ("dummy", "lr", "svm"):
        assert (out / f"{kind}.joblib").is_file()
        assert report["models"][kind]["test"]["n"] > 0
        assert len(report["models"][kind]["test"]["confusion_matrix"]) == 3
        assert "es" in report["models"][kind]["test"]["by_language"]
    metadata = json.loads((out / "model_manifest.json").read_text(encoding="utf-8"))
    assert set(metadata["model_metadata"]) == {"dummy", "lr", "svm"}
    assert all(len(metadata["model_metadata"][k]["sha256"]) == 64 for k in metadata["model_metadata"])
    assert all("texto artificial" not in json.dumps(report).lower() for _ in [0])


def test_rejects_tampered_split_file_even_if_schema_still_valid(corpus_paths):
    private, split_dir = corpus_paths
    testpath = split_dir / "test.csv"
    original = testpath.read_bytes()
    try:
        testpath.write_bytes(original.replace(b"artificial", b"altered"))
        with pytest.raises(CorpusError, match="alterada"):
            verified_splits(private, split_dir)
    finally:
        testpath.write_bytes(original)


def test_rejects_group_leakage_even_if_attacker_updates_hash(corpus_paths):
    private, split_dir = corpus_paths
    trainpath = split_dir / "train.csv"
    manifestpath = split_dir / "split_manifest.json"
    saved_train, saved_manifest = trainpath.read_bytes(), manifestpath.read_bytes()
    try:
        with trainpath.open("r", encoding="utf-8", newline="") as f:
            train_rows = list(csv.DictReader(f))
        with (split_dir / "test.csv").open("r", encoding="utf-8", newline="") as f:
            test_rows = list(csv.DictReader(f))
        # Reassign the first train row to a held-out group (same text/ID otherwise).
        train_rows[0]["group_id"] = test_rows[0]["group_id"]
        with trainpath.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=train_rows[0].keys())
            writer.writeheader()
            writer.writerows(train_rows)
        manifest = json.loads(saved_manifest)
        manifest["output_hashes"]["train"] = file_sha(trainpath)
        manifestpath.write_text(json.dumps(manifest), encoding="utf-8")
        # Cross-source correspondence detects tamper even before overlap check.
        with pytest.raises(CorpusError, match="no coincide|Fuga"):
            verified_splits(private, split_dir)
    finally:
        trainpath.write_bytes(saved_train)
        manifestpath.write_bytes(saved_manifest)


def test_tuning_grid_and_train_only_calibration():
    assert len(MODEL_GRID["dummy"]) == 1
    assert len(MODEL_GRID["lr"]) == 2
    assert len(MODEL_GRID["svm"]) == 2
    svm = _new_estimator("svm", {"C": 2.0}, 42)
    assert svm.cv == 3
    assert svm.estimator.named_steps["classifier"].C == 2.0
    assert "tfidf" in svm.estimator.named_steps


def test_rejects_public_output_directory(corpus_paths, tmp_path):
    private, split_dir = corpus_paths
    with pytest.raises(CorpusError, match="privadas"):
        run_benchmark(private, split_dir, tmp_path / "public", demo_only=True)


def test_rejects_changed_seed(corpus_paths, tmp_path):
    private, split_dir = corpus_paths
    with pytest.raises(CorpusError, match="semilla"):
        run_benchmark(private, split_dir, tmp_path / "artifacts" / "b", seed=999, demo_only=True)
