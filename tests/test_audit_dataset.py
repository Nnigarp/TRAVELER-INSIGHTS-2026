from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_dataset.py"

spec = importlib.util.spec_from_file_location("audit_dataset", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_normalize_text_nfkc_whitespace_case():
    assert module.normalize_text("  HOLA\tMundo  ") == "hola mundo"


def test_target_profile_warns_for_012_mapping():
    df = pd.DataFrame({"Sentimiento": [0, 1, 2, 2]})
    p = module.target_profile(df, "Sentimiento")
    assert p["unique_non_null"] == 3
    assert any("must be verified" in note for note in p["notes"])


def test_text_profile_detects_duplicates():
    df = pd.DataFrame({"review": ["Muy bueno", " muy   bueno ", "Regular"]})
    p = module.text_profile(df, "review")
    assert p["duplicate_normalized_text_rows"] == 2
    assert p["duplicate_normalized_text_values"] == 1


def test_cross_file_overlap_detects_leakage(tmp_path: Path):
    a = pd.DataFrame({"review": ["Excelente lugar", "Normal"], "place": ["A", "B"]})
    b = pd.DataFrame({"review": [" excelente  lugar ", "Malo"], "place": ["A", "C"]})
    checks = module.cross_file_checks([(tmp_path/"train.csv", a), (tmp_path/"test.csv", b)], "review", "place")
    text = next(c for c in checks if c["check"] == "normalized_text_overlap")
    group = next(c for c in checks if c["check"] == "group_overlap")
    assert text["overlap_count"] == 1
    assert text["risk"] == "HIGH"
    assert group["overlap_count"] == 1


def test_cli_writes_reproducible_report(tmp_path: Path):
    train = tmp_path / "train.csv"
    out = tmp_path / "report.json"
    pd.DataFrame({
        "ID": [1, 2, 3],
        "review": ["bien", "regular", "mal"],
        "Sentimiento": [2, 1, 0],
        "place": ["A", "B", "C"],
    }).to_csv(train, index=False)
    proc = subprocess.run([
        sys.executable, str(SCRIPT), str(train),
        "--target", "Sentimiento", "--text-col", "review",
        "--id-col", "ID", "--group-col", "place",
        "--out", str(out)
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["files"][0]["rows"] == 3
    assert len(data["files"][0]["sha256"]) == 64
    assert data["files"][0]["id_profile"]["is_unique_non_null"] is True
