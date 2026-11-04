from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from traveler_agent.emoji_analysis import emoji_profile


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", " ", text)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".txt"}:
        # sep=None delegates delimiter sniffing to the Python engine.
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported tabular format: {suffix}")


def safe_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def column_profile(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    non_null = s.dropna()
    examples = [safe_scalar(v) for v in non_null.head(5).tolist()]
    return {
        "dtype": str(s.dtype),
        "missing": int(s.isna().sum()),
        "missing_pct": round(float(s.isna().mean() * 100), 4),
        "unique_non_null": int(non_null.nunique(dropna=True)),
        "examples": examples,
    }


def text_profile(df: pd.DataFrame, col: str) -> dict[str, Any]:
    normalized = df[col].map(normalize_text)
    lengths = normalized.map(len)
    nonempty = normalized[normalized != ""]
    dup_mask = nonempty.duplicated(keep=False)
    counts = Counter(nonempty[dup_mask])
    return {
        "empty_after_normalization": int((normalized == "").sum()),
        "length_chars": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "mean": round(float(lengths.mean()), 4) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        },
        "duplicate_normalized_text_rows": int(dup_mask.sum()),
        "duplicate_normalized_text_values": int(len(counts)),
        "top_duplicate_normalized_text": [
            {"text": text[:200], "count": int(count)}
            for text, count in counts.most_common(10)
        ],
    }


def target_profile(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    counts = s.value_counts(dropna=False)
    distribution = []
    for label, count in counts.items():
        distribution.append({
            "label": safe_scalar(label),
            "count": int(count),
            "pct": round(float(count / len(df) * 100), 4) if len(df) else 0.0,
        })
    unique_non_null = set(s.dropna().tolist())
    notes: list[str] = []
    if unique_non_null == {0, 1, 2} or unique_non_null == {"0", "1", "2"}:
        notes.append(
            "Observed labels are {0,1,2}; semantic meaning must be verified from authoritative competition files, not inferred."
        )
    if len(unique_non_null) != 3:
        notes.append(f"Expected 3 sentiment classes; observed {len(unique_non_null)} non-null label values.")
    return {
        "missing": int(s.isna().sum()),
        "unique_non_null": int(s.dropna().nunique()),
        "distribution": distribution,
        "notes": notes,
    }


def audit_file(
    path: Path,
    target: str | None,
    text_col: str | None,
    id_col: str | None,
    group_col: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = read_table(path)
    report: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "column_profiles": {str(c): column_profile(df, str(c)) for c in df.columns},
        "exact_duplicate_rows": int(df.duplicated(keep=False).sum()),
        "checks": [],
    }

    def require(col: str | None, role: str) -> bool:
        if not col:
            return False
        if col not in df.columns:
            report["checks"].append({
                "status": "FAIL", "check": f"{role}_column_exists", "detail": col
            })
            return False
        report["checks"].append({
            "status": "PASS", "check": f"{role}_column_exists", "detail": col
        })
        return True

    if require(target, "target"):
        report["target_profile"] = target_profile(df, target)  # type: ignore[arg-type]
    if require(text_col, "text"):
        report["text_profile"] = text_profile(df, text_col)  # type: ignore[arg-type]
        report["emoji_profile"] = emoji_profile(df[text_col].tolist(), df[target].tolist() if target and target in df.columns else None)
    if require(id_col, "id"):
        s = df[id_col]  # type: ignore[index]
        report["id_profile"] = {
            "missing": int(s.isna().sum()),
            "duplicate_non_null_ids": int(s.dropna().duplicated(keep=False).sum()),
            "is_unique_non_null": bool(s.dropna().is_unique),
        }
    if require(group_col, "group"):
        s = df[group_col]  # type: ignore[index]
        report["group_profile"] = {
            "missing": int(s.isna().sum()),
            "unique_non_null": int(s.dropna().nunique()),
        }
    return df, report


def cross_file_checks(
    data: list[tuple[Path, pd.DataFrame]], text_col: str | None, group_col: str | None
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            p1, d1 = data[i]
            p2, d2 = data[j]
            pair = f"{p1.name} <-> {p2.name}"
            if text_col and text_col in d1.columns and text_col in d2.columns:
                a = set(d1[text_col].map(normalize_text)) - {""}
                b = set(d2[text_col].map(normalize_text)) - {""}
                overlap = sorted(a & b)
                checks.append({
                    "pair": pair,
                    "check": "normalized_text_overlap",
                    "overlap_count": len(overlap),
                    "examples": overlap[:10],
                    "risk": "HIGH" if overlap else "NONE",
                    "interpretation": "Exact normalized review overlap across splits can create leakage.",
                })
            if group_col and group_col in d1.columns and group_col in d2.columns:
                a = set(d1[group_col].dropna().astype(str))
                b = set(d2[group_col].dropna().astype(str))
                overlap = sorted(a & b)
                checks.append({
                    "pair": pair,
                    "check": "group_overlap",
                    "overlap_count": len(overlap),
                    "examples": overlap[:20],
                    "risk": "CONTEXT_DEPENDENT" if overlap else "NONE",
                    "interpretation": (
                        "If group represents tourist place/location and the research goal is out-of-place generalization, "
                        "group overlap should be controlled with group-aware splitting."
                    ),
                })
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auditable inspection of Traveler Insights tabular data before modeling."
    )
    parser.add_argument("inputs", nargs="+", help="CSV/XLSX/XLS/Parquet files to audit")
    parser.add_argument("--target", default=None, help="Target label column")
    parser.add_argument("--text-col", default=None, help="Review text column")
    parser.add_argument("--id-col", default=None, help="Review identifier column")
    parser.add_argument("--group-col", default=None, help="Place/location/group column")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--license", dest="license_name", default=None)
    parser.add_argument("--out", default="reports/dataset_audit.json")
    args = parser.parse_args()

    audited: list[tuple[Path, pd.DataFrame]] = []
    file_reports: list[dict[str, Any]] = []
    for raw in args.inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        df, report = audit_file(path, args.target, args.text_col, args.id_col, args.group_col)
        audited.append((path, df))
        file_reports.append(report)

    final_report = {
        "schema_version": "1.0",
        "purpose": "Traveler Insights 2026 dataset audit before training",
        "source": {"url": args.source_url, "license": args.license_name},
        "declared_columns": {
            "target": args.target,
            "text": args.text_col,
            "id": args.id_col,
            "group": args.group_col,
        },
        "files": file_reports,
        "cross_file_checks": cross_file_checks(audited, args.text_col, args.group_col),
        "scientific_guardrails": [
            "Do not infer sentiment label semantics from numeric values alone.",
            "Keep preprocessing inside the training pipeline to avoid leakage.",
            "Deduplicate or group-split before model selection when the same review/place can repeat.",
            "Never report demo-data performance as evidence of generalization.",
            "Record file hashes and split seeds for reproducibility.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "report": str(out.resolve()),
        "files": len(file_reports),
        "rows_total": sum(r["rows"] for r in file_reports),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
