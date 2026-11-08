"""Frozen V3 catalog, crossed sample budgets and TRAIN-only coverage diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re

from .corpus import CorpusError, file_sha, text_sha
from .labels import LABELS
from .synthetic_diversity import (CATALOG as V2_CATALOG, canonical_sha, ignored_target,
                                  lexical_audit, write_json)

RESOURCE = Path(__file__).with_name("research_data")
CATALOG = RESOURCE / "synthetic_diversity_v3.json"
PROTOCOL = RESOURCE / "synthetic_v3_protocol.json"
DOMAINS = ("hotel", "restaurant", "museum", "transport", "tour")
ASPECTS = ("cleanliness", "safety", "accessibility", "price", "service")
LABELED_ROLES = ("train", "tune", "calibration", "policy_dev", "assessment")
UNLABELED_ROLES = ("ambiguity_policy", "ambiguity_assessment")
TAGS = {"negation", "double_negation", "negative_concordance", "explicit_contrast",
        "implicit_contrast", "mixed_aspects", "regionalism", "typo", "positive_words_negative_context"}


def v2_coverage() -> dict:
    """Inspect only V2 TRAIN/development metadata; never score or expose its test."""
    original = json.loads(V2_CATALOG.read_text(encoding="utf-8"))
    report = {}
    for role in ("train", "validation"):
        rows = [case for family in original["families"] if family["role"] == role
                for case in family["cases"]]
        report[role] = {
            "rows": len(rows),
            "phenomena_by_intended_class": {
                tag: dict(Counter(r["intended_label"] for r in rows if tag in r["phenomena"]))
                for tag in sorted({tag for r in rows for tag in r["phenomena"]})},
            "domain_and_aspect_contributions": "NOT_IDENTIFIABLE_FROM_V2_DESIGN_OR_METADATA",
        }
    return {"source_sha256": file_sha(V2_CATALOG), "roles_inspected": report,
            "held_out_rows_scored_or_used_for_selection": False,
            "interpretation": "Coverage and target association, not causal feature contribution."}


def materialize(catalog: dict) -> list[dict]:
    if (catalog.get("version") != "synthetic-diversity-v3.0.0"
            or catalog.get("provenance", {}).get("human_authors") is not False
            or catalog["provenance"].get("label_origin") != "AUTHOR_INTENT_NOT_HUMAN_GOLD"):
        raise CorpusError("La procedencia sintética no puede convertirse en gold humano.")
    rows, families, components = [], set(), {}
    for family in catalog["families"]:
        fid, role = family["family"], family["role"]
        if (fid in families or role not in LABELED_ROLES or family["domain"] not in DOMAINS
                or family["aspect"] not in ASPECTS or family["region"] not in {"es", "es-CO", "es-PE"}):
            raise CorpusError("Familia, rol o dimensiones inválidos.")
        families.add(fid)
        component = family["semantic_component"]
        if component in components and components[component] != role:
            raise CorpusError("Componente semántico compartido entre roles.")
        components[component] = role
        variants = defaultdict(list)
        for case in family["cases"]:
            if (case["intended_label"] not in LABELS or not set(case["phenomena"]) <= TAGS
                    or not case["structure"] or not isinstance(case["variant"], int) or case["variant"] < 0):
                raise CorpusError("Intención, fenómeno o variante inválidos.")
            variants[case["variant"]].append(case["intended_label"])
            rows.append({**case, **{k: family[k] for k in ("family", "role", "semantic_component", "domain", "aspect", "region")},
                         "source": "V3_ASSISTANT_AUTHORED", "source_batch": family["source_batch"],
                         "narrow_pool": family.get("narrow_pool", False), "intended_review": None})
        if not variants or set(variants) != set(range(len(variants))) or any(sorted(v) != sorted(LABELS) for v in variants.values()):
            raise CorpusError("Cada variante requiere las tres intenciones; índices contiguos.")
        if role != "train" and len(variants) != 1:
            raise CorpusError("Solo TRAIN admite variantes relacionadas.")
    for case in catalog["ambiguity"]:
        if (case["role"] not in UNLABELED_ROLES or case["intended_label"] is not None
                or case["intended_review"] is not True or case["family"] in families):
            raise CorpusError("Ambigüedad sin sentimiento gold, familia única y revisión prevista.")
        families.add(case["family"])
        rows.append({**case, "semantic_component": case["family"], "source": "V3_SYNTHETIC_AMBIGUITY",
                     "source_batch": case["role"], "domain": "unspecified", "aspect": "unspecified",
                     "region": "unspecified", "variant": 0, "narrow_pool": False, "phenomena": [], "structure": "unresolved"})
    seen = set()
    for row in rows:
        if not isinstance(row["text"], str) or not 8 <= len(row["text"]) <= 3000:
            raise CorpusError("Texto inválido.")
        fingerprint = text_sha(row["text"])
        if fingerprint in seen:
            raise CorpusError("Duplicado exacto normalizado dentro de V3.")
        seen.add(fingerprint)
        row["text_sha256"] = fingerprint
        row["review_id"] = "v3_" + canonical_sha([row["family"], row["variant"], row["intended_label"], row["text"]])[:20]
        count = len(row["text"].split())
        row["length_band"] = "short" if count <= 12 else "long" if count >= 70 else "medium"
        row["label_origin"] = "AUTHOR_INTENT_NOT_HUMAN_GOLD"
    # All V2 rows, including old reserved ones, are used only as a hash denylist.
    # No V2 labels, scores or raw examples enter model development or new text drafting.
    v2 = json.loads(V2_CATALOG.read_text(encoding="utf-8"))
    old_hashes = {text_sha(c["text"]) for f in v2["families"] for c in f["cases"]}
    old_hashes.update(text_sha(c["text"]) for field in ("ambiguity_cohort", "illustrative_user_examples") for c in v2[field])
    if seen & old_hashes:
        raise CorpusError("V3 no reutiliza textos V2, ni sus ambiguos o ejemplos ilustrativos.")
    return sorted(rows, key=lambda r: r["review_id"])


def coverage(rows: list[dict]) -> dict:
    result = {}
    for role in (*LABELED_ROLES, *UNLABELED_ROLES):
        part = [r for r in rows if r["role"] == role]
        result[role] = {"rows": len(part), "families": len({r["family"] for r in part}),
                        "intended_labels": dict(Counter(r["intended_label"] for r in part if r["intended_label"])),
                        "dimensions": {k: dict(Counter(r[k] for r in part)) for k in ("domain", "aspect", "region", "length_band")},
                        "phenomena_by_intended_class": {tag: dict(Counter(r["intended_label"] for r in part if tag in r["phenomena"]))
                                                        for tag in sorted(TAGS)}}
    return result


def duplicate_screen(rows, threshold=.70) -> dict:
    """Reuse the frozen lexical screen, remapping each pair of roles independently."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["role"]].append(row)
    reports = []
    for left, right in combinations(sorted(groups), 2):
        mapped = [{**r, "role": "train"} for r in groups[left]] + [{**r, "role": "challenge"} for r in groups[right]]
        report = lexical_audit(mapped, threshold=threshold)
        reports.append({"roles": [left, right], **report})
    return {"pairs": reports, "candidate_count": sum(len(r["candidates"]) for r in reports),
            "semantic_independence_proven": False}


def training_arms(rows: list[dict]) -> dict:
    train = [r for r in rows if r["role"] == "train"]
    arms = {}
    for width, families, counts in (("narrow", 10, (2, 4)), ("broad", 20, (1, 2))):
        eligible = [r for r in train if width == "broad" or r["narrow_pool"]]
        if len({r["family"] for r in eligible}) != families:
            raise CorpusError("Familias distintas del factorial registrado.")
        for n, variants in zip((60, 120), counts):
            arm = [r for r in eligible if r["variant"] < variants]
            if len(arm) != n or Counter(r["intended_label"] for r in arm) != Counter({label: n // 3 for label in LABELS}):
                raise CorpusError("Presupuesto/clases del factorial inválidos.")
            arms[f"{width}_{n}"] = arm
    return arms


def ablation_rows(rows: list[dict], field: str, omitted: str, budget=120) -> list[dict]:
    """Omit one TRAIN dimension; replenish from related TRAIN variants, equal budget."""
    if field not in {"domain", "aspect"}:
        raise CorpusError("Dimensión de ablación inválida.")
    eligible = [r for r in rows if r["role"] == "train" and r[field] != omitted]
    units = defaultdict(list)
    for r in eligible:
        units[(r["family"], r["variant"])].append(r)
    ordered = sorted(units, key=lambda unit: (unit[1], canonical_sha(unit[0])))
    chosen = ordered[:budget // 3]
    result = [r for unit in chosen for r in units[unit]]
    if len(result) != budget or Counter(r["intended_label"] for r in result) != Counter({label: budget // 3 for label in LABELS}):
        raise CorpusError("Variantes insuficientes para ablación igualada.")
    return result


def shortcut_audit(training: list[dict]) -> dict:
    """TRAIN only: family-safe metadata probe and recurrent class-linked endings."""
    if not training or any(r["role"] != "train" for r in training):
        raise CorpusError("La auditoría de atajos usa solo TRAIN.")
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .scientific_benchmark import _training_calibration_folds
    from .research_metrics import summarize_predictions
    inputs = [{**{k: r[k] for k in ("domain", "aspect", "region", "length_band", "structure")},
               **{f"tag:{tag}": 1 for tag in r["phenomena"]}} for r in training]
    folds = _training_calibration_folds([{**r, "sentiment": r["intended_label"], "group_id": r["family"]} for r in training], 42)
    target = [r["intended_label"] for r in training]
    probe = make_pipeline(DictVectorizer(), LogisticRegression(C=1., max_iter=1000, random_state=42))
    guessed = cross_val_predict(probe, inputs, target, cv=folds)
    shape_inputs = [{"log_word_count": float(np.log1p(len(r["text"].split()))),
                     "sentences": float(len(re.findall(r"[.!?]+", r["text"]))),
                     "question": float("?" in r["text"]),
                     "colon": float(":" in r["text"]),
                     "semicolon": float(";" in r["text"])} for r in training]
    shape_probe = make_pipeline(DictVectorizer(), StandardScaler(with_mean=False),
                                LogisticRegression(C=1., max_iter=1000, random_state=42))
    shape_guessed = cross_val_predict(shape_probe, shape_inputs, target, cv=folds)
    endings = defaultdict(set)
    for row in training:
        words = re.findall(r"\w+", row["text"].casefold())
        endings[(row["intended_label"], " ".join(words[-4:]))].add(row["family"])
    repeats = [{"intended_label": label, "ending": phrase, "distinct_families": len(families)}
               for (label, phrase), families in endings.items() if len(families) >= 2]
    return {"scope": "TRAIN ONLY; metadata never supplied to sentiment estimators",
            "metadata_only_group_oof": summarize_predictions(np.asarray(target), guessed),
            "text_shape_only_group_oof": summarize_predictions(np.asarray(target), shape_guessed),
            "metadata_probe_features": ["domain", "aspect", "region", "length_band", "structure", "phenomena"],
            "recurrent_four_word_endings_across_families": sorted(repeats, key=lambda r: (-r["distinct_families"], r["ending"])),
            "interpretation": "Full metadata includes target-related semantic tags; shape-only uses length/punctuation. High predictability flags author/style correlations, not causal proof of shortcuts. Neither feature set enters sentiment models."}


def prepare(output: Path) -> dict:
    output = ignored_target(output, kinds=("data/processed", "data/private"))
    if output.exists() and any(output.iterdir()):
        raise CorpusError("Destino no vacío; no sobrescribir datos o evidencia.")
    catalog, protocol = (json.loads(p.read_text(encoding="utf-8")) for p in (CATALOG, PROTOCOL))
    rows = materialize(catalog)
    counts = coverage(rows)
    if {role: counts[role]["rows"] for role in counts} != protocol["role_rows"]:
        raise CorpusError("Roles difieren del protocolo fijado.")
    training_arms(rows)
    duplicates = duplicate_screen(rows, protocol["near_duplicate_threshold"])
    if duplicates["candidate_count"]:
        raise CorpusError("Cercanía léxica entre roles pendiente de revisión previa.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"report_type": "V3_SYNTHETIC_AUTHOR_INTENT_NOT_HUMAN_GOLD", "coverage": counts,
                "catalog_sha256": file_sha(CATALOG), "protocol_sha256": file_sha(PROTOCOL),
                "records_sha256": file_sha(output / "records.jsonl"), "duplicate_screen": duplicates,
                "provenance": catalog["provenance"], "v2_pre_generation_coverage": v2_coverage()}
    write_json(output / "manifest.json", manifest)
    return manifest


def read_prepared(root: Path):
    root = ignored_target(root, kinds=("data/processed", "data/private"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for key, path in (("catalog_sha256", CATALOG), ("protocol_sha256", PROTOCOL), ("records_sha256", root / "records.jsonl")):
        if manifest.get(key) != file_sha(path):
            raise CorpusError("Especificación o registros cambiados tras congelar.")
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if rows != materialize(json.loads(CATALOG.read_text(encoding="utf-8"))):
        raise CorpusError("Registros ajenos al catálogo registrado.")
    if manifest.get("coverage") != coverage(rows):
        raise CorpusError("Recuentos/dimensiones de manifiesto no corresponden a registros.")
    return rows, manifest, json.loads(PROTOCOL.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.out)
    except (CorpusError, ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"V3 preparation stopped: {type(exc).__name__}.\n")
    print(json.dumps({"status": result["report_type"], "catalog_sha256": result["catalog_sha256"],
                      "role_rows": {k: v["rows"] for k, v in result["coverage"].items()}}, indent=2))


if __name__ == "__main__":
    main()
