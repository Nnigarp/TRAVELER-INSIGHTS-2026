"""Replay a frozen, assistant-authored Spanish tourism research catalog.

No noun/adjective substitution, API calls, human authors or human gold labels.
The seed controls subsequent sampling; it does not recreate LLM drafting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .corpus import CorpusError, file_sha, text_sha
from .labels import LABELS

RESOURCE = Path(__file__).with_name("research_data")
CATALOG = RESOURCE / "synthetic_diversity_v2.json"
PROTOCOL = RESOURCE / "synthetic_v2_protocol.json"
ROLES = ("train", "validation", "challenge")
SOURCE = "NEW_SYNTHETIC_ASSISTANT_AUTHORED_V2"
PHENOMENA = frozenset({"negation", "double_negation", "negative_concordance", "contrast",
                       "mixed_aspects", "positive_words_negative_context", "regionalism", "typo"})


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def ignored_target(path: Path, *, kinds=("data/processed", "artifacts")) -> Path:
    root, target = Path.cwd().resolve(), Path(path).resolve()
    if not any(target.is_relative_to(root / name) for name in kinds):
        raise CorpusError("Usar una ruta de investigación ignorada desde la raíz del proyecto.")
    return target


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def materialize(catalog: dict) -> list[dict]:
    if (catalog.get("version") != "synthetic-diversity-v2.0.0"
            or catalog.get("provenance", {}).get("human_authors") is not False
            or catalog.get("provenance", {}).get("label_origin") != "AUTHOR_INTENT_NOT_HUMAN_GOLD"):
        raise CorpusError("No atribuir autores o etiquetas humanas al catálogo sintético.")
    rows, families = [], set()
    for family in catalog["families"]:
        key, role = family["family"], family["role"]
        if key in families or role not in ROLES or family["region"] not in {"es", "es-CO", "es-PE"}:
            raise CorpusError("Familia, partición o región declarada inválida.")
        families.add(key)
        cases = family["cases"]
        if len(cases) != 3 or {c["intended_label"] for c in cases} != set(LABELS):
            raise CorpusError("Cada familia etiquetada requiere tres intenciones distintas.")
        for case in cases:
            if (not set(case["phenomena"]) <= PHENOMENA or not case["rationale"].strip()
                    or not case["discourse"].strip()):
                raise CorpusError("Metadatos de intención incompletos.")
            rows.append({"family": key, "role": role, "source": SOURCE,
                         "region": family["region"], **case, "intended_review": None})
    for field, role in (("ambiguity_cohort", "ambiguity"),
                        ("illustrative_user_examples", "illustrative")):
        for case in catalog[field]:
            if (case["family"] in families or case.get("intended_label") is not None
                    or case["role"] != role):
                raise CorpusError("Casos ambiguos/ilustrativos no son etiquetas de sentimiento.")
            families.add(case["family"])
            rows.append({**case, "source": "SYNTHETIC_AMBIGUITY" if role == "ambiguity"
                         else "USER_ILLUSTRATIVE_NOT_GOLD", "region": "unspecified",
                         "phenomena": [], "discourse": role,
                         "rationale": "Unlabeled; never fit or score sentiment accuracy."})
    seen = set()
    for row in rows:
        text = row["text"]
        if not isinstance(text, str) or not 3 <= len(text) <= 2500:
            raise CorpusError("Texto sintético vacío o fuera de rango.")
        token = text_sha(text)
        if token in seen:
            raise CorpusError("Duplicado normalizado dentro del catálogo; revisar antes del sellado.")
        seen.add(token)
        row["text_sha256"] = token
        row["review_id"] = "v2_" + canonical_sha([row["family"], row["intended_label"], text])[:20]
        words = len(text.split())
        row["length_band"] = "short" if words <= 12 else "long" if words >= 80 else "medium"
        row["label_origin"] = "AUTHOR_INTENT_NOT_HUMAN_GOLD"
    return sorted(rows, key=lambda row: row["review_id"])


def lexical_audit(rows: list[dict], *, threshold: float = .70) -> dict:
    """Unfitted all-pairs Jaccard screen; no learned vocabulary from challenge."""
    candidates, maximum, maximum_ids = [], 0., []
    signatures = []
    labeled = [r for r in rows if r["role"] in ROLES]
    for row in labeled:
        text = " ".join(re.findall(r"\w+", row["text"].casefold()))
        words = text.split()
        signatures.append((set(text[i:i+5] for i in range(max(0, len(text)-4))),
                           set(tuple(words[i:i+3]) for i in range(max(0, len(words)-2)))))
    for i, row in enumerate(labeled):
        for j in range(i):
            other = labeled[j]
            if row["role"] == other["role"]:
                continue
            similarities = []
            for left, right in zip(signatures[i], signatures[j]):
                similarities.append(len(left & right) / len(left | right) if left | right else 0.)
            similarity = max(similarities)
            if similarity > maximum:
                maximum = similarity
                maximum_ids = [row["review_id"], other["review_id"]]
            if similarity >= threshold:
                candidates.append({"ids": [row["review_id"], other["review_id"]],
                                   "similarity": round(similarity, 6)})
    return {"method": "unfitted_char5_word3_jaccard_all_pairs_across_new_roles",
            "threshold": threshold, "maximum_similarity": maximum,
            "maximum_pair_ids": maximum_ids, "candidates": candidates,
            "semantic_paraphrase_independence_proven": False}


def summarize(rows: list[dict]) -> dict:
    result = {}
    for role in (*ROLES, "ambiguity", "illustrative"):
        subset = [r for r in rows if r["role"] == role]
        result[role] = {
            "rows": len(subset), "families": len({r["family"] for r in subset}),
            "labels": dict(sorted(Counter(r["intended_label"] for r in subset
                                           if r["intended_label"] is not None).items())),
            "length_bands": dict(sorted(Counter(r["length_band"] for r in subset).items())),
            "phenomena": dict(sorted(Counter(t for r in subset for t in r["phenomena"]).items())),
            "regions": dict(sorted(Counter(r["region"] for r in subset).items())),
        }
    return result


def prepare(output: Path) -> dict:
    output = ignored_target(output, kinds=("data/processed", "data/private"))
    if output.exists() and any(output.iterdir()):
        raise CorpusError("Destino no vacío; conservar el corpus anterior y usar una versión nueva.")
    catalog, protocol = json.loads(CATALOG.read_text(encoding="utf-8")), json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = materialize(catalog)
    counts = summarize(rows)
    for role, expected in protocol["roles"].items():
        if counts[role]["families"] != expected:
            raise CorpusError("El catálogo difiere de las familias predefinidas en el protocolo.")
    audit = lexical_audit(rows, threshold=protocol["near_duplicate_audit"]["jaccard_threshold"])
    if audit["candidates"]:
        raise CorpusError("Candidatos a duplicados entre particiones; revisar antes de entrenar.")
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "records.jsonl"
    data_path.write_bytes("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8"))
    manifest = {
        "report_type": "FROZEN_SYNTHETIC_AUTHOR_INTENT_NOT_HUMAN_GOLD",
        "catalog_sha256": file_sha(CATALOG), "protocol_sha256": file_sha(PROTOCOL),
        "records_sha256": file_sha(data_path), "version": catalog["version"],
        "provenance": catalog["provenance"], "counts": counts, "lexical_audit": audit,
        "human_corpus_admitted": False, "scientific_training_executed": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def read_prepared(root: Path) -> tuple[list[dict], dict, dict]:
    root = ignored_target(root, kinds=("data/processed", "data/private"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (manifest.get("catalog_sha256") != file_sha(CATALOG)
            or manifest.get("protocol_sha256") != file_sha(PROTOCOL)
            or manifest.get("records_sha256") != file_sha(root / "records.jsonl")):
        raise CorpusError("Corpus o protocolo cambiado después del sellado.")
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if rows != materialize(json.loads(CATALOG.read_text(encoding="utf-8"))):
        raise CorpusError("Los datos no corresponden al catálogo sintético fijado.")
    return rows, manifest, protocol


def historical_train() -> tuple[list[dict], dict]:
    """Reconstruct original TRAIN only. Old held-out scores/texts are not used.

    The historical generator/splitter materialize partitions in memory; only
    TRAIN is returned. No historical evaluator or artifact writer is called.
    """
    from .synthetic_tourism import generate_records, ENDING, NEGATION, CONTRAST
    from .synthetic_evaluation import group_split
    raw = group_split(generate_records(seed=42, variants=2, generated_date="2026-11-05"), seed=42)["train"]
    if len(raw) != 204:
        raise CorpusError("TRAIN histórico no coincide con el protocolo.")
    phrases = []
    for pool in (ENDING, NEGATION, CONTRAST):
        for label, texts in pool.items():
            for text in ([texts] if isinstance(texts, str) else texts):
                n = sum(text.casefold() in r["text"].casefold() for r in raw if r["target_label"] == label)
                phrases.append({"label": label, "training_occurrences": n})
    rows = []
    for r in raw:
        rows.append({"review_id": r["review_id"], "text": r["text"],
                     "intended_label": r["target_label"], "family": r["scenario_family"],
                     "source": "HISTORICAL_V1_TRAIN_ONLY", "role": "train", "style": r["style"],
                     "text_sha256": text_sha(r["text"])})
    diagnostic = {"rows": len(rows), "families": len({r["family"] for r in rows}),
                  "historical_generator_sha256": file_sha(Path(__file__).with_name("synthetic_tourism.py")),
                  "fixed_authored_class_phrase_occurrences_in_TRAIN": phrases,
                  "old_held_out_scores_recomputed": False,
                  "interpretation": "Repeated class-linked clauses support a shortcut hypothesis, not a causal attribution of the historical score drop."}
    return rows, diagnostic


def select_families(rows: list[dict], count: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    if len(groups) < count:
        raise CorpusError("Familias insuficientes para el presupuesto fijado.")
    chosen = sorted(groups, key=lambda key: canonical_sha([seed, key]))[:count]
    result = []
    for key in chosen:
        for label in LABELS:
            options = [r for r in groups[key] if r["intended_label"] == label]
            preferred = [r for r in options if r.get("style") == "narrative_0"] or options
            if not preferred:
                raise CorpusError("Familia sin una clase requerida.")
            result.append(min(preferred, key=lambda r: r["review_id"]))
    return result


def assemble_strategies(rows: list[dict], protocol: dict) -> tuple[dict, dict]:
    old, diagnostic = historical_train()
    new = [r for r in rows if r["role"] == "train"]
    per_family = protocol["labeled_rows_per_family"]
    families = protocol["main_training_budget"] // per_family
    strategies = {
        "A_historical_matched": select_families(old, families, protocol["seed"]),
        "B_new": select_families(new, families, protocol["seed"]),
        "C_mixed_matched": select_families(old, families // 2, protocol["seed"])
                           + select_families(new, families // 2, protocol["seed"]),
        "A_historical_full_sensitivity": old,
    }
    held = {r["text_sha256"] for r in rows if r["role"] != "train"}
    for name, training in strategies.items():
        if any(r["role"] != "train" or r["intended_label"] not in LABELS for r in training):
            raise CorpusError("Ninguna fila de desarrollo/test/sin etiqueta puede entrar en entrenamiento.")
        if any(r["text_sha256"] in held for r in training):
            raise CorpusError("Texto de entrenamiento compartido con un rol reservado nuevo.")
        if name != "A_historical_full_sensitivity" and len(training) != protocol["main_training_budget"]:
            raise CorpusError("Presupuesto de entrenamiento descontrolado.")
    return strategies, diagnostic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.out)
    except (CorpusError, ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Synthetic preparation stopped: {type(exc).__name__}.\n")
    print(json.dumps({"status": manifest["report_type"], "counts": manifest["counts"],
                      "catalog_sha256": manifest["catalog_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
