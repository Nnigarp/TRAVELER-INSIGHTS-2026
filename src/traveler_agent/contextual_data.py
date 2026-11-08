"""Small representation experiment: existing TRAIN, new fixed development/assessment.

Synthetic author intent only. Historical evaluated examples enter only an exact
hash denylist, never fitting, model selection or drafting. No text generation API.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .corpus import CorpusError, file_sha, text_sha
from .labels import LABELS
from .synthetic_diversity import canonical_sha
from .research_v3_data import duplicate_screen

RESOURCE = Path(__file__).with_name("research_data")
CATALOG = RESOURCE / "contextual_supervised_v1.json"
PROTOCOL = RESOURCE / "contextual_supervised_protocol_v1.json"
V3 = RESOURCE / "synthetic_diversity_v3.json"


def load_experiment():
    specification = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    new = json.loads(CATALOG.read_text(encoding="utf-8"))
    if (new.get("version") != "contextual-supervised-v1" or
            new.get("provenance", {}).get("human_authors") is not False or
            new["provenance"].get("label_origin") != "AUTHOR_INTENT_NOT_HUMAN_GOLD"):
        raise CorpusError("No atribuir gold o autores humanos a este experimento.")
    old = json.loads(V3.read_text(encoding="utf-8"))
    rows = []
    # A fixed broad_120 arm, chosen by protocol, not the highest historical score.
    for family in old["families"]:
        if family["role"] != "train":
            continue
        for case in family["cases"]:
            if case["variant"] >= 2:
                continue
            rows.append({**case, "family": family["family"], "role": "train",
                         "semantic_component": family["semantic_component"],
                         "region": family["region"], "domain": family["domain"],
                         "source": "V3_TRAIN_ONLY_FIXED_BROAD_120"})
    if len(rows) != 120 or len({r["family"] for r in rows}) != 20:
        raise CorpusError("TRAIN difiere del presupuesto predefinido.")
    families = {r["family"] for r in rows}
    for family in new["families"]:
        fid, role = family["family"], family["role"]
        if fid in families or role not in {"development", "assessment", "ambiguity"}:
            raise CorpusError("Familia nueva duplicada o rol inválido.")
        families.add(fid)
        targets = [c["intended_label"] for c in family["cases"]]
        if ((role != "ambiguity" and sorted(targets) != sorted(LABELS)) or
                (role == "ambiguity" and targets != [None])):
            raise CorpusError("Familia sin las tres intenciones o ambigüedad etiquetada.")
        for case in family["cases"]:
            if not case.get("rationale", "").strip():
                raise CorpusError("Falta justificación de intención sintética.")
            rows.append({**case, **{k: family[k] for k in
                         ("family", "role", "semantic_component", "domain", "region")},
                         "source": "NEW_ASSISTANT_AUTHORED_CONTEXTUAL_V1"})
    seen, components = set(), {}
    for row in rows:
        if not isinstance(row["text"], str) or not 8 <= len(row["text"]) <= 3000:
            raise CorpusError("Texto fuera de rango.")
        digest = text_sha(row["text"])
        if digest in seen:
            raise CorpusError("Duplicado normalizado dentro del experimento.")
        seen.add(digest)
        component = row["semantic_component"]
        if component in components and components[component] != row["role"]:
            raise CorpusError("Componente compartido entre particiones.")
        components[component] = row["role"]
        row["review_id"] = "ctx_" + canonical_sha([row["family"], digest])[:20]
        row["text_sha256"] = digest
        row["label_origin"] = "AUTHOR_INTENT_NOT_HUMAN_GOLD"
        words = len(row["text"].split())
        row["length_band"] = "short" if words <= 12 else "long" if words >= 70 else "medium"
    parts = {role: [r for r in rows if r["role"] == role]
             for role in ("train", "development", "assessment", "ambiguity")}
    if {k: len(v) for k, v in parts.items()} != specification["role_rows"]:
        raise CorpusError("Recuentos distintos del protocolo sellado.")
    # Evaluated historical rows used only for exact exclusion, no score or labels.
    forbidden = {text_sha(c["text"]) for f in old["families"] if f["role"] != "train"
                 for c in f["cases"]}
    forbidden.update(text_sha(c["text"]) for c in old["ambiguity"])
    v2 = json.loads((RESOURCE / "synthetic_diversity_v2.json").read_text(encoding="utf-8"))
    forbidden.update(text_sha(c["text"]) for f in v2["families"] for c in f["cases"])
    for field in ("ambiguity_cohort", "illustrative_user_examples"):
        forbidden.update(text_sha(c["text"]) for c in v2[field])
    if seen & forbidden:
        raise CorpusError("Reutilización de texto histórico evaluado/ilustrativo.")
    screens = duplicate_screen(rows, specification["duplicate_jaccard_threshold"])
    audit = {"method": "unfitted_char5_word3_all_six_role_pairs",
             "threshold": specification["duplicate_jaccard_threshold"],
             "candidates": [c for pair in screens["pairs"] for c in pair["candidates"]],
             "maximum_similarity": max(pair["maximum_similarity"] for pair in screens["pairs"]),
             "role_pair_screens": screens["pairs"], "semantic_independence_proven": False}
    if audit["candidates"]:
        raise CorpusError("Candidatos a duplicación entre roles; revisar antes de ejecutar.")
    manifest = {"schema": "contextual-representation-data-v1",
        "catalog_sha256": file_sha(CATALOG), "protocol_sha256": file_sha(PROTOCOL),
        "v3_training_source_sha256": file_sha(V3), "records_sha256": canonical_sha(rows),
        "counts": {role: {"rows": len(part), "families": len({r["family"] for r in part}),
                            "labels": dict(Counter(r["intended_label"] for r in part if r["intended_label"]))}
                   for role, part in parts.items()},
        "duplicate_screen": audit, "historical_evaluated_rows_used_for_selection": False,
        "human_gold": False, "semantic_independence_proven": False}
    return parts, manifest, specification
