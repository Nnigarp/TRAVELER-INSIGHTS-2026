"""Source-specific local adapters: task/label provenance is never inferred as gold."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .corpus import CorpusError, text_sha

def _hash(source, value):
    return hashlib.sha256((source + "\0" + str(value)).encode()).hexdigest()[:24]

def require_approval(path: Path, source: str):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorpusError("Falta aprobación local de derechos y privacidad.") from exc
    if d.get("source") != source or d.get("reuse_rights_reviewed") is not True or d.get("privacy_reviewed") is not True:
        raise CorpusError("No existe verificación de derechos originales y privacidad para esta fuente.")
    if not all(isinstance(d.get(x),str) and d[x].strip()
               for x in ("license_reference", "original_content_rights_basis", "reviewer_id")):
        raise CorpusError("Falta referencia de licencia, derechos originales o responsable.")
    return d

def _safe(text):
    from .corpus_automation import privacy_flags
    if not isinstance(text,str) or not 3 <= len(text.strip()) <= 5000 or privacy_flags(text):
        raise CorpusError("Texto inválido o posible dato personal; revisar antes de importar.")
    return text

def _weak_rating(value):
    if not isinstance(value,str) or value not in {"1","2","3","4","5"}:
        raise CorpusError("Se esperaba rating 1–5 literal, no sentimiento humano.")
    return "negativo" if value in {"1","2"} else "neutral" if value=="3" else "positivo"

def argilla_adapter(rows, approval_path):
    """Real HF schema has prediction list, no human annotation, no hotel ID."""
    require_approval(approval_path,"argilla_tripadvisor")
    records=[]
    for row in rows:
        if not isinstance(row,dict) or not all(k in row for k in ("id","text","prediction","annotation")):
            raise CorpusError("Argilla: esquema real no coincide.")
        if row["annotation"] is not None:
            raise CorpusError("No mezclar con hipotéticas anotaciones humanas.")
        pred=row["prediction"]
        if not isinstance(pred,list) or len(pred)!=1 or not isinstance(pred[0],dict):
            raise CorpusError("Argilla: prediction debe contener una única etiqueta de rating.")
        rating=str(pred[0].get("label",""))
        label=_weak_rating(rating)
        text=_safe(row["text"])
        key=str(row["id"])
        if not key or len(key)>128: raise CorpusError("ID inválido.")
        records.append({
            "review_id":"arg_"+_hash("argilla",key), "text":text,
            "source":"argilla_tripadvisor", "task":"global_weak_rating",
            "weak_label":label, "original_rating":rating,
            "label_origin":"prediction_rating_derived_NOT_HUMAN_GOLD",
            "language":"en", "group_id":"review_"+_hash("argilla",key),
            "group_quality":"HOTEL_UNKNOWN", "text_sha256":text_sha(text),
        })
    hashes={}
    for r in records:
        fingerprint=r["text_sha256"]
        if fingerprint in hashes and hashes[fingerprint]!=r["weak_label"]:
            raise CorpusError("Duplicados con rating contradictorio.")
        hashes[fingerprint]=r["weak_label"]
    return records

def crawlfeeds_adapter(rows, approval_path):
    """Keep positive and negative fields; discard reviewer names/addresses."""
    require_approval(approval_path,"crawlfeeds_booking")
    result=[]
    for row in rows:
        keys=("uniq_id","hotel_name","positive_review_text","negative_review_text","language")
        if not isinstance(row,dict) or any(k not in row for k in keys):
            raise CorpusError("CrawlFeeds: esquema no coincide.")
        pos=row["positive_review_text"]
        neg=row["negative_review_text"]
        pos=pos.strip() if isinstance(pos,str) and pos.strip()!=r"\N" else ""
        neg=neg.strip() if isinstance(neg,str) and neg.strip()!=r"\N" else ""
        if not pos and not neg:continue
        text=_safe("\n".join(x for x in (pos,neg) if x))
        hotel=str(row["hotel_name"]).strip()
        if not hotel or not str(row["uniq_id"]).strip():
            raise CorpusError("CrawlFeeds: faltan hotel o ID.")
        result.append({
            "review_id":"crawl_"+_hash("crawlfeeds",row["uniq_id"]),
            "text":text,"source":"crawlfeeds_booking","task":"mixed_segments",
            "weak_label":None,"label_origin":"no_global_review_label",
            "positive_segment":pos,"negative_segment":neg,
            "language":str(row["language"]) or "unknown",
            "group_id":"hotel_"+_hash("crawlfeeds",hotel),
            "group_quality":"hotel_name_not_verified_unique",
            "text_sha256":text_sha(text),
        })
    return result

def hamos_adapter(rows, approval_path, *, mapping):
    """Require names audited from downloaded files; preserve official hotel splits."""
    require_approval(approval_path,"hamos_v4")
    needed=("text","hotel_id","review_id","quadruples","split")
    if not isinstance(mapping,dict) or any(not mapping.get(k) for k in needed):
        raise CorpusError("HAMOS: el mapping oficial debe ser verificado y explícito.")
    result=[]; hotel_splits={}
    for row in rows:
        if not all(mapping[k] in row for k in needed):
            raise CorpusError("HAMOS: esquema real no coincide con el mapping.")
        split=row[mapping["split"]]
        if split not in ("train","validation","test"):
            raise CorpusError("HAMOS: split no reconocido; no reconstruir.")
        hotel=_hash("hamos_hotel",row[mapping["hotel_id"]])
        if hotel in hotel_splits and hotel_splits[hotel]!=split:
            raise CorpusError("HAMOS: fuga entre particiones de hotel.")
        hotel_splits[hotel]=split
        quads=row[mapping["quadruples"]]
        if not isinstance(quads,list):raise CorpusError("Cuádruplas inválidas.")
        text=_safe(row[mapping["text"]])
        result.append({
            "review_id":"hamos_"+_hash("hamos",row[mapping["review_id"]]),
            "text":text,"source":"hamos_v4","task":"native_absa_quadruples",
            "weak_label":None,"label_origin":"ABSA_not_global",
            "native_quadruples":quads,"official_split":split,
            "group_id":"hotel_"+hotel,"group_quality":"source_hotel_identifier",
            "language":str(row.get(mapping.get("language","language"),"unknown")),
            "text_sha256":text_sha(text),
        })
    return result

def cross_source_overlap(batches):
    seen={};pairs=[]
    for source in sorted(batches):
        for record in batches[source]:
            key=record["text_sha256"]
            if key in seen and seen[key]["source"]!=source:
                pairs.append({"first":seen[key]["id"],"second":record["review_id"],
                              "action":"REVIEW_BEFORE_COMBINING"})
            else:seen[key]={"source":source,"id":record["review_id"]}
    return {"exact_cross_source_pairs":pairs,
            "warning":"Near duplicates and user overlap still require review."}
