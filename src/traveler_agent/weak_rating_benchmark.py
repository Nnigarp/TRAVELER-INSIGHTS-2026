"""Rating-proxy benchmark for approved Argilla records, NOT gold sentiment.

Separate from the adjudicated-corpus benchmark. Test measures agreement with
1–5 star-derived weak labels, not independently annotated tourist sentiment.
Hotel-disjoint claims are prohibited when hotel IDs are unavailable.
"""
from __future__ import annotations
import json
import platform
from collections import Counter
from importlib.metadata import version as package_version
import joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .corpus import CorpusError, file_sha
from .scientific_benchmark import _new_estimator, _score, MODEL_GRID
from .labels import LABELS


def _split_weak(rows,seed):
    # Exact duplicates share fingerprint, so they cannot cross partitions.
    groups=[r["text_sha256"] for r in rows]
    y=[r["weak_label"] for r in rows]
    if len(set(groups))<30 or set(y)!=set(LABELS):
        raise CorpusError("Necesitamos >=30 grupos y tres etiquetas débiles.")
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    working,test_idx=next(splitter.split(rows,y,groups))
    other=[rows[i] for i in working]
    vg=[r["text_sha256"] for r in other]
    splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1)
    train_idx,val_idx=next(splitter.split(other,[r["weak_label"] for r in other],vg))
    train=[other[i] for i in train_idx]
    val=[other[i] for i in val_idx]
    test=[rows[i] for i in test_idx]
    for name,subset in (("train",train),("validation",val),("test",test)):
        if set(r["weak_label"] for r in subset)!=set(LABELS):
            raise CorpusError("Faltan clases débiles en "+name+"; no evaluar.")
    keys=[{r["text_sha256"] for r in subset} for subset in (train,val,test)]
    if keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2]:
        raise CorpusError("Fuga entre particiones por duplicados exactos.")
    return train,val,test


def run_rating_proxy_benchmark(records_path:Path,out_dir:Path,*,seed=42):
    path=Path(records_path)
    if "private" not in path.parts:
        raise CorpusError("Entradas solamente desde carpeta privada.")
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    if not rows or any(r.get("source")!="argilla_tripadvisor"
        or r.get("task")!="global_weak_rating"
        or r.get("group_quality")!="HOTEL_UNKNOWN"
        or r.get("label_origin")!="prediction_rating_derived_NOT_HUMAN_GOLD"
        for r in rows):
        raise CorpusError("Se requieren exclusivamente etiquetas débiles Argilla con procedencia.")
    train,validation,test=_split_weak(rows,seed)
    if min(Counter(r["weak_label"] for r in train).values())<3:
        raise CorpusError("Calibración SVM requiere al menos tres ejemplos por clase.")
    # Adapter never claims hotel grouping; fold separation is by duplicate hash.
    from sklearn.model_selection import StratifiedGroupKFold
    inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    x=[r["text"] for r in train]
    y=[r["weak_label"] for r in train]
    groups=[r["text_sha256"] for r in train]
    folds=list(inner.split(x,y,groups))
    if any(set(y[i] for i in fit)!=set(LABELS) or
           set(y[i] for i in cal)!=set(LABELS) for fit,cal in folds):
        raise CorpusError("La calibración interna carece de clases completas.")
    report={}
    selected_models={}
    def convert(subset):
        return [{"text":r["text"],"sentiment":r["weak_label"],"language":r["language"]} for r in subset]
    for kind,grid in MODEL_GRID.items():
        candidates=[]
        for params in grid:
            clf=_new_estimator(kind,params,seed)
            if kind=="svm": clf.set_params(cv=folds)
            clf.fit(x,y)
            val_metric=_score(clf,convert(validation))
            candidates.append((val_metric["macro_f1"],params,clf,val_metric))
        best=max(enumerate(candidates),key=lambda a:(a[1][0],-a[0]))[1]
        score,params,clf,val_metric=best
        selected_models[kind]=clf
        report[kind]={
            "selected_parameters":params,"validation":val_metric,
            "test":_score(clf,convert(test))
        }
    out=Path(out_dir)
    if "artifacts" not in out.parts and "reports" not in out.parts:
        raise CorpusError("Salida privada en artifacts/reports.")
    if out.exists() and any(out.iterdir()):
        raise CorpusError("No sobrescribir resultados experimentales.")
    out.mkdir(parents=True,exist_ok=True)
    model_hashes={}
    for kind,clf in selected_models.items():
        path=out/(kind+".joblib")
        joblib.dump(clf,path)
        model_hashes[kind]=file_sha(path)
    result={
        "model_hashes":model_hashes,
        "runtime_versions":{"python":platform.python_version(),
                            "scikit-learn":package_version("scikit-learn"),
                            "numpy":package_version("numpy"),
                            "joblib":package_version("joblib")},
        "report_type":"WEAK_RATING_PROXY_NOT_HUMAN_SENTIMENT",
        "model_training":"REAL_TEXT_WEAK_LABELS_ONLY_IF_RIGHTS_APPROVED_BY_OPERATOR",
        "source":"argilla_tripadvisor","seed":seed,
        "input_sha256":file_sha(path),
        "split_grouping":"exact_duplicate_only_HOTEL_UNKNOWN",
        "sizes":{"train":len(train),"validation":len(validation),"test":len(test)},
        "models":report,
        "limitations":[
            "Test measures ratings-proxy agreement, not human sentiment accuracy.",
            "Hotel IDs unavailable; hotel-level separation NOT guaranteed.",
            "No Spanish evaluation. Source records tagged English.",
            "Source rights checked by operator, not verified by program.",
            "Model selection on validation only; test locked for one-time report.",
        ]
    }
    (out/"rating_proxy_report.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result



def main(argv=None):
    """Explicit CLI for an already-approved, locally imported Argilla corpus."""
    import argparse
    parser=argparse.ArgumentParser(description="Rating-proxy benchmark, not human sentiment evaluation")
    parser.add_argument("--records",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args(argv)
    try:
        result=run_rating_proxy_benchmark(args.records,args.out,seed=args.seed)
    except (CorpusError,OSError,ValueError):
        parser.exit(2,"Benchmark blocked: verify approved source, classes and split constraints.\n")
    print(json.dumps({
        "status":"ok","report_type":result["report_type"],
        "n":result["sizes"],"models":sorted(result["models"]),
        "warning":"Only rating-derived WEAK labels; no human gold, no Spanish validation."
    },ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
