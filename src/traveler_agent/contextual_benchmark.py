"""Reproducible supervised BETO experiment, independent of the DEMO and V2/V3 tests."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .beto_classifier import BetoClassifier, BetoEncoder, export_classifier, load_classifier, softmax
from .contextual_data import load_experiment
from .corpus import CorpusError, file_sha
from .labels import LABELS
from .model import _text_features
from .research_metrics import score_matrix, summarize_predictions, slice_metrics, review_mask
from .synthetic_diversity import canonical_sha, ignored_target, write_json


def targets(rows):
    return np.asarray([LABELS.index(r["intended_label"]) for r in rows], dtype=np.int64)


def selection_score(truth, probabilities):
    probabilities = np.asarray(probabilities)
    guessed = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    macro = summarize_predictions(np.asarray(LABELS)[truth], guessed)["macro_f1"]
    loss = float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)).mean())
    return {"macro_f1": macro, "cross_entropy": loss}


def rank(record):
    return record["macro_f1"], -record["cross_entropy"]


def train_baselines(train, development, specification):
    texts = [r["text"] for r in train]
    labels = [r["intended_label"] for r in train]
    dev = [r["text"] for r in development]
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=specification["seed"])
                 .split(texts, labels, groups=[r["semantic_component"] for r in train]))
    for fit, held in folds:
        if {train[i]["semantic_component"] for i in fit} & {train[i]["semantic_component"] for i in held}:
            raise CorpusError("Fuga en calibración interna SVM.")
    models, records = {}, {}
    for kind in ("dummy", "lr", "svm"):
        choices = [None] if kind == "dummy" else specification["baselines"][kind+"_C"]
        candidates = []
        best = None
        started = time.perf_counter()
        for c in choices:
            if kind == "dummy":
                model = Pipeline([("features", _text_features()), ("classifier", DummyClassifier(strategy="most_frequent"))])
            elif kind == "lr":
                model = Pipeline([("features", _text_features()), ("classifier", LogisticRegression(C=c,
                    max_iter=1500, class_weight="balanced", random_state=specification["seed"]))])
            else:
                inner = Pipeline([("features", _text_features()), ("classifier", LinearSVC(C=c,
                    class_weight="balanced", random_state=specification["seed"]))])
                model = CalibratedClassifierCV(inner, cv=folds, method="sigmoid")
            model.fit(texts, labels)
            record = {"C": c, **selection_score(targets(development), score_matrix(model, dev))}
            candidates.append(record)
            if best is None or rank(record) > rank(best):
                best, models[kind] = record, model
        records[kind] = {"selected": best, "candidates_development_only": candidates,
                         "fit_and_selection_seconds": time.perf_counter()-started}
    return models, records


def fit_frozen_head(features, y, dev_features, dev_y, specification):
    import torch
    torch.manual_seed(specification["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(specification["threads"])
    config = specification["frozen_head"]
    head = torch.nn.Linear(768, 3)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    x, labels = torch.as_tensor(features), torch.as_tensor(y)
    generator = torch.Generator().manual_seed(specification["seed"])
    log, best, weights = [], None, None
    start = time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        order = torch.randperm(len(x), generator=generator)
        total = 0.
        for ids in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(x[ids]), labels[ids])
            loss.backward(); optimizer.step()
            total += loss.item()*len(ids)
        record = {"epoch": epoch, "train_cross_entropy": total/len(x)}
        if epoch in config["selection_epochs"]:
            w, b = head.weight.detach().numpy().copy(), head.bias.detach().numpy().copy()
            record["development"] = selection_score(dev_y, softmax(dev_features @ w.T+b))
            if best is None or rank(record["development"]) > rank(best["development"]):
                best, weights = dict(record), (w, b)
        log.append(record)
    return weights, {"selected": best, "epochs": log, "fit_and_selection_seconds": time.perf_counter()-start,
        "encoder_trainable_parameters": 0, "head_trainable_parameters": 2307}


def fit_partial(encoder, train, development, weight, bias, specification):
    import torch
    torch.manual_seed(specification["seed"])
    config = specification["partial_finetune"]
    layer = encoder.model.encoder.layer[11]
    layer.requires_grad_(True)
    # Eval disables stochastic dropout, not gradients. A deliberate technical arm.
    encoder.model.eval()
    head = torch.nn.Linear(768, 3)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weight)); head.bias.copy_(torch.from_numpy(bias))
    optimizer = torch.optim.AdamW([
        {"params": layer.parameters(), "lr": config["encoder_learning_rate"]},
        {"params": head.parameters(), "lr": config["head_learning_rate"]}], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(specification["seed"])
    y = torch.as_tensor(targets(train))
    best, selected, adaptation, log = None, None, None, []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        total = 0.
        for ids in torch.randperm(len(train), generator=generator).split(config["batch_size"]):
            texts = [train[i]["text"] for i in ids.tolist()]
            optimizer.zero_grad(set_to_none=True)
            features = encoder.pooled(encoder.tokens(texts))
            loss = torch.nn.functional.cross_entropy(head(features), y[ids])
            loss.backward(); optimizer.step()
            total += loss.item()*len(ids)
        w, b = head.weight.detach().numpy().copy(), head.bias.detach().numpy().copy()
        dev_features = encoder.encode([r["text"] for r in development], batch_size=specification["encode_batch_size"])
        record = {"epoch": epoch, "train_cross_entropy": total/len(train),
                  "development": selection_score(targets(development), softmax(dev_features @ w.T+b))}
        log.append(record)
        if epoch in config["selection_epochs"] and (best is None or rank(record["development"]) > rank(best["development"])):
            best, selected = dict(record), (w,b)
            adaptation = {key: value.detach().clone() for key,value in encoder.model.state_dict().items()
                          if key.startswith("encoder.layer.11.")}
    encoder.model.load_state_dict(adaptation, strict=False)
    layer.requires_grad_(False)
    return selected, adaptation, {"selected": best, "epochs": log,
        "fit_and_selection_seconds": time.perf_counter()-started,
        "encoder_trainable_parameters": sum(p.numel() for p in layer.parameters()),
        "head_trainable_parameters": 2307, "encoder_dropout": "disabled_by_protocol_not_missing_gradients"}


def resource_report():
    result = {"peak_rss_mib": None, "method": "PENDING_UNAVAILABLE"}
    if platform.system() == "Linux":
        import resource
        result = {"peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  "method": "Linux_process_high_water_includes_all_models_imports_and_optimizer"}
    else:
        try:
            import psutil
            info = psutil.Process().memory_info()
            peak = getattr(info, "peak_wset", None)
            if peak:
                result = {"peak_rss_mib": peak/2**20, "method": "Windows_process_peak_working_set"}
        except (ImportError, OSError):
            pass
    return result


def run(output: Path, cache: Path, *, download=False):
    output = ignored_target(output, kinds=("artifacts",))
    if output.exists():
        raise CorpusError("Usar un directorio nuevo; no sobrescribir evaluaciones observadas.")
    parts, manifest, specification = load_experiment()
    output.mkdir(parents=True)
    write_json(output/"data-manifest.json", manifest)
    write_json(output/"protocol.json", specification)
    start, cpu_start = time.perf_counter(), time.process_time()
    models, selections = train_baselines(parts["train"], parts["development"], specification)
    encode_start = time.perf_counter()
    encoder = BetoEncoder(cache, download=download, threads=specification["threads"], max_length=specification["max_length"])
    features = {role: encoder.encode([r["text"] for r in parts[role]], batch_size=specification["encode_batch_size"])
                for role in ("train", "development")}
    encoding_seconds = time.perf_counter()-encode_start
    (w,b), selections["beto_frozen"] = fit_frozen_head(features["train"], targets(parts["train"]),
                                        features["development"], targets(parts["development"]), specification)
    models["beto_frozen"] = BetoClassifier(encoder,w,b,{})
    training_base = {"supervised": True, "label_origin": "AUTHOR_INTENT_NOT_HUMAN_GOLD",
        "data_manifest": manifest, "seed": specification["seed"], "protocol_sha256": manifest["protocol_sha256"]}
    export_classifier(output/"models/beto_frozen",cache,w,b,{**training_base,"mode":"frozen_encoder_trained_head",
                      "selection": selections["beto_frozen"]["selected"]})
    if specification["partial_finetune"]["enabled"]:
        partial = BetoEncoder(cache, threads=specification["threads"], max_length=specification["max_length"])
        (pw,pb), adaptation, selections["beto_partial"] = fit_partial(partial,parts["train"],parts["development"],w,b,specification)
        models["beto_partial"] = BetoClassifier(partial,pw,pb,{})
        export_classifier(output/"models/beto_partial",cache,pw,pb,{**training_base,"mode":"last_encoder_block_plus_head",
                          "selection": selections["beto_partial"]["selected"]}, adaptation=adaptation)
    # Durable selection boundary BEFORE any assessment prediction/embedding.
    selection = {"schema":"contextual-selection-v1","data_manifest_sha256":file_sha(output/"data-manifest.json"),
                 "models":selections,"selected_using":"new_development_only","assessment_used":False}
    write_json(output/"selection.json",selection)
    evaluation, predictions, timings = {}, [], {}
    for kind, model in models.items():
        assessment = parts["assessment"]
        begin = time.perf_counter()
        probabilities = score_matrix(model,[r["text"] for r in assessment])
        inference_seconds = time.perf_counter()-begin
        guessed = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        truth = np.asarray([r["intended_label"] for r in assessment])
        # Historical fixed rules shown descriptively; no calibrated safety claim.
        reviewed = review_mask(probabilities,[r["text"] for r in assessment])
        ambiguous = score_matrix(model,[r["text"] for r in parts["ambiguity"]])
        amb_review = review_mask(ambiguous,[r["text"] for r in parts["ambiguity"]])
        evaluation[kind] = {"classification":summarize_predictions(truth,guessed,reviewed=reviewed),
            "slices":slice_metrics(assessment,truth,guessed,reviewed),
            "ambiguity":{"n":len(amb_review),"referred_by_unchanged_DEMO_rules":int(amb_review.sum()),
                         "sentiment_accuracy":None,"gold_sentiment_available":False},
            "errors":[{"review_id":r["review_id"],"family":r["family"],"text":r["text"],
                       "author_intent":str(t),"predicted":str(g),"accepted_by_unchanged_rules":not bool(rev)}
                      for r,t,g,rev in zip(assessment,truth,guessed,reviewed) if t!=g]}
        for row,target,guess,probs,refer in zip(assessment,truth,guessed,probabilities,reviewed):
            predictions.append({"model":kind,"review_id":row["review_id"],"family":row["family"],
                "author_intent":str(target),"prediction":str(guess),"probabilities":probs.tolist(),"review":bool(refer)})
        # Warm single-request end-to-end latency measured on new development only.
        samples=[]
        model.predict_proba([parts["development"][0]["text"]])
        for row in parts["development"][:6]:
            begin=time.perf_counter_ns();model.predict_proba([row["text"]])
            samples.append((time.perf_counter_ns()-begin)/1e6)
        timings[kind]={"assessment_batch_rows":len(assessment),"assessment_batch_seconds":inference_seconds,
                       "single_request_ms":samples,"p50_ms":float(np.percentile(samples,50)),
                       "p95_ms":float(np.percentile(samples,95)),"scope":"warm loaded model, tokenizer/vectorizer included, 6 requests, descriptive only"}
    # A real portable-artifact reload must agree, not merely an in-memory head.
    portable={}
    for kind in ("beto_frozen","beto_partial"):
        if kind not in models:continue
        reloaded=load_classifier(output/"models"/kind,threads=specification["threads"])
        texts=[r["text"] for r in parts["development"][:3]]
        delta=float(np.max(np.abs(reloaded.predict_proba(texts)-models[kind].predict_proba(texts))))
        if delta>1e-6:raise CorpusError("Artefacto supervisado no reproduce inferencia tras recarga.")
        portable[kind]={"executed":True,"max_probability_difference":delta,
                        "artifact_bytes":sum(p.stat().st_size for p in (output/"models"/kind).rglob('*') if p.is_file())}
        del reloaded
    report={"schema":"contextual-supervised-report-v1","human_evaluation":False,
        "scope":"NEW_SMALL_SYNTHETIC_ASSESSMENT_AUTHOR_INTENT_NOT_HUMAN_GOLD",
        "data_manifest":manifest,"protocol":specification,"selection_sha256":file_sha(output/"selection.json"),
        "results":evaluation,"latency":timings,"portable_reload":portable,
        "resources":{"total_wall_seconds":time.perf_counter()-start,"process_cpu_seconds":time.process_time()-cpu_start,
            "train_development_base_encoding_seconds":encoding_seconds,**resource_report(),"device":"cpu",
            "threads":specification["threads"],"cuda_used":False,"monetary_cost":"NOT_MEASURED"},
        "environment":{"python":platform.python_version(),"platform":platform.platform(),
            "packages":{n:version(n) for n in ('torch','transformers','tokenizers','numpy','scikit-learn','safetensors')}},
        "implementation_sha256":{n:file_sha(Path(__file__).with_name(n)) for n in
                                 ('contextual_benchmark.py','contextual_data.py','beto_classifier.py')},
        "limits":specification["limits"]+["Review policy is unchanged, descriptive, and not validated for either contextual arm.",
                   "Repeated reproduction with the same seed measures software reproducibility, not uncertainty over training seeds."]}
    write_json(output/"assessment_predictions.json",predictions)
    write_json(output/"report.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('artifacts/beto-supervised-v1/checkpoint'))
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args(argv)
    try:report=run(args.output,args.cache,download=args.download)
    except (CorpusError,ImportError,OSError,ValueError) as exc:
        parser.exit(2,f'Contextual experiment stopped: {type(exc).__name__}; inspect local setup.\n')
    print(json.dumps({'macro_f1_synthetic':{k:r['classification']['macro_f1'] for k,r in report['results'].items()},
                      'resources':report['resources']},indent=2))


if __name__=='__main__':main()
