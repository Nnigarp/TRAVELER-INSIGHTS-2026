"""Rescore deployed V4 classifiers on unique texts without fitting."""
import json
from pathlib import Path

from sklearn.metrics import f1_score, recall_score
from traveler_agent.beto_classifier import load_classifier
from traveler_agent.v4_runtime import embedding_model


def main():
    models = {kind: embedding_model(kind) for kind in ("lr", "svm")}
    models["beto"] = load_classifier(Path("artifacts/synthetic-tourism-v4.2/beto-v4-partial256"))
    report = {}
    for version in ("v2", "v3"):
        path = Path(f"src/traveler_agent/research_data/v4_independent_eval_{version}.json")
        rows = json.loads(path.read_text(encoding="utf8"))["reviews"]
        unique = {}
        for row in rows:
            key = row["text_sha256"]
            if key in unique and unique[key]["intended_label"] != row["intended_label"]:
                raise ValueError("Etiquetas contradictorias para un mismo texto.")
            unique[key] = row
        selected = list(unique.values())
        index = {row["text_sha256"]: i for i, row in enumerate(selected)}
        report[version] = {}
        for kind, clf in models.items():
            pred = clf.predict([r["text"] for r in selected])
            metrics = {}
            for group, sample in (("rows", rows), ("unique", selected)):
                truth = [r["intended_label"] for r in sample]
                votes = [pred[index[r["text_sha256"]]] for r in sample]
                metrics[group] = dict(n=len(sample), macro_f1=f1_score(truth, votes, average="macro"),
                                      negative_recall=recall_score(truth, votes, labels=["negativo"], average=None)[0])
            report[version][kind] = metrics
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
