"""Inference adapters for the pinned academic V4 profile."""
from functools import lru_cache
from pathlib import Path
import joblib
from .corpus import file_sha
from .model_runtime import configured_artifact

def v4_selected():
    return configured_artifact() == Path("artifacts/synthetic-tourism-v4.2/beto-v4-partial256").resolve()

@lru_cache(maxsize=1)
def encoder():
    from .beto_classifier import BetoEncoder
    return BetoEncoder(Path("artifacts/beto-v3-pr28-local/checkpoint"), max_length=256)

class EmbeddingClassifier:
    context_verification = True

    def __init__(self, classifier, digest):
        self.artifact_digest = digest
        self.encoder = encoder()
        self.classifier = classifier
        self.classes_ = classifier.classes_

    def predict_proba(self, texts):
        return self.classifier.predict_proba(self.encoder.encode(texts))

    def predict(self, texts):
        return self.classifier.predict(self.encoder.encode(texts))

def embedding_model(kind):
    digests = {
        "lr": "08604c266d8c391f401394904cbf73ead9655044ecdb5173d2e70ff34dda966e",
        "svm": "2e74b31ed32c60845764fc8d307a45a344576ddea70d765e1098f77178b3fe44",
    }
    path = Path("artifacts/synthetic-tourism-v4.3") / f"beto_emb_{kind}.joblib"
    if kind not in digests or file_sha(path) != digests[kind]:
        raise ValueError("Artefacto contextual V4.3 incompatible.")
    return EmbeddingClassifier(joblib.load(path)["classifier"], digests[kind])
