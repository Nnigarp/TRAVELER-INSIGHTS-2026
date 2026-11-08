"""Versioned local BETO supervised adapter; never downloads during prediction.

The immutable base checkpoint is reused from the encoder probe. A trained head
is mandatory. Safe NumPy/safetensors files contain no executable Python objects.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from .beto_encoder_probe import MODEL, REVISION, WEIGHT_SHA256, checkpoint, file_sha
from .corpus import CorpusError
from .labels import LABELS
from .synthetic_diversity import ignored_target, write_json

SCHEMA = "traveler-beto-supervised-v1"
POOLING = "masked_mean_including_special_tokens_l2"


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise CorpusError("Logits supervisados inválidos.")
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


class BetoEncoder:
    def __init__(self, cache: Path, *, download=False, threads=2, max_length=128):
        if not 1 <= threads <= 8 or not 16 <= max_length <= 512:
            raise CorpusError("Presupuesto de encoder inválido.")
        self.manifest = checkpoint(cache, download=download)
        import torch
        from transformers import AutoModel, AutoTokenizer
        if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2, 6):
            raise CorpusError("El checkpoint legacy requiere torch>=2.6 y weights_only=True.")
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(cache), local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(cache), local_files_only=True,
            trust_remote_code=False, use_safetensors=False, weights_only=True,
            add_pooling_layer=False).eval()
        self.max_length, self.threads = max_length, threads
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def tokens(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")

    def pooled(self, tokens):
        import torch
        hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def encode(self, texts, *, batch_size=8):
        texts = list(texts)
        if (not texts or not 1 <= batch_size <= 32 or
                any(not isinstance(t, str) or not t.strip() or len(t) > 5000 for t in texts)):
            raise CorpusError("Textos o tamaño de lote inválidos.")
        import torch
        result = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                result.append(self.pooled(self.tokens(texts[start:start+batch_size])).cpu().numpy())
        values = np.concatenate(result).astype(np.float32)
        if values.shape != (len(texts), 768) or not np.isfinite(values).all():
            raise CorpusError("Embeddings inválidos.")
        return values


class BetoClassifier:
    """Implements the existing SentimentClassifier contract without agent edits."""
    classes_ = np.asarray(LABELS)

    def __init__(self, encoder, weight, bias, metadata):
        self.encoder, self.metadata = encoder, metadata
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        if (self.weight.shape != (3, 768) or self.bias.shape != (3,) or
                not np.isfinite(self.weight).all() or not np.isfinite(self.bias).all()):
            raise CorpusError("Cabeza de tres clases inválida.")

    def predict_features(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 768 or not np.isfinite(features).all():
            raise CorpusError("Representación contextual inválida.")
        return softmax(features @ self.weight.T + self.bias)

    def predict_proba(self, texts):
        return self.predict_features(self.encoder.encode(texts))

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def export_classifier(destination: Path, encoder_cache: Path, weight, bias,
                      training: dict, *, adaptation=None):
    destination = ignored_target(destination, kinds=("artifacts", "data/private"))
    if destination.exists():
        raise CorpusError("Artefacto ya existente; conservar versiones previas.")
    source = checkpoint(encoder_cache)
    # Portable and fully local after explicit preparation. No symlink to a cache.
    destination.mkdir(parents=True)
    portable_cache = destination / "checkpoint"
    portable_cache.mkdir()
    # Export only verified files. A shared cache can contain abandoned download
    # temporaries or unrelated files that must not become a model artifact.
    for name in source["files"]:
        shutil.copy2(encoder_cache / name, portable_cache / name)
    shutil.copy2(encoder_cache / "verified_checkpoint.json", portable_cache / "verified_checkpoint.json")
    np.savez(destination / "head.npz", weight=np.asarray(weight, dtype=np.float32),
             bias=np.asarray(bias, dtype=np.float32))
    metadata = {"schema": SCHEMA, "model": MODEL, "revision": REVISION,
        "base_weight_sha256": WEIGHT_SHA256, "checkpoint_manifest": source,
        "classes": list(LABELS), "pooling": POOLING, "max_length": 128,
        "head_sha256": file_sha(destination / "head.npz"), "head_parameters": 2307,
        "training": training, "human_validated": False,
        "scores": "UNCALIBRATED_NOT_A_SAFE_AUTOMATION_POLICY",
        "license": "Upstream CC BY 4.0 intent with original-corpus compatibility caveat; see checkpoint/README.md"}
    if adaptation:
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k,v in adaptation.items()},
                  str(destination / "adaptation.safetensors"))
        metadata["adaptation_sha256"] = file_sha(destination / "adaptation.safetensors")
        metadata["adapted_layer"] = 11
    write_json(destination / "model_manifest.json", metadata)
    return metadata


def load_classifier(artifact: Path, *, threads=2):
    artifact = ignored_target(artifact, kinds=("artifacts", "data/private"))
    metadata = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    if (metadata.get("schema") != SCHEMA or metadata.get("model") != MODEL or
            metadata.get("revision") != REVISION or metadata.get("classes") != list(LABELS) or
            metadata.get("pooling") != POOLING or metadata.get("max_length") != 128 or
            metadata.get("base_weight_sha256") != WEIGHT_SHA256 or
            not metadata.get("training", {}).get("supervised") or
            metadata.get("human_validated") is not False):
        raise CorpusError("Artefacto sin procedencia supervisada compatible.")
    if file_sha(artifact / "head.npz") != metadata["head_sha256"]:
        raise CorpusError("Cabeza modificada después de entrenar.")
    # Resolved descendants prevent accidentally loading symlinked external files.
    for name in ("head.npz", "checkpoint", "model_manifest.json"):
        if not (artifact / name).resolve().is_relative_to(artifact):
            raise CorpusError("Artefacto enlazado fuera de su versión.")
    encoder = BetoEncoder(artifact / "checkpoint", threads=threads)
    if encoder.manifest != metadata["checkpoint_manifest"]:
        raise CorpusError("Checkpoint distinto al usado por la cabeza supervisada.")
    if "adaptation_sha256" in metadata:
        if metadata.get("adapted_layer") != 11 or not (artifact / "adaptation.safetensors").resolve().is_relative_to(artifact):
            raise CorpusError("Adaptación fuera del contrato versionado.")
        if file_sha(artifact / "adaptation.safetensors") != metadata["adaptation_sha256"]:
            raise CorpusError("Adaptación modificada.")
        from safetensors.torch import load_file
        values = load_file(str(artifact / "adaptation.safetensors"), device="cpu")
        expected = {key for key in encoder.model.state_dict() if key.startswith("encoder.layer.11.")}
        if set(values) != expected:
            raise CorpusError("Adaptación ajena a la última capa permitida.")
        encoder.model.load_state_dict(values, strict=False)
    with np.load(artifact / "head.npz", allow_pickle=False) as head:
        if set(head.files) != {"weight", "bias"}:
            raise CorpusError("Contenido de cabeza inesperado.")
        return BetoClassifier(encoder, head["weight"], head["bias"], metadata)
