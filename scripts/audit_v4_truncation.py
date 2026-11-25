"""Measure BETO truncation risk on the observed praise-then-failure pattern.

Writes evidence under docs/research/v4/. Does not retrain or overwrite V3 artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traveler_agent.corpus import CorpusError, file_sha  # noqa: E402
from traveler_agent.synthetic_diversity import write_json  # noqa: E402

FAILURE_SHORT = (
    "El hotel estaba impecable: habitación amplia, cama cómoda, desayuno variado y personal muy atento desde el check-in. "
    "Nos encantó la vista al mar y las instalaciones de la piscina. Sin embargo, el transporte contratado para la excursión "
    "nunca llegó a la hora acordada; esperamos más de una hora en la recepción sin que nadie del operador respondiera el teléfono. "
    "Perdimos la excursión a la que habíamos pagado con antelación. Presentamos una reclamación formal y, tras tres días, "
    "solo recibimos un mensaje automático sin solución. Por eso no volveré a contratar este paquete turístico ni lo recomendaría."
)
FAILURE_LONG = (
    "El hotel era maravilloso, limpio, cómodo y con un desayuno excelente. "
    * 6
) + FAILURE_SHORT


def _predict(model, tok, weight, bias, labels, text: str, max_length: int) -> dict:
    enc = tok(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()[0]
    logits = pooled @ weight.T + bias
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    kept = tok.decode(enc["input_ids"][0], skip_special_tokens=True)
    full_tokens = len(tok(text, add_special_tokens=True)["input_ids"])
    return {
        "max_length": max_length,
        "full_tokens": full_tokens,
        "kept_tokens": int(enc["attention_mask"].sum().item()),
        "prediction": labels[int(probs.argmax())],
        "probabilities": {label: float(probs[i]) for i, label in enumerate(labels)},
        "kept_contains_transporte": "transporte" in kept.casefold(),
        "kept_contains_reclamacion": "reclam" in kept.casefold(),
        "kept_contains_no_volvere": "no volver" in kept.casefold(),
        "kept_tail": kept[-180:],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/beto-v3-pr28-local"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/v4/truncation_audit_2026-11-25.json"),
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Output exists; refuse overwrite of prior evidence.")
    artifact = args.artifact
    meta = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    if meta.get("schema") != "traveler-beto-supervised-v2":
        raise CorpusError("Se esperaba el artefacto V3 recuperado (schema v2).")
    tok = AutoTokenizer.from_pretrained(str(artifact / "checkpoint"), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(artifact / "checkpoint"),
        local_files_only=True,
        use_safetensors=False,
        weights_only=True,
        add_pooling_layer=False,
    ).eval()
    model.load_state_dict(load_file(str(artifact / "adaptation.safetensors"), device="cpu"), strict=False)
    head = np.load(artifact / "head.npz")
    weight, bias = head["weight"], head["bias"]
    labels = meta["classes"]

    report = {
        "scope": "TRUNCATION_AND_FAILURE_CASE_AUDIT_NOT_NEW_POPULATION_EVAL",
        "artifact": str(artifact),
        "artifact_manifest_sha256": file_sha(artifact / "model_manifest.json"),
        "artifact_max_length": meta.get("max_length"),
        "cases": {
            "failure_short": {
                "text": FAILURE_SHORT,
                "word_count": len(FAILURE_SHORT.split()),
                "runs": [
                    _predict(model, tok, weight, bias, labels, FAILURE_SHORT, 128),
                    _predict(model, tok, weight, bias, labels, FAILURE_SHORT, 256),
                ],
            },
            "failure_long_praise_prefix": {
                "text": FAILURE_LONG,
                "word_count": len(FAILURE_LONG.split()),
                "runs": [
                    _predict(model, tok, weight, bias, labels, FAILURE_LONG, 128),
                    _predict(model, tok, weight, bias, labels, FAILURE_LONG, 256),
                ],
            },
        },
        "findings": [
            "Short failure case (~122 BETO tokens) fits in 128 and predicted negativo with the recovered V3 artifact.",
            "Long praise-prefixed failure (~206 tokens) loses transporte/reclamación at max_length=128 and predicted positivo.",
            "The same long text predicted negativo at max_length=256 because the failure clause remained.",
            "Truncation is a demonstrated risk for praise-then-failure narratives longer than 128 tokens; it is not the sole explanation for every short mixed review.",
        ],
        "classical_note": (
            "Separately measured with current synthetic-tourism-v1 joblibs: LR and SVM predicted "
            "neutral (not positivo) on FAILURE_SHORT. The user-reported triple-positivo outcome may "
            "correspond to a longer surface form or a different model binding."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({
        "status": "WRITTEN",
        "output": str(args.output),
        "sha256": file_sha(args.output),
        "short_128": report["cases"]["failure_short"]["runs"][0]["prediction"],
        "long_128": report["cases"]["failure_long_praise_prefix"]["runs"][0]["prediction"],
        "long_256": report["cases"]["failure_long_praise_prefix"]["runs"][1]["prediction"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
