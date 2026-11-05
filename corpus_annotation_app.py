"""Private standalone Gradio annotator for a locally authorized corpus.

No public sharing, no raw-text exports. This does not auto-label sentiments.
"""
from __future__ import annotations
import json
from pathlib import Path

import gradio as gr

from traveler_agent.corpus import (
    CorpusError, record_annotation, parse_aspects, validate_reviews
)
from traveler_agent.corpus_review import (
    record_evidence, validate_span_entries, annotator_agreement,
    near_duplicate_candidates
)
from traveler_agent.corpus import validate_annotations

ROOT = Path("data/private/tourism-corpus-v1")


def available_reviews() -> list[str]:
    try:
        records, _ = validate_reviews(ROOT / "reviews.csv", ROOT / "sources.csv")
        return [row["review_id"] for row in records]
    except (FileNotFoundError, CorpusError, OSError):
        return []


def read_review(review_id: str) -> str:
    try:
        records, _ = validate_reviews(ROOT / "reviews.csv", ROOT / "sources.csv")
        for row in records:
            if row["review_id"] == review_id:
                return row["text"]
        return "Reseña no encontrada."
    except (FileNotFoundError, CorpusError, OSError):
        return "No hay corpus local autorizado. Primero usa corpus_cli init y registra fuentes válidas."


def preview_evidence(review_id: str, spans_json: str) -> str:
    try:
        text = read_review(review_id)
        if not isinstance(text, str) or text.startswith("No hay corpus") or text == "Reseña no encontrada.":
            raise CorpusError("Reseña no disponible.")
        spans = json.loads(spans_json)
        checked = validate_span_entries(text, spans)
        return json.dumps(checked, ensure_ascii=False, indent=2)
    except (ValueError, TypeError, CorpusError) as exc:
        return "Error: " + str(exc)


def submit_annotation(review_id: str, annotator_id: str, sentiment: str,
                      aspects_json: str, spans_json: str) -> str:
    try:
        records, _ = validate_reviews(ROOT / "reviews.csv", ROOT / "sources.csv")
        record = next((r for r in records if r["review_id"] == review_id), None)
        if record is None:
            raise CorpusError("Reseña inexistente.")
        aspects = parse_aspects(aspects_json)
        spans = validate_span_entries(record["text"], json.loads(spans_json))
        if any(aspects.get(s["aspect"]) != s["polarity"] for s in spans):
            raise CorpusError("Evidencia y polaridad anotada no coinciden.")
        path = ROOT / "annotations.csv"
        record_annotation(path, {r["review_id"] for r in records},
                          review_id=review_id, annotator_id=annotator_id,
                          sentiment=sentiment, aspects=aspects)
        if spans:
            record_evidence(ROOT / "evidence_spans.csv", review_id, annotator_id,
                            record["text"], spans, aspects)
        return "Anotación guardada localmente; revisar adjudicación por separado."
    except (CorpusError, ValueError, OSError, TypeError) as exc:
        return "Error: " + str(exc)


def quality_preview():
    try:
        reviews, _ = validate_reviews(ROOT / "reviews.csv", ROOT / "sources.csv")
        annotations, _ = validate_annotations(
            ROOT / "annotations.csv", {r["review_id"] for r in reviews}
        )
        return json.dumps({
            "acuerdo": annotator_agreement(annotations),
            "casi_duplicados_para_revision": near_duplicate_candidates(reviews),
        }, ensure_ascii=False, indent=2)
    except (CorpusError, OSError, FileNotFoundError) as exc:
        return "Error: " + str(exc)


with gr.Blocks(title="Traveler Insights | Anotación privada") as app:
    gr.Markdown(
        "# Corpus turístico — anotación humana LOCAL\n"
        "No compartir ni publicar esta interfaz; los textos son privados. "
        "Los aspectos, polaridades y offsets son ingresados y revisados por personas. "
        "Los candidatos a casi duplicados no se eliminan automáticamente."
    )
    review_id = gr.Dropdown(label="Reseña a anotar", choices=available_reviews())
    refresh = gr.Button("Actualizar lista local")
    text = gr.Textbox(label="Texto original (sólo lectura)", lines=5, interactive=False)
    refresh.click(lambda: gr.update(choices=available_reviews()), outputs=review_id)
    review_id.change(read_review, inputs=review_id, outputs=text)
    annotator = gr.Textbox(label="ID seudónimo del anotador (p. ej. a01)")
    sentiment = gr.Dropdown(
        label="Sentimiento global",
        choices=["positivo", "neutral", "negativo", "desconocido"],
        value="desconocido"
    )
    aspects = gr.Textbox(label="Aspectos y polaridades JSON", value="{}",
                         info='Ejemplo: {"higiene":"negativo","servicio":"positivo"}')
    spans = gr.Textbox(label="Offsets literales JSON", value="[]", lines=3,
                       info='Ejemplo: [{"aspect":"servicio","polarity":"positivo","start":12,"end":20}]')
    inspect = gr.Button("Verificar evidencias")
    preview = gr.Code(label="Evidencias con texto exacto", language="json")
    inspect.click(preview_evidence, inputs=[review_id, spans], outputs=preview)
    save = gr.Button("Guardar anotación humana", variant="primary")
    feedback = gr.Markdown()
    save.click(submit_annotation, inputs=[review_id, annotator, sentiment, aspects, spans],
               outputs=feedback)
    gr.Markdown("## Control de calidad — resumen sin transcripciones")
    quality_button = gr.Button("Calcular acuerdo y listar posibles duplicados")
    quality_output = gr.Code(label="Resultados privados de auditoría", language="json")
    quality_button.click(quality_preview, outputs=quality_output)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", share=False)
