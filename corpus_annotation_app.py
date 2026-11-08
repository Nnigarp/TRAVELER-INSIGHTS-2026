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
        raw_spans = json.loads(spans_json)
        spans = validate_span_entries(record["text"], raw_spans)
        if any(aspects.get(s["aspect"]) != s["polarity"] for s in spans):
            raise CorpusError("Evidencia y polaridad anotada no coinciden.")
        path = ROOT / "annotations.csv"
        record_annotation(path, {r["review_id"] for r in records},
                          review_id=review_id, annotator_id=annotator_id,
                          sentiment=sentiment, aspects=aspects)
        if spans:
            record_evidence(ROOT / "evidence_spans.csv", review_id, annotator_id,
                            record["text"], raw_spans, aspects)
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


def build_legacy_app():
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
    return app

def packet_callbacks(packet_path, responses_path):
    """Only assigned text and this person's drafts; never reads the master CSVs."""
    from traveler_agent.human_intake import (load_packet, _answers, save_answer, seal_answers)
    packet = load_packet(packet_path)

    def load(item_id):
        _, response = _answers(packet_path, responses_path)
        item = next((r for r in packet["items"] if r["item_id"] == item_id), None)
        own = response["answers"].get(item_id, {})
        names={True:"presente",False:"ausente",None:"incierto"}
        return (item["text"] if item else "", own.get("sentiment"), own.get("mixed", False),
                own.get("ambiguous", False), names[own.get("negation")], names[own.get("contrast")],
                json.dumps(own.get("aspects", {}), ensure_ascii=False),
                json.dumps(own.get("spans", []), ensure_ascii=False))

    def save(item_id, sentiment, mixed, ambiguous, negation, contrast, aspects, spans):
        try:
            flags={"presente":True,"ausente":False,"incierto":None}
            negation=flags[negation] if isinstance(negation,str) else negation
            contrast=flags[contrast] if isinstance(contrast,str) else contrast
            result = save_answer(packet_path, responses_path, item_id,
                {"sentiment":sentiment,"mixed":mixed,"ambiguous":ambiguous,"negation":negation,"contrast":contrast,
                 "aspects":parse_aspects(aspects),"spans":json.loads(spans)})
            return f"Borrador propio guardado: {result['answered']}/{result['assigned']}. Editable hasta finalizar."
        except (CorpusError, ValueError, TypeError, OSError, KeyError):
            return "No guardado: revisar etiqueta, marcas, aspectos/spans y estado del paquete. No se muestran otros dictámenes."

    def seal(independent):
        try:
            seal_answers(packet_path,responses_path,independent=independent)
            return "Respuestas finalizadas. Entregar solo este archivo al coordinador por el canal privado acordado."
        except (CorpusError, ValueError, TypeError, OSError):
            return "No finalizado: completar el lote y declarar anotación humana independiente."

    return packet, load, save, seal


def build_packet_app(packet_path, responses_path):
    packet, load, save, seal = packet_callbacks(packet_path, responses_path)
    with gr.Blocks(title="Traveler Insights | Lote humano privado") as application:
        gr.Markdown("# Anotación humana independiente\nSolo tu lote y tus borradores. No usar predicciones ni consultar etiquetas de otra persona.")
        gr.Markdown(packet["guide"])
        item = gr.Dropdown(label="Reseña asignada", choices=[(f"Reseña {i+1}",r["item_id"]) for i,r in enumerate(packet["items"])], value=None)
        text = gr.Textbox(label="Texto original", lines=6, interactive=False)
        sentiment = gr.Dropdown(label="Sentimiento global", choices=["negativo","neutral","positivo","desconocido"],value=None)
        mixed = gr.Checkbox(label="Hay opiniones mixtas",value=False)
        ambiguous = gr.Checkbox(label="No puedo resolver sentimiento global: ambiguo",value=False)
        negation = gr.Dropdown(label="Negación con alcance relevante",choices=["presente","ausente","incierto"],value="incierto")
        contrast = gr.Dropdown(label="Contraste explícito o implícito",choices=["presente","ausente","incierto"],value="incierto")
        aspects = gr.Textbox(label="Aspectos opcionales, JSON",value="{}")
        spans = gr.Textbox(label="Evidencias opcionales, offsets JSON",value="[]")
        item.change(load,inputs=item,outputs=[text,sentiment,mixed,ambiguous,negation,contrast,aspects,spans])
        feedback = gr.Markdown()
        gr.Button("Guardar mi borrador",variant="primary").click(save,
            inputs=[item,sentiment,mixed,ambiguous,negation,contrast,aspects,spans],outputs=feedback,concurrency_limit=1)
        independent = gr.Checkbox(label="Declaro anotación humana propia, sin modelos ni etiquetas ajenas",value=False)
        gr.Button("Finalizar lote").click(seal,inputs=independent,outputs=feedback,concurrency_limit=1)
    return application


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Anotación local, sin compartir servidor")
    parser.add_argument("--packet",type=Path)
    parser.add_argument("--responses",type=Path)
    args = parser.parse_args()
    if bool(args.packet) != bool(args.responses): parser.error("Usar --packet y --responses juntos.")
    app = build_packet_app(args.packet,args.responses) if args.packet else build_legacy_app()
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False)
