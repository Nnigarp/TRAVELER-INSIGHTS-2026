"""Demo local Gradio. No disponible para uso productivo ni publicar sin auditoría."""
import json
from pathlib import Path
import gradio as gr
from traveler_agent.model import load_model, train, save_model
from traveler_agent.demo_data import DEMO
from traveler_agent.agent import inspect_review
MODEL_PATH=Path('artifacts/demo_model.joblib')
if not MODEL_PATH.exists():
    model=train([t for t,_ in DEMO],[y for _,y in DEMO]);save_model(model,MODEL_PATH,provenance='SINTETICO_DEMO_NO_PUBLICABLE')
model=load_model(MODEL_PATH)

def analyze(review):
    try:
        r=inspect_review(review,model)
        summary=f"**Sentimiento:** {r['sentimiento']}  |  **Revisión humana:** {'Sí' if r['revision_humana'] else 'No'}\n\n**Sugerencia:** {r['recomendacion']}"
        return summary, json.dumps(r,ensure_ascii=False,indent=2)
    except Exception as exc:
        return f'Error: {exc}', '{}'
with gr.Blocks(title='Traveler Insights 2026 | Demo académico') as app:
    gr.Markdown('# Traveler Insights 2026 · Prototipo académico\n**AVISO: modelo entrenado con ejemplos ficticios para demostrar la integración. No son datos de Kaggle ni resultados científicos.**')
    inp=gr.Textbox(label='Reseña turística',lines=4,value='El guía fue amable, pero esperamos demasiado para entrar.')
    run=gr.Button('Analizar reseña',variant='primary')
    result=gr.Markdown(label='Resultado')
    detail=gr.Code(label='Trazabilidad JSON',language='json')
    run.click(analyze,inputs=inp,outputs=[result,detail])
if __name__=='__main__':app.launch(server_name='127.0.0.1',share=False)
