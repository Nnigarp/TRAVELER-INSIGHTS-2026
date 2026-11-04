# Incremento 02 — Auditoría de v0.1 y mejora conservadora de aspectos

Fecha: 2026-11-04. Alcance: reglas de aspectos y recomendaciones; modelo TF-IDF + regresión logística **sin cambios**.

## Confirmado en código real
- `agent.py`: revisión si máximo de `predict_proba` < 0.56, o presencia de `pero|aunque|sin embargo` con máximo < 0.80. No es incertidumbre calibrada.
- `aspects.py`: vocabulario literal con límites de palabra. No asigna polaridad por aspecto.
- `policy.py`: una sola acción seleccionada por prioridad, aunque existan varios aspectos.
- `experiments.py`: único holdout 80/20 estratificado; no existe validación por lugar ni validación externa.
- `model.py`: TF-IDF de palabras y caracteres dentro de un Pipeline, con clase balanceada y seed 42.
- `app.py`: carga artefacto joblib; Gradio solo en localhost y `share=False`.

## Cambios
1. Se agregan vocablos generales a servicio/higiene y aspectos ruido/alimentación para mejorar cobertura exploratoria; **no son ABSA**.
2. La política para reseñas clasificadas negativas con varios aspectos propone verificar cada dimensión detectada, sin asignarles polaridad ni elegir arbitrariamente una sola.
3. Se preserva el gate de revisión, el JSON existente, la clasificación y la interfaz.
4. Se agregan casos cualitativos de regresión y un `xfail` estricto para la negación; **no** se optimiza el modelo sobre esas frases.
5. Se integra auditor de datasets independiente (Incremento 01), sin entrenamiento ni interpretación automática de etiquetas numéricas.

## Cautelas
- Agregar expresiones al vocabulario puede incrementar falsos positivos de aspectos; medir precisión/recall contra anotación humana.
- El vocabulario es heurístico, el gate no está calibrado y no hay evidencia de generalización ni ABSA.
- Los cinco casos manuales están destinados a regresión funcional, no al test científico independiente.
- La política devuelve un texto largo si se detectan muchos aspectos; es una mejora intermedia. La siguiente interfaz puede mostrar acciones estructuradas por aspecto.
- El dataset real y su licencia/procedencia deben verificarse antes del entrenamiento científico.

## Próximo incremento
Dataset oficial verificado → protocolo de splits sin fuga → baseline y registro experimental. No usar prueba final como tuning.
