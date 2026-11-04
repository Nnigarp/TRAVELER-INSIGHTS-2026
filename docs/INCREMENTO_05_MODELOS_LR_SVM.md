# Incremento 05 — Selección reproducible de clasificadores LR/SVM

**Fecha:** 2026-11-04  
**Alcance:** funcionalidad ejecutable con ejemplos inventados; **sin métricas científicas del reto**.

## Código incorporado

1. `src/traveler_agent/model.py` ofrece `build_pipeline(model_kind="lr"|"svm")` y `train(..., model_kind=...)` con una misma interfaz `predict`, `predict_proba` y `classes_` usada por el agente existente.
2. Ambos modelos comparten los mismos dos canales TF-IDF de palabras (unigramas y bigramas) y caracteres (3–5); únicamente cambia el clasificador.
3. LR sigue siendo el valor predeterminado para no cambiar el comportamiento de las pruebas existentes.
4. SVM utiliza `LinearSVC(class_weight="balanced", random_state=42)` con `CalibratedClassifierCV(method="sigmoid", cv=3)` para ofrecer `predict_proba` a la interfaz previa. **Todo TF-IDF + LinearSVC está encapsulado dentro de `CalibratedClassifierCV`**, evitando que los folds de calibración compartan un vocabulario entrenado fuera de ellos. Es imprescindible disponer de por lo menos **tres muestras por clase**. La calibración interna no demuestra fiabilidad fuera de los ejemplos de entrenamiento; el umbral de revisión humana sigue siendo demostrativo.
5. `cli.py`: `--model-kind lr|svm` para `demo-train`, `train`, `evaluate`. El nombre `artifacts/demo_model.joblib` se conserva para el LR predeterminado.
6. `experiments.py`: evaluación de holdout reutilizando el mismo protocolo y parámetro de modelo. No ejecutar sobre DEMO para reportar métricas.
7. `app.py`: selector de modelo visible en las dos pestañas; SVM entrenado en memoria con DEMO al usarlo por primera vez. Cada JSON y CSV de lote registra `modelo_utilizado` sin modificar la salida original del clasificador.
8. `tests/test_model_selection.py` y `tests/test_ui_smoke.py`: verificaciones de interfaz, regresión LR, validación de tamaño mínimo, exportación y serialización local.

## Ejecución en Windows

Desde la rama de desarrollo, con `pip install -e ".[ui,test]"`:

```cmd
python -m pytest -q
python app.py
python -m traveler_agent.cli demo-train --model-kind svm
python -m traveler_agent.cli predict --model artifacts/demo_svm.joblib --text "El hotel estaba limpio, pero había mucho ruido."
```

**Seguridad:** solo cargar `joblib` creado por el usuario en su entorno de confianza. Los artefactos y reseñas no deben subirse al repositorio público.

## Lo que NO demuestra

- No hay datos oficiales Kaggle ni corpus externos verificados.
- No hay mejora de Macro-F1 ni precisión demostrada; un test con DEMO solo comprueba que el software funciona.
- SVM no es un Transformer, ABSA ni un sistema robusto ante ironía o negaciones.
- Falta un protocolo formal de comparación con splits idénticos, selección de hiperparámetros y calibración independiente.
- Se conserva el `xfail(strict=True)` documentado para la negación errónea de la regresión logística DEMO.

## Próximos incrementos

Análisis lingüístico contextual con evidencia posicional y perfiles multilingües; entrada de audio y ASR local opcional; comparación secuencial/paralela; Transformer multilingüe opcional, con consentimiento de descargas y respeto de licencias.
