# Incremento 04 — Lotes, revisión humana y exportación (2026-11-04)

## Objetivo
Incremento independiente del dataset oficial. Extiende la demostración local sin alterar el modelo de sentimiento, los controles existentes ni el flujo de la pestaña individual.

## Código ejecutable
- `src/traveler_agent/batch.py`: análisis por lotes con el agente actual; entrada por líneas o CSV UTF-8; un máximo de 200 reseñas y 5000 caracteres por reseña; CSV limitado a 1 MiB.
- `app.py`: nueva pestaña Gradio, carga de CSV, tabla de resultados, conteos descriptivos por predicción, selección de ID, corrección humana con nota y descargas JSON/CSV.
- La corrección guarda `etiqueta_corregida_humano` separada de `sentimiento`; **no** reentrena el estimador ni dispara acciones negativas automáticamente.
- Las tendencias cuentan etiquetas predichas por el modelo DEMO; no son métricas de exactitud ni resultados de Kaggle.
- Los exportables incluyen el texto introducido y se generan en archivos temporales locales. No compartir ni publicar reseñas privadas.

## Verificación
Pruebas nuevas: `tests/test_batch.py` y `tests/test_ui_smoke.py`. CI en Python 3.11/3.12 instala `.[ui,test]` y ejecuta `pytest -q`. Los cinco casos manuales de regresión originales siguen siendo cualitativos; la negación errónea continúa documentada con `xfail(strict=True)`.

## Ejecución Windows
```cmd
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[ui,test]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe app.py
```
Abrir la URL local indicada. Mantener `share=False`.

## Limitaciones
- No hay login, persistencia transaccional ni múltiples revisores; las correcciones existen en la sesión y en el archivo descargado.
- No hay audio/ASR ni entrenamiento real.
- No hay análisis de polaridad por aspecto; las recomendaciones siguen siendo exploratorias.
- No hay inferencia paralela optimizada ni mediciones de rendimiento.
- No se debe cargar `.joblib` de terceros; conservar únicamente artefactos generados localmente.

## Hoja de ruta incremental, sin esperar Kaggle
1. **Inc. 04 (este PR):** lotes, revisión humana, tendencias y exportaciones, con pruebas.
2. **Inc. 05:** interfaz común de modelos y TF-IDF + LinearSVC, evaluación con dataset sintético únicamente para comprobar ejecución, sin declarar métricas científicas.
3. **Inc. 06:** normalización no destructiva, spans posicionales, idioma/negaciones/emojis con tests; polaridad por aspecto marcada como `desconocida` hasta disponer de anotación.
4. **Inc. 07:** audio local opcional, grabación/carga, validación y transcripción ASR con transcripción editable; manejar descarga inicial/licencia de pesos y CPU Windows.
5. **Inc. 08:** benchmark secuencial/paralelo con igualdad de resultados; medición real antes de activar paralelismo.
6. **Inc. 09:** Transformer preentrenado local opcional y pruebas de integración sin confundir preentrenamiento con ajuste en el reto.
7. **Inc. 10:** dashboards consolidados y API ligera solo si aparece un consumidor real.

Cada incremento: rama, pruebas, PR, CI, documentación y revisión antes de integrar en `main`. La obtención y auditoría de Kaggle continúan como línea paralela y no bloquean la ingeniería.
