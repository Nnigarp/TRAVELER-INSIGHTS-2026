# Validación de integración Qwen/Ollama — 17 de noviembre de 2026

Base inspeccionada: `main` en `6400d52055424dfb41bebb4b381d9894c7cba34e`,
con PR #29 fusionado. Desarrollo aislado en
`feat/qwen-ollama-tourism-report-20261117`, [PR #30](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/pull/30).
No se modifica `main` ni se fusiona automáticamente.

## Evidencia ejecutada en Linux

| Comprobación | Resultado observado | Alcance |
|---|---|---|
| Regresión completa | 539 aprobadas, 1 xfail, 65,31 s | Incluye 65 pruebas nuevas; entorno de investigación instalado |
| Gradio HTTP anterior | 22 comprobaciones, 0 violaciones | LR/SVM reales DEMO, sesiones, cargas, correcciones y descargas |
| Ollama | Servidor HTTP local simulado | JSON, errores, timeout de pared, desconexión, modelo inesperado, límites y no redirección |
| Contratos de clasificación/control | Conservados | Rechazo de cambios de etiqueta/aspectos/revisión y de hechos/propuestas no admitidos |
| BETO nuevo | Interfaz supervisada inyectada | Comprueba ausencia de fallback a LR por fallo generativo; no acredita pesos reales |
| Audio nuevo | Procedencia y transcripción editada simuladas | No se prueba micrófono físico ni Qwen real |
| Evaluación predictiva histórica | SHA-256 sin cambios | `57e3e4b78fb5208fd8ddd430b4fc2b769491197edc96f8745a1baa1ebe23f243` |

Comandos ejecutados desde la raíz con `PYTHONPATH=src`:

```text
python -m pytest -q --disable-warnings --junitxml=artifacts/qwen-full-regression.xml
python scripts/acceptance_gradio_http.py --registration-commit e20326484aa33d335e85bf0f8551c24adbbd13e1 --output artifacts/qwen-existing-http-final.json
```

El xfail sigue siendo la limitación histórica de negación del modelo clásico,
`test_negation_expected_positive_is_known_limitation`; no se elimina ni se oculta.
Las advertencias de dependencias de Gradio/Starlette/Pandas no son errores de las
pruebas. Los tiempos son del entorno de desarrollo y no una medición del portátil.

No se modifican corpus, particiones, semillas, umbrales, pesos ni resultados
científicos congelados. No se vuelve a entrenar BETO V3. La CI ya existente puede
reproducir sus controles supervisados históricos al cambiar `app.py`/lotes;
no se presentan esas reproducciones como experimentos nuevos ni se usan para
selección de modelos. La CI Qwen nueva no descarga modelos y se ejecuta en Linux
y Windows. Sus resultados concretos se consultan en los checks del PR.

## Aceptación real pendiente

El responsable ya confirmó LR/SVM, BETO V3 y Whisper Tiny en Windows y Qwen3 1.7B
por CLI independiente. **No se deduce de ello que esta integración funcione en su
portátil.** Este entorno no tiene un servidor Ollama real; no se inventan respuestas,
latencia de Qwen, memoria del portátil ni medidas de calidad generativa.

Ejecutar [la guía Windows](QWEN_OLLAMA_WINDOWS.md): callback primero, después
Gradio HTTP con BETO efectivo, informe Qwen validado, versión/digest y recursos.
Conservar JSON de ambos modos, exportación del lote y observación de audio. Un
fallo es evidencia de aceptación pendiente, no autorización para sustituir Qwen
por reglas y declarar éxito.

La composición actual usa citas completas y un catálogo de propuestas. Verifica
formato y procedencia, pero no garantiza que Qwen elija la cita más pertinente.
El caso solicitado es funcional: no mide calidad predictiva humana, calidad de
redacción libre ni capacidad de generalización. Ver el [guion académico](QWEN_DEMOSTRACION_2026-11-17.md).
