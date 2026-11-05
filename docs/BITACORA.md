# Bitácora de investigación y desarrollo

## 2026-11-04 — prototipo inicial v0.1.0

- Contexto: opción 3 de Traveler Insights 2026, seguimiento del curso IA ISCO-A06A de UTB.
- Continuidad: Parte 1 definió problema, reseña como entrada, clasificación triple y recomendación. Parte 2 justificó y diseñó arquitectura para Cartagena como aplicación **futura**.
- Construido: paquete Python, clasificador TF-IDF + regresión logística, aspectos por reglas, fragmentos literales, abstención, recomendaciones, UI local Gradio, CLI y suite de pruebas.
- Datos de demostración: ejemplos creados manualmente con fines de integración; sin validez estadística ni licencia de dataset externo.
- Datos reales: **NO descargados / NO evaluados**.
- Métricas: **NO reportadas**. Las pruebas unitarias solo validan funcionamiento del código.
- Fuentes teóricas: reto oficial, Charfaoui & Mussard (2024), Yang & Hsu (2026), Fehle et al. (2026).
- Riesgos: mapeo ambiguo 0/1/2 del reto, falta de dataset local de Cartagena, sobreajuste, baja calibración, calidad variable de anotaciones, privacidad, carga de datos.
- Próximo paso: revisar licencia y CSV del reto; verificar etiquetas; crear partición real reproducible y documentar métricas.

## Plantilla para cada experimento

Fecha y commit | ID de experimento | SHA del dataset | tipo de partición | arquitectura | parámetros | semilla | hardware | tiempo | métricas | errores / sesgos | decisión

## 2026-11-04 — Incremento 04: lotes y revisión humana (PR de desarrollo)

- Implementados `batch.py` y una nueva pestaña Gradio para reseñas por líneas/CSV, conteos de etiquetas DEMO, corrección humana por ID y exportación JSON/CSV.
- La corrección conserva por separado predicción del modelo, etiqueta humana y nota; no modifica pesos ni crea etiquetas científicas.
- Entrada validada con límites explícitos (200 filas, 5000 caracteres, CSV UTF-8 <= 1 MiB).
- Pruebas de núcleo y callbacks UI; CI configurado para instalar UI en Python 3.11/3.12. Verificar estado del run de PR antes de fusionar.
- Ambiente local de desarrollo antes del PR: 27 passed, 1 xfailed sobre copia de checkpoint anterior; smoke test de callbacks funcionó. El remoto incluye además pruebas de emojis, por lo que el total definitivo procede de CI.
- Sin dataset real, sin métricas científicas; la prueba de negación incorrecta continúa como xfail.
- Próximo: interfaz común LR/SVM y arquitectura ASR opcional, con resultados claramente demostrativos.

## 2026-11-04 — Incremento 05: interfaz común LR/SVM (PR encadenado)

- Implementados LR y LinearSVC + calibración interna para un contrato `predict_proba` común.
- Selección por CLI y Gradio, rastreabilidad del modelo en salida y exportaciones.
- Pruebas nuevas del motor SVM, compatibilidad LR y smoke UI.
- La rama `feat/model-selection-svm` deriva de `feat/batch-human-review`, no de `main`. El PR debe ser revisado contra la rama padre antes de integrarse.
- Sin métricas científicas: DEMO solo prueba ejecución. Mantener `xfail` de negación; evaluar después con datos reales.


## 2026-11-04 — Incremento 07: audio/ASR local (PR)

- Código modular de validación WAV y ASR opcional por faster-whisper, con CPU int8 y sin descarga automática.
- Gradio: grabación o carga (normalización a WAV), transcripción editable, análisis con LR/SVM y trazabilidad de original/corrección.
- Tests ASR sin modelos mediante mocks, pruebas del flujo UI, errores y negaciones. CI Python 3.11/3.12 como puerta de calidad.
- Limitaciones: sólo formato WAV PCM tras conversión; silencio digital comprobado, ruido no completamente detectado; no análisis de emoción vocal ni datos científicos.
- Mantenimiento posterior: migrar acciones checkout@v4 y setup-python@v5 a versiones compatibles Node recientes, en PR independiente.

## 2026-11-05 — Corpus académico propio (rama independiente en revisión)

- Auditoría de main: dataset.py y scripts/audit_dataset.py no ofrecían procedencia/consentimiento ni anotación; experiments.py tenía holdout aleatorio sin separación por grupos. No se sobrescriben.
- Creado corpus.py: fuentes con permiso revisado, reseñas seudónimas, anotaciones separadas por revisor, adjudicación justificada y auditoría con hashes.
- Creado corpus_splits.py: componentes de grupos y duplicados normalizados, separación reproducible train/validation/test, manifiestos y rechazo de splits incompletos.
- Creado corpus_cli.py: inicio de plantillas, contribución voluntaria con confirmación, anotación, adjudicación, validación y split local; todo en data/private o data/processed.
- Tests con datos sintéticos cubren permisos, etiquetas, desacuerdos, fuga exacta y reproducción de split. Sin datos turísticos reales ni métricas.
- Auditoría adicional: revisión de casi duplicados (candidatos), spans humanos y acuerdo entre anotadores incorporados con pruebas; la revisión PII/derechos y la doble anotación obligatoria siguen pendientes.
- El PR #6 de ASR fue integrado en main (commit 85dbb73a); este PR conserva su código y documentación mediante merge de dos padres.
