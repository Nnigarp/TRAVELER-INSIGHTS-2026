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
