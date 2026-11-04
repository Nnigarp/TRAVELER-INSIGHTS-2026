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
