# Revisión interna con la docente — 6 de noviembre de 2026

**Proyecto:** Traveler Insights 2026, Inteligencia Artificial (UTB).  
**Objetivo:** reunión interna de avance, no sustentación final.

## Guion de un minuto

“Traveler Insights es un agente de apoyo al análisis de experiencias turísticas. El repositorio estable ya permite clasificar sentimiento de demostración, identificar aspectos mediante reglas, activar revisión humana, generar recomendaciones y exportaciones, analizar lotes y elegir regresión logística o SVM. La interfaz Gradio tiene tests automatizados. Además desarrollamos ASR local opcional con faster-whisper: la grabación real, transcripción editable y análisis LR/SVM fueron comprobados en Windows; falta la prueba de subida de archivo para fusionar ese PR. La línea de investigación siguiente es un corpus turístico propio con autorizaciones, anotación humana y adjudicación, seguido de experimentos reproducibles Dummy/LR/SVM con train, validation y test independiente. **Todavía no tenemos un dataset turístico real autorizado, ni métricas de generalización**. Buscamos orientación sobre anotación, permisos y protocolo experimental.”

## Matriz de estado de revisión

| Frente | Comprobado | Pendiente |
| --- | --- | --- |
| Principal | Código estable de texto, LR/SVM, Gradio, lotes y NLP heurístico; CI | Validación turística externa |
| Audio (PR #6) | Grabación real, ASR, edición y LR/SVM en Windows; mocks CI | Prueba de carga de archivo y fusión |
| Corpus (PR #7) | Contratos, CLI, anotaciones, adjudicación, splits y tests | Datos reales, revisión local, PII, casi duplicados |
| Benchmark (PR #8, dependiente de #7) | Código para Dummy/LR/SVM, tune validation, test reservado, manifiestos | Validación con corpus real autorizado; CI y revisión |
| ABSA | Heurísticas de aspectos y evidencia literal | Polaridad por aspecto y anotación de spans |
| TLN UniTo | Experiencia de análisis lingüístico y sistemas de recuperación | Embeddings, FAISS/BM25, RAG, DM, BERTopic |

## Contribución original demostrable

- Agente dirigido a decisiones y revisión humana.
- Pipeline voz–transcripción editable–NLP–clasificador con trazabilidad y ejecución local.
- Ingeniería de corpus propio con procedencia, consentimiento y etiquetas adjudicadas.
- Diseño de investigación reproducible para comparación de modelos y control de fugas.

## Decisiones para consultar a la docente

1. Alcance del corpus: Cartagena/Colombia u otros destinos, unidad de muestreo.
2. Fuente de reseñas permitida, términos de recolección, consentimiento, retención y derecho de retiro.
3. Guía de etiquetas positivas, neutrales y negativas y criterios de desconocido.
4. Doble anotación ciega, adjudicación y acuerdo entre personas.
5. Agrupación por establecimiento, usuario seudónimo, fecha o fuente para la hipótesis de generalización.
6. Macro-F1 como métrica principal, resultados por clase/idioma y test reservado; extensión a corpus externo.
7. ABSA supervisado y fine-tuning multilingüe solo con corpus y anotaciones suficientes.

## Límites explícitos

- Las métricas de tests sintéticos son verificación de software, no rendimiento turístico.
- El clasificador global no determina correctamente polaridad por aspecto en una opinión mixta.
- El corpus no detecta automáticamente casi duplicados ni realiza anonimización jurídica.
- No están integrados spaCy, XLM-R ajustado, BM25/FAISS, BERTopic, RAG ni gestor de diálogo.
- Los artefactos joblib sólo se cargan si son locales y confiables.

Repositorio: https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026
