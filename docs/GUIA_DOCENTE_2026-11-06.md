# Traveler Insights 2026 — Guion breve para la docente (viernes 6 de noviembre)

**Alcance de la exposición:** 6–8 minutos; prototipo académico estable + evidencia de dos pruebas sintéticas, no producto comercial ni clasificador validado en reseñas reales.

## 1. Problema y objetivo (45 segundos)

Las opiniones turísticas pueden combinar elogios y quejas sobre servicio, higiene, precio, espera, seguridad o accesibilidad. Revisarlas manualmente exige tiempo y los modelos de sentimiento pueden fallar ante contrastes y lenguaje cotidiano.

**Traveler Insights 2026** es un **agente académico de apoyo a decisiones**, no un evaluador autónomo infalible: recibe reseñas, muestra una clasificación global DEMO, identifica aspectos exploratorios y fragmentos literales, determina si conviene revisión humana y propone una recomendación justificable. Los fragmentos son evidencias descriptivas, no explicaciones causales.

## 2. Arquitectura y funcionamiento (90 segundos)

Flujo de texto: **reseña → TF-IDF de palabras/caracteres → clasificador LR o SVM → etiqueta positivo/neutral/negativo → extracción de aspectos por reglas → evidencias textuales → política de revisión humana → recomendación + JSON**.

Flujo alterno de audio: **micrófono o WAV válido → ASR local faster-whisper tiny → transcripción editable → mismo análisis**. Se registra procedencia de transcripción original y corregida. El audio mide voz a texto, **no emociones por la voz**.

Flujo de lotes: reseñas individuales/CSV → predicciones y conteos → corrección manual en sesión que **no sobrescribe la etiqueta del modelo** → exportación JSON/CSV.

La interfaz Gradio Midnight Indigo sirve como presentación; no garantiza validez de las predicciones. La clasificación de la interfaz utiliza ejemplos ficticios de demostración. Las puntuaciones no están calibradas externamente.

## 3. Funcionalidad comprobada (45 segundos)

En Windows con Python 3.12 se comprobó clasificación LR y SVM, grabación y transcripción de 7,32 s, carga correcta de WAV PCM16 16 kHz mono, corrección textual, procesamiento de lote, exportaciones y trazabilidad. **Limitación abierta:** un OGG/Opus de WhatsApp produjo error no diagnosticado; la conversión a WAV funcionó. Cobertura de aspectos por reglas incompleta.

El último corte de regresión en `main` del PR #12 correspondió al SHA `dfac831aa6050a7242234e95122d97e43fa7a877`; los tests automatizados ejecutados en el PR #13 reportaron 162 passed y 1 xfailed en Python 3.12. Los experimentos son funcionalidades de investigación, independientes del modelo de la interfaz.

## 4. Datos y metodología (90 segundos)

**Estudio 1 (PR #12, congelado):** 336 reseñas 100 % sintéticas, generador local sin LLM ni scraping, semilla 42; 56 familias disjuntas entre entrenamiento, validación y test. Etiquetas: intención del generador, **NO gold humano**. Dummy, TF-IDF+LR, TF-IDF+SVM; tuning solo con validación y test reservado.

**Estudio 2 (PR #13, pendiente de merge al preparar este guion):** 1.260 textos inventados de 180 núcleos de situaciones redactados aparte del primer generador, semilla 913. Se cargan los documentos del experimento original mediante un run ID y hash fijo; los modelos se **reconstruyen** desde TRAIN original y parámetros congelados porque no se archivaron los joblib originales. No se entrenó ni ajustó usando el desafío. Cero duplicados exactos detectados entre conjuntos; una auditoría de similitud exploratoria no demuestra independencia semántica absoluta.

## 5. Resultados comparables con cautela (90 segundos)

| Modelo | Macro-F1 test sintético inicial | Macro-F1 desafío sintético | Accuracy desafío |
| --- | ---: | ---: | ---: |
| Dummy | 0,1667 | 0,1667 | 33,33 % |
| LR | 1,0000 | 0,3926 | 40,56 % |
| SVM | 1,0000 | 0,3882 | 40,48 % |

Ambos conjuntos son **artificiales y de distribuciones distintas**. No se trata de medidas de precisión sobre clientes auténticos de hoteles, museos o restaurantes.

El error más notable: de 420 ejemplos de intención negativa, **232 (55,24 %) fueron etiquetados como positivos por LR** y **236 (56,19 %) por SVM**. El recall de negativos fue de 24,05 % (LR) y 21,43 % (SVM).

**Explicación prudente:** el primer generador repetía frases descriptivas que diferenciaban fácilmente las clases, incluso entre familias; LR/SVM probablemente aprendieron esas pistas. Con otro vocabulario el Macro-F1 cae. La diferencia no demuestra por sí sola causalidad y algunas intenciones de etiqueta del nuevo corpus podrían ser ambiguas.

## 6. Limitaciones y próximos pasos (60 segundos)

- No se ha demostrado generalización a datos humanos auténticos, a Cartagena o a español colombiano/peruano natural.
- Los 1.260 textos del desafío son siete variaciones de 180 núcleos y las etiquetas no están adjudicadas por expertos.
- Las marcas automáticas de fenómeno (`negation`, `contrast`, `misspelling`) son aproximadas: no garantizan negación de sentimiento, contradicción real ni error ortográfico presente.
- El sistema de aspectos es exploratorio, no ABSA contextual supervisado; ASR OGG/Opus no soportado de forma comprobada.
- **Después de recibir retroalimentación docente:** identificar corpus auténtico con términos de licencia/acceso públicos y verificables; registrar procedencia y privacidad; obtener etiquetas humanas cuando corresponda; congelar test auténtico independiente y comparar entrenamiento sintético, auténtico e híbrido sobre el mismo test, con Macro-F1, F1 por clase, matrices y análisis de errores.
- No modificar modelos ni reabrir los test sintéticos para maximizar puntuaciones históricas.

## Preguntas que podría plantear la docente

**¿Por qué 100 % en el primer test?** Porque las expresiones sintéticas compartidas entre familias crearon atajos de vocabulario; no demuestra comprensión general.

**¿Entonces el modelo sirve?** Como prototipo de clasificación y agente de apoyo con revisión humana, sí; **no** como predictor científicamente validado para turistas reales.

**¿El corpus fue anotado por personas reales?** No: los textos son inventados y las etiquetas son intenciones del generador. La validación humana genuina es un objetivo pendiente.

**¿Por qué se evaluó una vez?** Para evitar seleccionar hiperparámetros, reformular textos o ajustar reglas a partir del test reservado y producir un resultado engañoso.

**¿Qué aporta el proyecto?** Una arquitectura funcional con trazabilidad y, sobre todo, una evaluación reproducible que identifica una limitación real del método de entrenamiento sintético.

### Material verificable

- [PR #12](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/pull/12) — corpus original.
- [PR #13](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/pull/13) — desafío, documentación y métricas.
- [CI del desafío](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/42010214928).
- [Auditoría puntual de errores y etiquetas](REVISION_ERRORES_Y_ETIQUETAS_PR13_2026-11-05.md).

**Frase de cierre sugerida:** “Hemos demostrado el funcionamiento del agente y detectado que un resultado perfecto en datos inventados no equivale a generalización. Ahora necesitamos evaluarlo en opiniones humanas independientes y autorizadas”.
