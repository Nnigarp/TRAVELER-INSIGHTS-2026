# Avance ejecutable: BETO supervisado y regresión de la DEMO

Preparación para la presentación del **lunes 9 de noviembre de 2026**, desde `main@188f9f7159c38810a4f228e7e467a3e1cfeb1a70`. El «90 %» es el avance académico solicitado; no se calcula ni acredita ese porcentaje sin una rúbrica y su evidencia.

## Qué está ejecutado

- Clasificación BETO real de tres sentimientos: encoder congelado + cabeza entrenada y ajuste parcial de la última capa + cabeza. Son dos artefactos versionados, no un encoder presentado como clasificador.
- Comparación justa con Dummy, TF-IDF+LR y TF-IDF+SVM con las mismas filas, selección solo en desarrollo nuevo y evaluación nueva.
- Selección opcional en Midnight Indigo mediante configuración local. LR sigue siendo el valor inicial. Ante un fallo de BETO, la operación completa se ejecuta con LR y lo registra explícitamente; no se atribuye a BETO la predicción de respaldo.
- Los cinco agentes participan realmente. El servidor HTTP/cola de Gradio ejecutó LR/SVM/BETO, sirvió frontend/configuración con HTTP 200 y devolvió 10 eventos de traza por análisis.
- Texto, lotes, CSV, corrección humana simulada, exportaciones y callbacks de audio conservan sus contratos. El test ASR usa PCM artificial y un backend simulado; **no acredita voz real**.

No cambian los agentes, el entrenamiento clásico por defecto, los umbrales históricos, el aspecto Midnight Indigo ni los experimentos V2/V3. El análisis por aspectos sigue siendo heurístico y los fragmentos literales no explican causalmente la red neuronal.

## Experimento y límites científicos

Hipótesis nueva: comparar **representaciones** con el mismo presupuesto de TRAIN, no incrementar volumen. Se reutiliza únicamente `broad_120` de TRAIN V3 (20 familias, variantes 0/1); no se elige el brazo de mayor puntuación histórica. Se añaden solo 54 textos originales sintéticos: 18 de desarrollo/6 familias, 30 de evaluación/10 familias y 6 ambiguos sin etiqueta de sentimiento. Las etiquetas representan intención de redacción del asistente, **no gold humano**.

Familias/componentes disjuntos, control de duplicados exactos y Jaccard en las seis parejas de roles. Los textos históricos evaluados entran exclusivamente en una lista de hashes para excluir copias exactas. La independencia semántica y la ausencia de sesgos del autor no quedan demostradas por esos controles. TRAIN conserva las limitaciones y atajos posibles de su catálogo original. Ningún ID, fenómeno o metadato entra como característica del clasificador.

El catálogo y el [protocolo](../src/traveler_agent/research_data/contextual_supervised_protocol_v1.json) se fijaron antes de las predicciones. Semilla 42; CPU, 2 hilos; truncamiento a 128 tokens. Pooling: media con máscara, incluidos tokens especiales, normalización L2. Cabeza lineal: 2307 parámetros, AdamW, 60 epochs; selección entre epochs 10/30/60, elegida **30**. Ajuste parcial: 7 087 872 parámetros del bloque 11 + cabeza, 2 epochs, elegida **1**. Dropout del encoder desactivado deliberadamente, con gradientes habilitados en ese bloque. Las pérdidas por epoch y los tiempos se guardan en [selección](research/contextual-supervised-v1-selection.json).

LR/SVM eligen C entre 0.1/1/10, ambos seleccionan 10. SVM calibra por sigmoid en tres folds **solo TRAIN y por componentes**, con TF-IDF dentro de cada fold. No se usa la evaluación para escoger epochs, C, umbrales o variante BETO. Se conservan los resultados de ambos métodos contextuales.

## Resultados sintéticos observados

30 textos, 10 por clase. Matrices: filas = intención, columnas = predicción; orden **negativo, neutral, positivo**.

| Modelo | Macro-F1 | Recall negativo | Negativo→positivo | Matriz de confusión |
| --- | ---: | ---: | ---: | --- |
| Dummy | 0.1667 | 1.00 | 0/10 | `[[10,0,0],[10,0,0],[10,0,0]]` |
| TF-IDF + LR | 0.7684 | 0.80 | 1/10 | `[[8,1,1],[2,8,0],[2,1,7]]` |
| TF-IDF + SVM | 0.7079 | 0.80 | 1/10 | `[[8,1,1],[3,7,0],[4,0,6]]` |
| BETO congelado + cabeza | 0.7701 | 0.70 | 3/10 | `[[7,0,3],[1,7,2],[1,0,9]]` |
| BETO última capa + cabeza | 0.7290 | 0.50 | 4/10 | `[[5,1,4],[0,8,2],[1,0,9]]` |

La diferencia BETO congelado−LR es aproximadamente 0.0018. El bootstrap pareado descriptivo de diez familias da intervalo −0.2564 a +0.2445: **no demuestra superioridad**. BETO empeora el error negativo→positivo respecto de LR. El ajuste parcial obtiene mejor desarrollo pero peor evaluación; no se promociona automáticamente por ese desarrollo.

El [informe completo](research/contextual-supervised-v1-report.json) conserva métricas por clase, rebanadas lingüísticas, matrices, errores con ID/texto y costos. Las [150 predicciones guardadas](research/contextual-supervised-v1-predictions.json) corresponden a cinco modelos × treinta textos. Tres ejecuciones de reproducción con la misma semilla conservaron las clases y probabilidades (diferencia máxima 0). Son reproducciones de software, no tres muestras independientes. El primer informe se conserva [sin reescribir](research/contextual-supervised-v1-initial-report.json); se endurecieron controles y empaquetado, sin cambiar datos o hiperparámetros por los resultados.

Esta evaluación ya está observada: no se anunciará como nuevo test final intacto en trabajos posteriores. Nuevas elecciones requerirán desarrollo/evaluación nuevos apropiados.

## Abstención y calibración

Solo se describe la política histórica sin reajustarla. Aceptó 8/30 casos de BETO congelado, con 0 errores entre esos ocho; derivó 5/6 ambiguos. El ajuste parcial aceptó 14/30, con dos errores **negativo→positivo automáticos**. No hay política contextual admitida ni garantía de automatización segura.

La referencia matemática IID para 0/8 errores aún permite un límite superior unilateral del 95 % de 31.2 %. Las dependencias de familias y cualquier elección de política impiden usarla como certificación. «Revisar todo» tampoco se ofrece como solución operativa. NLL, Brier, ECE y bootstrap son [análisis descriptivo posterior](research/contextual-supervised-v1-analysis.json) sobre predicciones guardadas, no selección de calibrador/umbral. Datos humanos y desarrollo suficiente siguen pendientes.

## Recursos, checkpoint y aislamiento

Última ejecución completa local: 34.55 s de pared, 75.51 s CPU y pico 1727.33 MiB del proceso Linux (incluye imports, modelos y optimizador), 2 hilos, sin GPU. Las primeras repeticiones tardaron aproximadamente 32–38 s. La latencia p50/p95 se conserva por modelo: seis solicitudes calientes con tokenización/vectorización incluidas; muestra descriptiva pequeña, no SLA. Primera carga HTTP de BETO: aproximadamente 4.4 s, incluyendo recarga fría; no confundirla con latencia caliente.

No se midieron costo monetario, GPU ni consumo Windows. Para el ensayo Windows se propone Python 3.12 x64, varios GB de disco y margen de RAM sobre el pico Linux observado; no son requisitos mínimos Windows validados. No se descarga ni importa torch con la ruta clásica de la app.

Checkpoint `dccuchile/bert-base-spanish-wwm-cased`, revisión **`c4d86612f51b4f46759c8390d1798c2febe71b93`**, peso base SHA-256 **`e131a95091c777bbd45250fb647fec415010cb8cd1ad6e1d59babeb82a0be360`**. Se verifican hashes; `trust_remote_code=False`, carga local `weights_only=True`. Cabeza NumPy sin pickle; adaptación safetensors restringida al bloque 11. La recarga portátil reprodujo las probabilidades exactamente.

El [aviso de licencia original](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/blob/c4d86612f51b4f46759c8390d1798c2febe71b93/README.md) expresa intención CC BY 4.0 y advierte incertidumbre sobre compatibilidad de sus corpus de preentrenamiento. No se presenta como autorización comercial universal. Los pesos y cachés no se publican en Git; CI conserva solo informes sintéticos. Los directorios pueden contener temporales del runtime: el exportador copia únicamente los archivos verificados y el tamaño de directorio observado no equivale al peso lógico del checkpoint (439 621 341 bytes).

## Windows: ejecutar y preparar la presentación

Desde la raíz del clon, en la rama `feat/beto-supervised-demo-20261107`:

```powershell
py -3.12 -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-research\Scripts\python.exe -m pip install -r research/requirements-contextual-v1.txt -e '.[ui,test]'
.\.venv-research\Scripts\python.exe -m pytest -q
.\.venv-research\Scripts\python.exe -m traveler_agent.contextual_benchmark --download --output artifacts/contextual-presentacion-v1
.\.venv-research\Scripts\python.exe scripts/verify_contextual_reproduction.py --actual artifacts/contextual-presentacion-v1/assessment_predictions.json --reference docs/research/contextual-supervised-v1-predictions.json
.\.venv-research\Scripts\python.exe -m traveler_agent.contextual_analysis --run artifacts/contextual-presentacion-v1 --out artifacts/contextual-presentacion-v1/analysis.json
$env:TRAVELER_BETO_ARTIFACT = 'artifacts/contextual-presentacion-v1/models/beto_frozen'
.\.venv-research\Scripts\python.exe -m traveler_agent.demo_validation --ui --beto-artifact $env:TRAVELER_BETO_ARTIFACT --out artifacts/presentacion-funcional-v1.json
.\.venv-research\Scripts\python.exe scripts/smoke_gradio_http.py --beto-artifact $env:TRAVELER_BETO_ARTIFACT --out artifacts/presentacion-http-v1.json
.\.venv-research\Scripts\python.exe app.py
```

La descarga requiere conexión solo para preparar dependencias/checkpoint. La app carga el artefacto local y nunca descarga BETO durante el análisis. Elegir BETO manualmente en el selector; comparar con LR/SVM sin llamarlo benchmark justo: los clásicos de la UI usan DEMO y el contextual usa TRAIN experimental. La comparación justa es el runner anterior. Para repetir usar un destino nuevo; no sobrescribir resultados. Para volver a la configuración clásica, quitar `TRAVELER_BETO_ARTIFACT` y reiniciar.

Prueba real de voz, **pendiente del responsable en su equipo**: instalar `.[asr]`, preparar Whisper con la autorización explícita existente (`TRAVELER_ASR_ALLOW_DOWNLOAD=1` solo para la primera descarga o usar un modelo local), grabar voz propia, revisar/editar la transcripción y analizarla. Comprobar original, edición, modelo efectivo y traza; guardar evidencia privadamente. Con WAV PCM autorizado local, el CLI admite:

```powershell
.\.venv-research\Scripts\python.exe -m traveler_agent.demo_validation --ui --audio data/private/presentacion/voz-propia.wav --out data/private/presentacion/voz-verificada.json
```

Verificar también micrófono/permisos, descarga JSON/CSV y visualización en el navegador Windows antes de la exposición. El HTTP smoke no acredita esos pasos. [Evidencia de callbacks](research/functional-validation-2026-11-07.json) y [servidor HTTP](research/gradio-http-validation-2026-11-07.json).

## Datos humanos, PR #23 y extensiones

PR #23 sigue separado y sin fusionar. Su revisión final corrigió la integridad de spans y mantiene 285 pruebas + un xfail; el formulario ya probado por el responsable en Windows y su JSON de ensayo **no son una reseña admitida**. El ensayo de dos anotadores/adjudicación usa fixtures artificiales aislados. No se accede a archivos de consentimiento adjuntos ni se los registra automáticamente. No hay benchmark humano ni BETO entrenado con datos humanos.

Kaggle sigue pendiente de acceso personal autorizado, reglas, procedencia/derechos y diccionario auténtico de etiquetas de `train.csv`. No se accede al test oculto ni se infiere el mapeo numérico. UNED no entra en este entrenamiento.

BM25/Sentence-BERT/FAISS y RAG se evaluaron como extensiones de **recuperación**, no clasificación. El repositorio no tiene un corpus documental turístico admitido con permisos y relevancias; los textos sintéticos de evaluación no son hechos turísticos. Por eso no se despliega búsqueda/recomendación factual ni un RAG de apariencia funcional. [Sentence Transformers](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) aporta similitud semántica; [FAISS](https://faiss.ai/) indexa vectores, pero hoy no hay evidencia de volumen/ganancia que justifique añadirlo. El paso habilitante es admitir documentos/versiones/licencias y consultas con relevancia juzgada, después comparar BM25 y encoder multilingüe con Recall@k/MRR/nDCG. No hay LLM externo, secretos o costos externos activados.

La CI contextual instala CPU explícitamente, entrena las dos variantes, verifica las 150 predicciones fijas, ejecuta optimizer/callbacks/HTTP y conserva informes. La CI general mantiene el xfail estricto de negación de la DEMO; no se lo elimina porque BETO resuelva un ejemplo manual. La evidencia final de CI debe consultarse en el PR, no inferirse del documento.
