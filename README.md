# Traveler Insights 2026 — agente turístico (v0.1.0)

**Proyecto académico de Inteligencia Artificial (ISCO-A06A, UTB).** Proponente: Nilver García Pimentel. Caso de aplicación futura: Cartagena de Indias, Colombia.

> **Estado:** prototipo funcional de demostración, NO modelo científico validado. La opción 3 del reto pide clasificar reseñas **positivas, neutrales o negativas**. Esta implementación agrega aspectos por reglas, fragmentos literales y recomendaciones auditables. No afirma contar con datos locales de Cartagena.

## Qué funciona

1. **Percepción:** recibe reseñas e identifica validaciones de entrada (vacío, longitud).
2. **Clasificación supervisada:** `TfidfVectorizer` de palabras y caracteres + regresión logística (tres etiquetas semánticas).
3. **Aspectos exploratorios:** diccionario visible y extensible (servicio, precio, higiene, acceso, espera y seguridad). **No equivale a ABSA entrenado.**
4. **Evidencia descriptiva:** muestra fragmentos **literales** que contienen términos del diccionario; no ofrece una explicación causal del modelo.
5. **Decisión:** reglas deterministas y abstención/revisión humana cuando la puntuación es baja o hay indicios de opinión mixta.
6. **Trazabilidad:** salida JSON, manifiesto del modelo y hash del dataset (cuando exista CSV real).
7. **Evaluación separada:** holdout estratificado, Macro-F1, accuracy, métricas por clase y matriz de confusión; **únicamente para CSV real**.

## Instalación (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ui,test]"
pytest -q
traveler demo-train
traveler predict --text "El precio fue excesivo y esperamos horas."
python app.py
```

Abrir la URL local que muestre Gradio (generalmente `http://127.0.0.1:7860`). **No habilitar enlaces públicos ni introducir información personal real en esta demo.**

## Linux / Colab

```bash
python -m pip install -e '.[ui,test]'
pytest -q
traveler demo-train
traveler predict --text "El servicio fue excelente y la visita hermosa."
python app.py
```

## Modelo real: cuando se obtenga el dataset con permisos

El CSV debe contener columnas textuales `reseña` y `sentimiento`, este último con etiquetas **positivo, neutral, negativo** (sin números). En caso de que Kaggle proporcione 0/1/2, **se debe comprobar el mapeo contra archivos, descripciones y ejemplos oficiales antes de convertirlo**. La descripción pública del reto contiene inconsistencias. No asumir una correspondencia por intuición.

```bash
traveler evaluate --csv data/mi_dataset_verificado.csv --text-col reseña --label-col sentimiento --out artifacts/holdout.json
traveler train --csv data/mi_dataset_verificado.csv --text-col reseña --label-col sentimiento --out artifacts/real_model.joblib
traveler predict --model artifacts/real_model.joblib --text "La atención fue muy amable."
```

**Precaución metodológica:** primero evaluar una partición independiente; el comando `train` entrena posteriormente con todos los registros válidos para crear el artefacto final. Evite evaluar el mismo modelo en sus propios datos de entrenamiento. Para publicación científica, también exigir validación cruzada, partición por lugar/fecha si aplica, calibración y evaluación externa con muestras de Cartagena.

**Seguridad:** los archivos `.joblib` solamente deben abrirse si se crearon localmente y provienen de una fuente confiable; deserializar un archivo ajeno puede ejecutar código. El entrenamiento no descarga modelos remotos ni llama a servicios de IA.

## Arquitectura

```text
Reseña + ID --> Validación --> Vectorización TF-IDF --> Clasificación 3 clases
                                                |             |
                                Aspectos por reglas             v
                                      |                 Revisión de incertidumbre
                                      v                         |
                               Fragmentos literales --> Política transparente
                                                            |
                                           Etiqueta, evidencia, aspecto, sugerencia, revisión
```

## Qué **no** hace aún

- No utiliza todavía los archivos originales de Kaggle ni tiene etiquetas o métricas reales.
- La DEMO predeterminada conserva LR/SVM. [BETO supervisado experimental](docs/AVANCE_EJECUTABLE_BETO_Y_DEMO_2026-11-07.md) puede habilitarse expresamente; sus resultados son sintéticos. LLM, recuperación, RAG y ABSA supervisado siguen pendientes.
- [BETO Training V2](docs/BETO_TRAINING_V2_2026-11-08.md) compara entrenamiento por etapas/completo, datos sintéticos identificados y calibración separada; incluye matrices, costos y receta Windows. No demuestra aún superioridad estable ni valida reseñas humanas.
- [BETO V3: piloto de calidad y ampliación](docs/BETO_V3_PILOTO_Y_ESCALADO_2026-11-10.md) aporta un corpus original con revisión AI, particiones independientes y un benchmark común; distingue el piloto ejecutado del plan de 5.000 reseñas.
- El valor mostrado como confianza es **puntuación no calibrada**. No debe tratarse como probabilidad fiable.
- El extractor de aspectos y la selección de fragmentos son heurísticos. Las recomendaciones requieren supervisión humana.
- No se publica automáticamente a GitHub ni se despliega públicamente.

## Investigación / artículo

Véanse `docs/PROTOCOLO_INVESTIGACION.md`, `docs/BITACORA.md`, `docs/DECISIONES.md`, `docs/FUENTES.md` y `docs/PLAN_GITHUB.md`. **No se garantiza publicación**: dependerá de rigor de los experimentos, novedad, calidad de datos, evaluación externa y revisión por pares.

## Fuente del reto

[Traveler Insights 2026 (Kaggle)](https://www.kaggle.com/competitions/travelerinsights2026/overview). La competencia describe tres clases y datos turísticos vinculados a fuentes distintas; su referencia apunta a un dataset de Sucre. Cartagena es un **dominio objetivo futuro**, no se asume como origen de sus reseñas.

## Incremento 04 — análisis por lotes y revisión humana (DEMO)

La interfaz local conserva el análisis individual y añade una pestaña de lotes. Permite introducir hasta 200 reseñas, una por línea, o cargar un CSV UTF-8 de hasta 1 MiB con columna `reseña`. Muestra conteos de **predicciones DEMO**, permite registrar una etiqueta corregida y una nota por ID sin sobrescribir la predicción original, y exporta JSON/CSV. No existe aún un modelo validado con Kaggle ni ABSA supervisado. Los archivos exportados contienen texto de entrada; trátalos como privados.

Instala `.[ui,test]`, ejecuta `python -m pytest -q` y abre `python app.py`. Consulta [el alcance y las limitaciones del incremento](docs/INCREMENTO_04_LOTES_REVISION.md). El audio/ASR, SVM y Transformers quedan para incrementos posteriores, independientes de la adquisición del dataset.

## Incremento 05 — Elegir modelo: regresión logística o SVM

En Gradio, el selector de modelo utiliza **TF-IDF + LR** (predeterminado) o **TF-IDF + LinearSVC con calibración interna CV=3**. Ambos se entrenan para la demo con ejemplos ficticios. Esto es una prueba del software, **no evidencia de rendimiento en Kaggle ni fiabilidad probabilística externa**. También se registra `modelo_utilizado` en resultados individuales y exportaciones de lotes.

```cmd
python -m traveler_agent.cli demo-train --model-kind svm
python -m traveler_agent.cli predict --model artifacts/demo_svm.joblib --text "El servicio fue bueno."
python -m pytest -q
python app.py
```

Ver [alcance del incremento](docs/INCREMENTO_05_MODELOS_LR_SVM.md).


## Incremento 07 — micrófono y transcripción local opcional

Gradio incorpora una pestaña para grabar o subir audio. Convierte a WAV y verifica el formato PCM, longitud (0,25–120 s), tamaño (12 MiB) y silencio digital; ofrece transcripción ASR local con faster-whisper opcional, editable antes del análisis con LR/SVM. El JSON registra transcripción original, texto editado, hash del WAV, idioma solicitado y modelo ASR; no incluye el audio. **No se infieren emociones desde la voz.** No hay métricas científicas de ASR ni del reto.

Para mantener el análisis textual sin dependencias pesadas: instala solamente ".[ui,test]". Para activar ASR en Windows instala ".[ui,test,asr]"; el modelo permanece **offline por defecto** y su primera descarga requiere autorización explícita. Revisa [el manual del Incremento 07](docs/INCREMENTO_07_AUDIO_ASR.md) antes de usar grabaciones reales.

## Corpus académico propio (incremento en revisión)

El PR #7 añade una infraestructura independiente del dataset Kaggle y del audio: fuentes autorizadas, reseñas privadas, anotación humana por revisor, adjudicación final razonada y particiones reproducibles por grupo/duplicado. Los CSV y particiones se guardan únicamente bajo data/private/ y data/processed/ (excluidos de Git). El sistema no verifica permisos legalmente, no elimina PII de forma automática y detecta candidatos a casi duplicados para revisión humana; no realiza anonimización completa. No existe aún corpus real ni métricas experimentales.

Comenzar en la rama feat/tourism-corpus-foundation:

~~~powershell
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli init
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli validate
~~~

La validación exigirá primero registrar una fuente autorizada y al menos una reseña; ver docs/CORPUS_TURISTICO_2026.md. La CLI permite contribuir, anotar, adjudicar y dividir train/validation/test; sin descargar nada ni publicar textos.

Para **PILOTO-01 con aportaciones originales**, usar el [flujo privado de recogida y anotación ciega](docs/PILOTO01_RECOGIDA_HUMANA_Y_ANOTACION_CIEGA_2026-11-07.md). Reutiliza el gestor y añade formulario offline, consentimiento versionado, dos paquetes de anotación separados, tercera adjudicación y agrupación por autor/lugar/textos relacionados. No hay aún reseñas humanas admitidas ni resultados predictivos humanos.

~~~powershell
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli init
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli participant-form --out data/private/delivery/participant-001.html
~~~

## Benchmark reproducible del corpus adjudicado (PR encadenado)

En el PR #8 (rama feat/corpus-benchmark) se desarrolla infraestructura para comparar DummyClassifier, LR y SVM con el mismo train/validation/test del corpus académico. Comprueba permisos registrados, hashes y ausencia de grupos y textos exactos compartidos antes de entrenar. La selección de hiperparámetros sólo usa validation; SVM calibra dentro de folds separados por grupos de train. Produce informes, modelos y manifiestos privados. **No hay resultados científicos de reseñas reales**; tests usan datos sintéticos. Manual: docs/INCREMENTO_09_BENCHMARK.md.


## Aceptación académica y sustentación

La [guía breve](docs/GUION_SUSTENTACION_BREVE_2026-11-13.md), la [preparación Windows y micrófono](docs/SUSTENTACION_Y_ACEPTACION_2026-11-13.md) y el [análisis de respuestas congeladas](docs/ANALISIS_ACEPTACION_2026-11-13.md) separan contratos funcionales de medición predictiva sintética. Incluyen errores observados y límites de revisión humana; LR conserva el valor inicial y BETO V3 continúa opcional.


## BETO V3 local sin repetir entrenamiento

[Instalación Windows del artefacto recuperado y comprobación de ejecución real](docs/ACTIVAR_BETO_V3_WINDOWS_SIN_ENTRENAR_2026-11-13.md). El paquete externo conserva los pesos ajustados verificados del PR28, reutiliza la caché base y mantiene LR como valor inicial. No contiene una nueva receta ni una nueva evaluación predictiva.
