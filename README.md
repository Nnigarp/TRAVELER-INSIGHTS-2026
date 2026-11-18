# Traveler Insights 2026

Prototipo local de apoyo a la revisión de reseñas turísticas, desarrollado por **Nilver García Pimentel** para Inteligencia Artificial (ISCO-A06A, UTB). Recibe texto o una transcripción revisada, clasifica sentimiento, muestra aspectos y evidencia y propone una respuesta para valoración humana. También procesa lotes y conserva las correcciones sin sobrescribir la predicción original.

**LR es el clasificador inicial. BETO V3 es experimental y opcional. La generación Qwen comienza desactivada.** El sistema no realiza reservas, reclamaciones, contactos ni pagos. Su funcionamiento técnico no demuestra precisión turística ni seguridad para automatizar decisiones.

## Estado implementado, experimental y no validado

| Componente | Implementado y probado | Condición y límites |
|---|---|---|
| LR y SVM | Inferencia, selector Gradio, lotes y exportaciones; pruebas automáticas y funcionamiento local confirmado por el usuario | DEMO con 36 ejemplos ficticios. TF-IDF de palabras/caracteres; LR o LinearSVC con calibración interna. Las puntuaciones no tienen validación probabilística externa. |
| BETO V3 | Carga del artefacto ajustado, inferencia efectiva y comprobación de ausencia de fallback; ejecución Windows confirmada por el usuario | **Experimental sintético**: 300 reseñas TRAIN dentro de un piloto de 562 muestras admitidas. Los pesos se distribuyen fuera de Git. |
| Whisper Tiny | ASR local mediante faster-whisper, transcripción editable, procedencia y validaciones de audio; voz confirmada por el usuario | Reconocimiento de palabras; no detección de emociones. Ruido y negaciones requieren revisión. Sin evaluación WER humana publicada. |
| Qwen3 1.7B / Ollama | Generación opcional posterior a Control y validación por Acción. BETO + Qwen, LR + Qwen y audio confirmados por el usuario en Windows para PR #30 | **Redacción restringida**. El protocolo v2 de esta mejora separa síntesis/evidencia y adapta propuestas. Sus contratos y API simulada se prueban automáticamente; la aceptación de esta revisión con Qwen real en Windows queda pendiente. |
| Coordinación jerárquica | Un orquestador, cuatro especialistas, diez eventos principales y tres generativos adicionales cuando se activa Qwen | Plan fijo y memoria de sesión; no son cinco LLM autónomos ni aprendizaje en línea. |
| Evaluación predictiva externa | Infraestructura disponible para datos autorizados, particiones y comparación | **No validada con turistas reales independientes**. Las métricas históricas citadas abajo son sintéticas. |

## Arquitectura y responsabilidades

`TravelerOrchestrator` valida la entrada y coordina los contratos entre especialistas. El flujo principal ejecuta Percepción, Semántica, Control y Acción. Con generación activada consulta Ollama **después de Control y antes de Acción**.

| Componente | Responsabilidad |
|---|---|
| `PerceptionAgent` | Valida texto, longitud, ID y modalidad; recibe la transcripción corregida, no infiere emoción de la voz. |
| `SemanticsAgent` | Ejecuta LR/SVM/BETO, conserva sentimiento y puntuaciones, extrae aspectos por diccionario y fragmentos descriptivos. Los aspectos no son ABSA supervisado. |
| `ControlAgent` | Decide continuar o solicitar revisión mediante la política existente de puntuación/contraste. Qwen no puede modificar esa decisión. |
| Qwen / adaptador Ollama | Compone el informe con el esquema y las formulaciones admitidas para la evidencia. No clasifica ni tiene herramientas para actuar. |
| `ActionAgent` | Revalida informe, evidencia y coherencia con los resultados originales; registra propuestas y `acciones_externas_ejecutadas=false`. |

`multiagente.traza` conserva los **diez eventos principales**. `multiagente.traza_generacion` registra **tres eventos**: solicitud, respuesta y validación de Acción. Las trazas contienen metadatos, no razonamientos internos; el JSON conserva resultado, procedencia y, en audio, transcripción original/editada. Qwen no altera sentimiento, puntuaciones, aspectos o revisión humana.

## Entradas, revisión y exportaciones

- **Texto:** reseña individual de hasta 5.000 caracteres; vacío o exceso de longitud se rechazan.
- **Audio:** micrófono o archivo convertido por Gradio a WAV PCM; 0,25–120 s y hasta 12 MiB tras conversión. Whisper Tiny transcribe localmente. Revisa y corrige antes de analizar con el clasificador seleccionado y, opcionalmente, Qwen.
- **Lotes:** hasta 200 reseñas, una por línea, o CSV UTF-8 de hasta 1 MiB con columna `reseña`. CSV malformados se rechazan antes de clasificar. Las tendencias son conteos, no exactitud.
- **Revisión humana:** permite guardar etiqueta corregida y nota por ID. Mantiene predicción, revisión original e informe anterior; no reentrena ni regenera automáticamente.
- **JSON/CSV:** incluyen modelo efectivo, fallback explícito, texto y correcciones. Con Qwen incluyen estado, informe, evidencia literal, modo de síntesis y procedencia. CSV protege celdas que podrían interpretarse como fórmulas. Los archivos exportados pueden contener información privada.

## Interfaz UTB Claro Premium / Caribe Nocturno

Gradio conserva los componentes y callbacks reales. El diseño toma como referencia el prototipo visual **Traveler Insights 2026 — Cartagena · UTB Intelligence**, con dos temas seleccionables sin reiniciar ni volver a inferir. Usa fuentes locales disponibles y fuentes del sistema como alternativa; no requiere CDN. La navegación, el selector de tema y los controles admiten teclado y foco visible.

| Sección | Uso |
|---|---|
| **Análisis** | Selecciona LR/SVM/BETO disponible y activa Qwen opcionalmente. Entrada vacía inicial, contador, ejemplos identificados como demostrativos, análisis real y limpieza. Entrada/resultados en dos columnas amplias y apiladas en móvil. |
| **Audio** | Micrófono o archivo real, Whisper Tiny, transcripción editable y análisis posterior. Usa el modelo y la opción Qwen seleccionados en Análisis; conserva procedencia y correcciones. |
| **Lotes** | Texto o CSV, resultados y conteos descriptivos reales, corrección por ID, exportaciones y consulta de la traza de una fila. Conserva los límites existentes. |
| **Agentes** | Funciones y estados del último registro consultado en la sesión: orquestador y cuatro especialistas. Qwen/Ollama aparece separado como servicio subordinado. Sin registro no muestra tareas completadas. |

El informe presenta síntesis de ancho completo, indicadores compactos y una cuadrícula de dos columnas para problema, evidencia, recomendación y acción propuesta; en móvil se reduce a una. Advertencias y detalles técnicos quedan separados. **Sin Qwen o ante un fallo se identifica la orientación tradicional por reglas**, sin rellenar un resumen generativo ficticio. Las puntuaciones se rotulan como no calibradas. El JSON completo y las trazas siguen accesibles en secciones plegables.

La navegación utiliza pestañas nativas, y el tema un selector explícito en vez de los controles ilustrativos del HTML. UTB Claro Premium se aplica en la primera visita. La última selección se guarda en este navegador para la misma dirección local (incluido el puerto). Si el navegador bloquea el almacenamiento, el selector funciona durante la visita y se vuelve al tema claro al recargar. La adaptación y los contratos automáticos no sustituyen la aceptación de esta interfaz con los modelos, el micrófono y el navegador reales en Windows.

En Audio y Lotes se muestra la selección de clasificador/Qwen de Análisis y un acceso directo para cambiarla. El editor JSON ajusta líneas en pantalla y conserva el documento completo para copiar o descargar. Consulta la [auditoría y prioridades de UX/UI](docs/AUDITORIA_UX_UI_PREMIUM.md).

## Informes Qwen: síntesis con evidencia separada

El protocolo `ollama-tourism-report-v2` organiza la tarjeta en **resumen ejecutivo, sentimiento/aspectos, problema, evidencia, recomendación, acción propuesta, revisión humana y advertencias**. La información técnica queda plegada; el JSON y las trazas mantienen sus detalles.

La mejora no consiste en aceptar prosa libre por una coincidencia de palabras. Un contrato visible reconoce cláusulas completas de un conjunto acotado de hechos turísticos —comodidad, limpieza, servicio favorable, transporte ausente, cancelación, espera, discrepancia de cobro, ruido y algunas descripciones/dudas— y construye formulaciones respaldadas por esas cláusulas. Qwen selecciona entre ellas y entre propuestas contextuales. Las **citas literales se conservan separadas**. Cuando coexisten aspectos favorables y un incidente reconocible, la síntesis conserva ambos.

Ante sintaxis no reconocida, negación de alcance incierto, preguntas, ironía o regionalismos no cubiertos, utiliza una **cita conservadora** y puede dejar el incidente indeterminado. Se solicita aclaración en vez de inventar causas o responsables. El modo queda en `validacion_contenido`; esto es composición controlada, no una verificación semántica general. Las propuestas no prometen compensaciones ni afirman que se haya ejecutado una acción. La responsabilidad contractual y la veracidad del relato siguen sin verificarse.

| Control generativo | Comportamiento |
|---|---|
| Servicio y modelo | API local `http://127.0.0.1:11434`, `qwen3:1.7b`; endpoint HTTP(S) de loopback, sin proxies ni redirecciones |
| Activación | Checkbox desactivado inicialmente; sin consultas a Ollama cuando está apagado |
| Entrada | Hasta 1.200 caracteres, ocho unidades de evidencia de hasta 600; la clasificación mantiene su límite propio de 5.000 |
| Generación | `think=false`, temperatura 0, semilla 42, hasta 1.024 tokens y contexto 4.096; no asegura determinismo entre runtimes |
| Tiempo y lotes | Timeout de pared de 120 s, configurable hasta 600. Hasta cinco consultas secuenciales por lote, con presupuesto compartido |
| Fallo / rechazo | Clasificación y recomendación tradicional por reglas conservadas; `generacion_llm=false`, `informe=null`, código explícito. Sin reintento ni sustitución generativa |

Las filas sin presupuesto generativo siguen clasificándose. Thinking separado o bloques completos se descartan; respuestas incompletas, JSON inválido, cambios de campos bloqueados, evidencia no verificable o propuestas desconectadas del problema se rechazan. Una validación aprobada no certifica que el relato sea verdadero ni resuelve todas las alucinaciones.

## Requisitos y ejecución en Windows

Entorno de uso confirmado por el usuario: **Windows, Python 3.12.7, `.venv-research`, Intel Core i7-13620H y 15,73 GB de RAM**. Es una referencia operativa, no una medición de memoria mínima o una garantía de latencia. `pyproject.toml` declara Python >=3.10; CI usa 3.12.

Si ya funcionan los modelos, conserva el entorno y sus cachés. Desde la raíz del repositorio:

```powershell
.\.venv-research\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python app.py
```

Gradio abre una URL local, normalmente `http://127.0.0.1:7860`, sin compartir públicamente. Selecciona LR/SVM o BETO cuando el artefacto esté disponible; marca Qwen expresamente si necesitas informe.

Para una instalación nueva de la ruta textual LR/SVM:

```powershell
py -3.12 -m venv .venv-research
.\.venv-research\Scripts\Activate.ps1
python -m pip install -e ".[ui,test]"
python app.py
```

ASR necesita el extra `.[asr]` y una caché local de Whisper Tiny; la primera descarga requiere autorización explícita. Consulta el [manual de audio](docs/INCREMENTO_07_AUDIO_ASR.md). BETO necesita las dependencias fijadas del entorno de investigación y el artefacto verificado: [activación sin entrenar](docs/ACTIVAR_BETO_V3_WINDOWS_SIN_ENTRENAR_2026-11-13.md). Clonar el repositorio no incluye pesos ajustados ni descarga modelos. Ollama y `qwen3:1.7b` se instalan aparte; no hace falta un SDK Python de Ollama.

Para la ruta textual en Linux/Colab se conserva la instalación `python -m pip install -e '.[ui,test]'` y la ejecución `python app.py`; en un entorno remoto, el micrófono y la API Ollama de tu portátil no se comparten automáticamente.

### Verificar BETO y Qwen sin repetir entrenamiento

Con el artefacto ya instalado y Ollama abierto:

```powershell
$env:TRAVELER_BETO_ARTIFACT = (Resolve-Path '.\artifacts\beto-v3-pr28-local').Path
$env:TRAVELER_OLLAMA_ENDPOINT = 'http://127.0.0.1:11434'
$env:TRAVELER_OLLAMA_MODEL = 'qwen3:1.7b'
$env:TRAVELER_OLLAMA_TIMEOUT = '120'
ollama list
python scripts/check_beto_activation.py --artifact artifacts/beto-v3-pr28-local --output artifacts/beto-readme-01.json
if ($LASTEXITCODE -ne 0) { throw 'BETO efectivo pendiente; revisar el JSON.' }
python scripts/check_qwen_local.py --classifier beto --output artifacts/qwen-v2-check-01.json
if ($LASTEXITCODE -ne 0) { throw 'BETO + Qwen pendientes; revisar el JSON.' }
python app.py
```

Adapta únicamente la ruta si tu artefacto tiene otro nombre. Los comprobadores preservan la evidencia anterior: usa un nombre nuevo en cada repetición. BETO requerido no acepta LR como éxito. Qwen requiere respuesta validada y versión/digest declarados por la API local; ese digest no es una atestación remota. El JSON debe mostrar `modelo_utilizado=beto`, `seleccion_modelo.fallback=false`, `generacion_qwen.generacion_llm=true`, `modelo_ejecutado=qwen3:1.7b` y `validado_por=ActionAgent`.

Para repetir **12 casos variados de informe** sobre el servidor Gradio ya abierto, en otra terminal con el mismo entorno y variables:

```powershell
python scripts/check_qwen_report_quality.py --actual --classifier beto --gradio-url http://127.0.0.1:7860 --output artifacts/qwen-v2-quality-windows-01.json
```

Son doce inferencias secuenciales de comprobación, no entrenamiento. El informe conserva respuestas, fallos, citas conservadoras y duplicaciones observables. Sin `--actual`, el mismo script solo comprueba el contrato: **no ejecuta Qwen ni clasificador** y lo declara expresamente. [Procedimiento de aceptación Windows](docs/QWEN_OLLAMA_WINDOWS.md).

## Pruebas automatizadas

```powershell
python -m pytest -q
python -m pytest -q tests/test_ollama_report.py tests/test_qwen_integration.py tests/test_qwen_gradio_http.py tests/test_qwen_local_check.py tests/test_qwen_report_quality.py
python scripts/check_qwen_report_quality.py --output artifacts/qwen-contracts-01.json
```

La suite completa utiliza el entorno de investigación para los contratos de optimización existentes; la ruta `.[ui,test]` ejecuta los contratos disponibles sin instalar pesos. CI de Qwen en Linux y Windows usa HTTP Ollama **simulado**, LR/SVM de la DEMO y una interfaz BETO inyectada para probar consistencia/fallback. También ejecuta cola, sesiones, cargas CSV, correcciones y descargas por **HTTP Gradio real**. No descarga Qwen ni exige Ollama real. Los tests de optimización con fixtures pequeños no constituyen un nuevo experimento BETO.

El workflow `ui-browser.yml` añade contratos de presentación en Linux/Windows y un navegador Chromium en Linux sobre Gradio real: cuatro pestañas, ambos temas, LR/SVM, revisión de lotes, descarga JSON y disposición móvil. Conserva capturas y un registro de comprobaciones como artefacto de CI. Playwright es una dependencia exclusiva de esa prueba, no de la aplicación. No usa pesos BETO/Whisper ni Ollama real; los informes Qwen v2 y sus fallos se cubren con API simulada en los contratos existentes.

La batería de contenido está versionada en [tests/fixtures/qwen_report_quality_v2.json](tests/fixtures/qwen_report_quality_v2.json). Incluye comentarios positivos, negativos, mixtos, neutrales, ambiguos, insuficientes y de alcance incierto. Las pruebas adversariales rechazan invenciones de hoteles, fechas, pagos, atribuciones, evidencia alterada y supresión de revisión. Una API simulada aprobada no demuestra calidad de las respuestas del Qwen real; esta revisión se debe aceptar nuevamente en Windows.

## Resultados históricos y límites científicos

Los informes congelados se mantienen. El [piloto BETO V3](docs/BETO_V3_PILOTO_Y_ESCALADO_2026-11-10.md) distingue la reproducción de una cohorte ya observada de un test nuevo y documenta la pérdida de probabilidades originales. La [evaluación de aceptación de noviembre de 2026](docs/ANALISIS_ACEPTACION_2026-11-13.md) usa 36 casos sintéticos, doce por clase, más ocho ambiguos:

| Configuración histórica | Macro-F1 sintético /36 | Negativos → positivos /12 |
|---|---:|---:|
| LR — DEMO36 | 0,4056 | 0 |
| SVM — DEMO36 | 0,3702 | 0 |
| BETO V3 — TRAIN300 | 0,9722 | 1 |

Los modelos usan fuentes y cantidades de entrenamiento diferentes. La comparación no aísla el efecto de la arquitectura ni estima generalización. BETO aceptó ese error sin revisión y derivó **0/8** ambiguos; LR/SVM derivaron **33/36** casos. Ninguno de esos resultados valida automatización segura. Esta mejora del informe no reentrena modelos, no modifica la política ni vuelve a presentar esos casos como una evaluación intacta.

- Los datos y etiquetas experimentales son sintéticos, no un corpus humano independiente de Cartagena. No se afirma rendimiento humano validado ni precisión perfecta.
- Puntuaciones, aspectos por diccionario y fragmentos descriptivos no son probabilidades externamente fiables, ABSA entrenado ni explicaciones causales.
- BETO tiene una ventana limitada de tokens; una reseña larga puede recortarse en su inferencia aunque respete el límite de caracteres.
- El contrato Qwen cubre patrones limitados. Sintaxis nueva, ironía y regionalismos pueden conservar la reseña literalmente y dejar el incidente indeterminado. La pertinencia final y la verdad del relato necesitan revisión humana.
- Whisper puede cambiar negaciones o nombres por ruido. La transcripción debe revisarse antes del análisis. No se publica una métrica de voz humana.
- No hay acciones externas, RAG, recuperación web, aprendizaje en línea ni ABSA supervisado implementados en esta ruta. La ampliación a 5.000 reseñas sigue siendo un plan histórico, no trabajo realizado en esta revisión.

## Datos autorizados y evaluación separada

El CSV debe contener columnas textuales `reseña` y `sentimiento`, este último con etiquetas **positivo, neutral, negativo** (sin números). En caso de que Kaggle proporcione 0/1/2, **se debe comprobar el mapeo contra archivos, descripciones y ejemplos oficiales antes de convertirlo**. La descripción pública del reto contiene inconsistencias. No asumir una correspondencia por intuición.

```bash
traveler evaluate --csv data/mi_dataset_verificado.csv --text-col reseña --label-col sentimiento --out artifacts/holdout.json
traveler train --csv data/mi_dataset_verificado.csv --text-col reseña --label-col sentimiento --out artifacts/real_model.joblib
traveler predict --model artifacts/real_model.joblib --text "La atención fue muy amable."
```

**Precaución metodológica:** primero evaluar una partición independiente; el comando `train` entrena posteriormente con todos los registros válidos para crear el artefacto final. Evite evaluar el mismo modelo en sus propios datos de entrenamiento. Para publicación científica, también exigir validación cruzada, partición por lugar/fecha si aplica, calibración y evaluación externa con muestras de Cartagena.

**Seguridad:** los archivos `.joblib` solamente deben abrirse si se crearon localmente y provienen de una fuente confiable; deserializar un archivo ajeno puede ejecutar código. El entrenamiento no descarga modelos remotos ni llama a servicios de IA.

## Corpus propio y benchmark: infraestructura conservada

El gestor permite registrar fuentes autorizadas, anotaciones, adjudicación razonada y particiones reproducibles por grupo/duplicado. Los datos privados viven bajo `data/private/` y `data/processed/`, excluidos de Git. No certifica permisos legalmente ni anonimiza por completo; los candidatos a duplicados requieren revisión. Consulta [Corpus turístico](docs/CORPUS_TURISTICO_2026.md), [recogida y anotación ciega PILOTO-01](docs/PILOTO01_RECOGIDA_HUMANA_Y_ANOTACION_CIEGA_2026-11-07.md) y [benchmark del corpus](docs/INCREMENTO_09_BENCHMARK.md).

El benchmark compara Dummy, LR y SVM sobre particiones compartidas y comprueba hashes y separación de grupos. Selecciona hiperparámetros en validation y calibra SVM dentro de train. Estas capacidades de infraestructura no son resultados predictivos de turistas reales.

## Referencias y documentación existente

- [Fuente histórica del reto Traveler Insights 2026](https://www.kaggle.com/competitions/travelerinsights2026/overview). Cartagena es un dominio futuro de aplicación; no se asume origen local de los datos sintéticos.
- [Lotes y revisión humana](docs/INCREMENTO_04_LOTES_REVISION.md) · [LR/SVM](docs/INCREMENTO_05_MODELOS_LR_SVM.md) · [audio](docs/INCREMENTO_07_AUDIO_ASR.md).
- [BETO supervisado inicial](docs/AVANCE_EJECUTABLE_BETO_Y_DEMO_2026-11-07.md) · [BETO Training V2](docs/BETO_TRAINING_V2_2026-11-08.md) · [piloto BETO V3](docs/BETO_V3_PILOTO_Y_ESCALADO_2026-11-10.md).
- [Preparación operativa y micrófono](docs/SUSTENTACION_Y_ACEPTACION_2026-11-13.md) · [análisis congelado de aceptación](docs/ANALISIS_ACEPTACION_2026-11-13.md).
- [Protocolo de investigación existente](docs/PROTOCOLO_INVESTIGACION.md), [bitácora](docs/BITACORA.md), [decisiones](docs/DECISIONES.md), [fuentes](docs/FUENTES.md) y [plan GitHub](docs/PLAN_GITHUB.md) conservan el historial; sus estados y pendientes deben leerse en su fecha original.
