# Acta de aceptación funcional — Traveler Insights 2026

**Fecha del registro:** 6 de noviembre de 2026  
**Proyecto:** Traveler Insights 2026, prototipo académico de Inteligencia Artificial (UTB)  
**Tipo de acta:** aceptación funcional reportada por el responsable del proyecto; **no** certificación científica ni auditoría independiente del entorno Windows  
**Versión de referencia:** `main` SHA `52b9b1ce57910a07397de141b7adc91632e9b304`  
**PR de interfaz aceptado:** [#16, visualización jerárquica Midnight Indigo](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/pull/16), fusionado  
**Evidencia automatizada verificada:** [GitHub Actions run 42185971785](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/42185971785), Python 3.12: **180 passed, 1 xfailed**, conclusión `success`  
**Evidencia manual:** resultados comunicados por el responsable el 6 de noviembre de 2026; no se adjuntaron a esta acta capturas, archivos WAV ni exportaciones originales, y por tanto no se atribuye su inspección al autor del acta.

## 1. Objeto y alcance

Registrar el cierre funcional del prototipo con **un agente principal `TravelerOrchestrator` y cuatro especialistas**: Percepción, Semántica y sentimiento, Evaluación y control, Acción y recomendación. La aplicación conserva Gradio Midnight Indigo, clasificación DEMO TF-IDF + LR/SVM, ASR local, edición humana, lotes, JSON/CSV y visualización de la traza por delegaciones reales.

Se distinguen **funcionamiento del software**, **aceptación manual en Windows** y **validez estadística de los clasificadores**. Las dos primeras no acreditan la tercera.

## 2. Acta de las seis pruebas manuales reportadas

Las seis observaciones siguientes son declaraciones explícitas del responsable de proyecto y no resultados generados nuevamente durante la creación de este documento.

| ID | Prueba | Evidencia reportada | Resultado de aceptación |
| --- | --- | --- | --- |
| AM-01 | Ejecución normal | Estados `COMPLETADO` observados en `demo-001` y `demo-003` durante el análisis por lotes | **Aprobada (reportada)**: ruta de finalización normal |
| AM-02 | Audio WAV real y ASR | Grabación real de **10,2 segundos**; transcripción local con `faster-whisper tiny`; español reconocido | **Aprobada (reportada)**: percepción por voz a texto |
| AM-03 | Transcripción corregida | Edición del texto de Whisper, preservando original y corregido; `fue_corregida=true`, `multiagente.transcripcion_editada=true` | **Aprobada (reportada)**: procedencia y corrección humana |
| AM-04 | Procesamiento por lotes | **3 reseñas** procesadas; resultados positivo, negativo y neutral; trazas individuales completas | **Aprobada (reportada)**: flujo multipetición y trazabilidad |
| AM-05 | Revisión humana | `demo-002` con etiqueta corregida negativa y comentario; predicción del modelo conservada | **Aprobada (reportada)**: corrección no destructiva y sin aprendizaje en línea |
| AM-06 | Exportación | JSON y CSV producidos con **3 registros** y anotación humana conservada | **Aprobada (reportada)**: persistencia de salidas de sesión |

### Caso de aceptación visual previamente confirmado (PR #16)

**Texto de entrada:** «El guía fue amable, pero esperamos demasiado para entrar». El responsable verificó: sentimiento **negativo**, puntuación **no calibrada 0,393**, aspectos **servicio** y **espera**, contraste «pero», estado `REVISION_HUMANA`, **10 eventos** y coherencia entre tarjetas de cinco agentes y el JSON técnico completo. La recomendación fue revisión manual.

**Aceptación visual:** aprobada por prueba manual comunicada por el responsable. No demuestra que el modelo acierte en opiniones turísticas reales.

### Caso de voz: Centro Histórico de Cartagena

La reseña transcrita, según el responsable, describe una experiencia bonita en el Centro Histórico **pero menciona basura y calles sucias**. Se mantuvo la procedencia del texto original y corregido, sin publicar aquí la grabación ni el texto completo.

| Momento | Sentimiento DEMO | Puntuación no calibrada |
| --- | --- | ---: |
| Antes de la corrección | Negativo | 0,3739 |
| Después de la corrección | Negativo | 0,3750 |

Tras la corrección: aspecto **higiene**, estado **`REVISION_HUMANA`** y traza de **10 eventos** de coordinación de cuatro especialistas por el orquestador.

**Interpretación permitida:** la aplicación procesó y conservó una edición real, trazó el flujo y solicitó supervisión humana. Una variación entre 0,3739 y 0,3750 no constituye mejora validada del clasificador ni representa probabilidad calibrada.

## 3. Evidencias categorizadas y cobertura

| Fuente | Evidencia | Alcance | Límites |
| --- | --- | --- | --- |
| CI automatizada GitHub Actions | Run `42185971785` sobre SHA exacto de `main`: **180 passed, 1 xfailed** en Python 3.12, `success` | Regressión de código según la suite del repositorio, incluyendo prueba de UI multiagente y componentes científicos | CI no ejecuta un micrófono físico ni verifica apariencia en el navegador Windows |
| Prueba manual declarada | AM-01 a AM-06 y caso visual de revisión humana | Aceptación funcional del usuario en Windows, incluyendo ASR local y exportaciones | La declaración no sustituye capturas o evidencias instrumentadas que aún no se adjuntaron |
| Experimento científico sintético original | 336 reseñas sintéticas, semilla 42; test Macro-F1 LR/SVM **1,0000** | Comportamiento del clasificador en lenguaje de su generador original | No es precisión real; riesgo alto de atajos léxicos |
| Desafío sintético de generalización | 1.260 textos, 180 núcleos, semilla 913; Macro-F1 LR **0,3926**, SVM **0,3882** | Transferencia limitada hacia otro inventario **sintético** | Etiquetas de intención no son anotaciones humanas independientes; distribución diferente |
| Evaluación auténtica independiente | **No disponible** | No se emite estimación de desempeño humano real | Pendiente conseguir reseñas con permisos de entrenamiento/evaluación verificables y etiquetas adecuadas |

**Regla de integridad:** no se modificaron datos de los experimentos, modelos, parámetros ni umbrales para elaborar esta acta; tampoco se reabrió el test reservado para optimizar el agente.

## 4. Matriz de evidencia frente a la rúbrica académica de IA UTB

| Criterio de rúbrica | Componente/acción verificable | Evidencia vinculada | Alcance y honestidad |
| --- | --- | --- | --- |
| Problema y usuario | Apoyo a interpretación de reseñas de visitantes por operador humano | Caso del guía y caso de Cartagena, AM-01/AM-02 | Prototipo académico, sin estudio de usuarios |
| Percepción | PerceptionAgent recibe texto validado; audio se transcribe previamente por ASR local | AM-02, AM-03, procedencia de transcripción | No se infieren emociones desde la voz |
| Razonamiento / planificación | Orquestador mantiene plan de cuatro delegaciones, valida contratos y decide continuación | Trazas de 10 eventos, prueba visual PR #16, CI | Planificación **fija y condicionada**, no planificador general |
| Decisión | Control utiliza señales de puntuación y contraste para requerir revisión | Caso `REVISION_HUMANA`, caso Cartagena | Puntuaciones **no calibradas**; políticas demostrativas |
| Acción verificable | Acción produce recomendación o abstención sin llamadas a terceros | Estados `COMPLETADO` y `REVISION_HUMANA`, AM-01/AM-05 | No reserva ni contacta servicios turísticos |
| Memoria | Estado y traza de ejecución, procedencia y correcciones en sesión | AM-03, AM-04, AM-05, JSON | **Memoria episódica temporal**; no memoria de largo plazo |
| Aprendizaje | TF-IDF + LR/SVM entrenados offline; evaluación científica separada | Corpus sintético y desafío; CI del código | **Aprendizaje supervisado offline**, no online; la corrección humana no reentrena |
| Caso completo | Entrada → percepción → semántica → control → acción → salida y traza | Guía, Cartagena, AM-01 a AM-06 | Demuestra flujo funcional, no acierto estadístico real |
| Limitaciones, sesgos y mejoras | Restricciones científicas, puntuaciones sin calibrar, riesgo de fuga léxica | Dos experimentos sintéticos documentados | Test auténtico independiente sigue pendiente |

## 5. Acuerdo de cierre y pendientes

**Se registra aceptación funcional reportada:** AM-01 a AM-06 aprobadas por el responsable de proyecto y aceptación visual previa del caso de revisión humana.

**No se registra:** certificación de precisión del sentimiento, validación real de recomendaciones en turismo, pruebas con anotadores independientes, aprendizaje continuo o inferencia de emociones vocales.

**Pendiente para sustentación:** incorporar, si se dispone de ellas, capturas o exportaciones sanitizadas que corroboren la aceptación manual; no crear evidencia retroactiva. Verificar el alcance exacto de los criterios con la rúbrica docente. Conservar la misma arquitectura e interfaz y priorizar, después de la retroalimentación docente, un conjunto de reseñas humanas genuinas, autorizado y reservado para evaluar generalización.

**Control GitHub:** el acta se aporta mediante PR **únicamente documental** y no requiere cambios en código, clases, agentes, umbrales, corpus, modelos, CI o benchmarks. La integración del PR es decisión exclusiva del responsable.
