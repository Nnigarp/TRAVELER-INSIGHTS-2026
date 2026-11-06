# PILOTO-01 — Auditoría documental y puerta de admisión de datos (2026-11-06)

**Registro inicial del PR #18.** La revisión posterior de Work, los accesos nuevos a Kaggle/UNED y las correcciones de la herramienta se documentan en [PILOTO01_REVISION_WORK_2026-11-06.md](PILOTO01_REVISION_WORK_2026-11-06.md). Las afirmaciones de inaccesibilidad de esta primera auditoría describen su sesión original, no el estado posterior.

**Resultado:** `NO_CORPUS_ADMITTED`. Este documento informa evidencia obtenida de fichas públicas y distingue hechos **comprobados externamente** de las propiedades **que no pudieron inspeccionarse en archivos**. Ningún corpus externo se descargó, importó, adjudicó o utilizó para entrenar. Los experimentos sintéticos #12 y #13 permanecen congelados.

**Estado inicial del repositorio:** `main` SHA `8a3576d0cfba8e0af910587820b91616f2236fc2`; PR #17 fusionado. El manifiesto auditable de candidatas está en [piloto01_candidate_manifest.json](piloto01_candidate_manifest.json).

## 1. Identidad y bloqueo de la competencia exacta

**Competencia correcta:** `travelerinsights2026`, https://www.kaggle.com/competitions/travelerinsights2026/overview ; **NO** confundir con `traveler-insights-utb` o `traveler-insights-uptc-2026`.

### Evidencias verificadas

1. Página pública de descripción: inicio **19 de mayo**, cierre **31 de mayo de 2026**; reseñas de visitantes de plataformas `like Foursquare` (no confirma muestra, autoría ni licencia original de cada texto). Fuente: https://www.kaggle.com/competitions/travelerinsights2026/overview
2. Ficha **Data** muestra **CC BY-SA 4.0** como metadata de la competencia, pero **no** demuestra que contenido de terceros esté cubierto íntegramente ni aclara restricciones adicionales de uso, publicación y redistribución. Fuente: https://www.kaggle.com/competitions/travelerinsights2026/data
3. Página **Rules** existe pero no ofreció contenido comprobable desde la herramienta de lectura: https://www.kaggle.com/competitions/travelerinsights2026/rules . No se trasladan las reglas de otras competencias.
4. Página oficial **Leaderboard** indica que el marcador privado utiliza aproximadamente **50 % del test**. La existencia de puntuación privada NO equivale a disponibilidad de las etiquetas gold ni a autorización de usarlas para ajuste: https://www.kaggle.com/competitions/travelerinsights2026/leaderboard
5. **Inconsistencia literal:** el texto de Evaluation dice `positive for 2, neutral for 1, and negative for 1`, mientras el ejemplo de entrega contiene `1001,2` / `1002,1` / `1003,0`. No hay resolución documental concluyente del código del negativo. El ejemplo **sugiere** 0, pero **no prueba** el mapeo. No se utiliza ningún mapeo deducido para entrenamiento.
6. La página Data accesible no permitió confirmar los archivos, las cabeceras de un CSV, conteos, distribución, idiomas, registro de anotadores ni autoría de etiquetas. La interfaz de API de Kaggle para listar datos tampoco estuvo accesible en esta sesión. **Ningún archivo de la competencia fue leído**.

### Decisión de admisibilidad

| Operación | Estado | Razón |
| --- | --- | --- |
| Lectura de metadatos públicos de la competencia | Permitida y realizada | Páginas públicamente visibles |
| Descargar CSV de entrenamiento | **No realizado** | No se acreditó acceso a los archivos ni se aceptaron reglas en nombre del usuario |
| Entrenar en datos Kaggle | **Bloqueado** | Licencias/reglas y origen de etiquetas pendientes |
| Evaluar sobre Kaggle como gold humano | **Bloqueado** | No hay verificación de anotación humana independiente |
| Publicar **métricas agregadas obtenidas legítimamente** | **Condicionado** | Necesario revisar reglas y obligaciones del conjunto específico antes de publicación |
| Redistribuir reseñas o CSV en GitHub | **No autorizado en esta auditoría** | La ficha CC BY-SA no despeja derechos de terceros ni reglas de competencia |
| Resolver 0/1/2 | **Pendiente de fuente autoritativa** | No optimizar contra test ni reinterpretar para elevar métricas |

## 2. Alternativas contrastadas con fuente primaria

| Conjunto | Tamaño que declara la fuente | Etiquetas y procedencia publicada | Licencia / acceso real | Estado |
| --- | ---: | --- | --- | --- |
| [HotelReview, UNED](https://nlp.uned.es/~lplaza/resources/hotel-review-corpus/) | 1.000 Booking.com | Categorías **manualmente anotadas** 3 clases (Good/Fair/Poor) y 5 clases; score 0–10 calculado por Booking.com a partir de valoraciones de aspectos | Descarga ZIP de XML **para investigación** indicada en página oficial, sin términos explícitos de redistribución identificados | Candidata condicionada; ZIP detectado pero descarga no completada en entorno. **Idioma no verificado desde XML.** |
| [COAR, SINAI](https://huggingface.co/datasets/SINAI/COAR) | **2.202** opiniones españolas de restaurantes TripAdvisor | Escala ordinal 1–5: **565, 246, 188, 333, 870**, respectivamente; sin verificación del método de asignación de las etiquetas | Ficha `cc-by-nc-sa-4.0`; **gated**, pide aceptar intercambio de información de cuenta con autores | **Acceso no otorgado / no solicitado**; no derivar automáticamente gold humano de ratings |
| [Hopinion, CLiC-UB](https://clic.ub.edu/corpus/es/hopinion) | **17.934** opiniones, **2.388.848** palabras, principalmente hoteles TripAdvisor; 4.740 textos con anotación morfológica | Metadatos de ítem incluyen puntuaciones, y metadatos de usuario pueden incluir alias, edad, sexo, procedencia y otros atributos | Descripción pública menciona descarga CSV/base de datos; condiciones legales precisas no verificadas | **No admitido**; evitar PII y no convertir puntuaciones de hotel en sentimiento humano adjudicado |

Las estadísticas precedentes son declaraciones de las fichas oficiales, NO conteos producidos desde archivos locales. En COAR 565 + 246 + 188 + 333 + 870 = **2.202**, coherente con su ficha. Eso no demuestra que una transformación de cinco clases a tres sea válida; exigiría justificación previa.

## 3. Herramienta de inspección local, sin entrenamiento

Archivo: `src/traveler_agent/corpus_metadata_probe.py`; **biblioteca estándar exclusivamente**. Para un CSV legítimamente obtenido y revisado, inspecciona cabeceras reales, conteos, SHA-256, duplicados exactos normalizados, conflictos de etiquetas, grupos, duplicados de ID y solapamientos de particiones propias autorizadas. Desde la corrección v2, los valores de etiqueta/idioma se ocultan por defecto; solo se muestran códigos originales incluidos explícitamente en una lista de valores revisados. No se traduce su significado. Las cabeceras y esos códigos siguen siendo metadatos privados que necesitan revisión antes de compartir. El programa no certifica anonimización.

Ejemplo sintáctico para un archivo local con columnas **confirmadas al abrirlo**, no asumidas en Kaggle:

~~~powershell
python -m traveler_agent.corpus_metadata_probe --train data/private/kaggle_local/train.csv --text-column "Comentario" --label-column "Sentimiento" --id-column "ID" --group-column "Nombre del lugar"
~~~

En la sesión inicial estos nombres procedían de otra competencia. Posteriormente se observaron también en la **descripción** de Data de `travelerinsights2026`, pero todavía no se han verificado como cabeceras del CSV real. Primero hay que disponer del archivo autorizado y confirmar su esquema. Para mostrar códigos crudos ya revisados, agregar, por ejemplo, `--allow-label-value 0 --allow-label-value 1 --allow-label-value 2`; eso NO establece el mapeo numérico a sentimiento. Los archivos permanecen en `data/private/` u otra ubicación ignorada por Git. El test de Kaggle queda excluido de este piloto; `--test` se reserva para particiones propias autorizadas y no ocultas.

Al repetir `--group-column hotel --group-column autor`, v2 detecta valores compartidos de hotel **o** autor por separado; no exige que coincida la pareja. Esos conteos todavía no son componentes conectadas ni crean particiones. La semántica del informe está versionada como `piloto01_metadata_probe_v2`.

La salida marca incondicionalmente `annotation_origin=UNVERIFIED`, `mapping=UNRESOLVED_NEEDS_ORIGINAL_DATA_DICTIONARY`, `near_duplicate_semantic_audit=NOT_RUN`, `privacy_review=MANUAL_PENDING` y `admission=BLOCKED...`. Un programa no puede inferir permisos legales ni calidad gold a partir de CSV.

## 4. Riesgos de fuga y protocolo de partición NO ejecutado

Una vez elegida una fuente admisible con esquema verificado:

1. Revisar origen y derechos por fuente, privacidad del texto y, si existe, metodologías de anotación y adjudicación humana. Diferenciar **etiqueta humana independiente**, **puntuación de estrellas convertida** y **etiqueta sintética**.
2. Examinar grupos con identidad semántica: `source_id`, `group_id` (p. ej., lugar/alojamiento, usuario anonimizado cuando exista y sea legal usarlo), texto normalizado y duplicados exactos. Unir registros relacionados en componentes conectadas; no poner componentes relacionadas en particiones distintas. Auditar casi duplicados semánticos aparte, sin afirmar que la inspección exacta es suficiente.
3. Propuesta provisional: **60 % train / 20 % validation / 20 % test** por grupos independientes, sujeto a viabilidad de clases y número real de grupos. Reutilizar `corpus_splits.py` para datos adjudicados, con revisión previa de grupos y casos ambiguos.
4. Seleccionar `C` de LR y SVM mediante **validación**; calibración de SVM y preprocesamiento dentro de folds de **TRAIN**. Comparar Dummy, TF-IDF+LR y TF-IDF+SVM con Macro-F1, F1/recall por clase y matrices de confusión; reportar grupos por fuente e idioma auditado.
5. Solo cuando haya validación humana, predefinir evaluación de negaciones, contrastes y ambivalencia, y curva riesgo/cobertura para abstención. **No usar test oculto de Kaggle para elegir mapeos, parámetros, umbrales o ejemplos.**
6. Test auténtico final se ejecuta una vez con configuración sellada, y queda reservado para comparación posterior con BETO. Nada de este procedimiento ocurrió en esta fase.

## 5. Criterios de admisión y próximos pasos bloqueantes

**Admitir fuente únicamente si:** (A) licencia/reglas y derechos de contenido permiten la investigación prevista, (B) formato/CSV inspeccionados y hashes guardados, (C) origen de reseñas e idioma conocido, (D) taxonomía de etiquetas y mapeo definido por documentación fiable o anotación humana independiente, (E) privacidad y duplicados auditados, (F) grupos suficientes para un test independiente y sin fuga.

**Rechazar o poner en espera** si hay permiso ambiguo, datos con identificadores privados sin protección, etiquetas inferidas sin documentación, train/test mezclados, clases faltantes o volumen insuficiente de grupos. Si no hay gold humano válido, se admite como máximo una **evaluación proxy** con su advertencia explícita, nunca una afirmación de generalización humana verificada.

**Estado del incremento:** solo herramienta de auditoría, fixtures sintéticos de pruebas y documentos. No altera agentes, Gradio, audio, políticas, umbrales, corpus histórico, modelos, benchmarks o estado de producción.
