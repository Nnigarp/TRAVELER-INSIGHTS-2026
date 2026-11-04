# Estrategia de datos multidominio y validación externa — Traveler Insights 2026

**Estado:** propuesta de investigación, sin resultados empíricos ni datos descargados.  
**Fecha:** 2026-11-04  
**Rama:** `feat/official-data-protocol`

## 1. Alcance y principio central

Traveler Insights 2026 es una plataforma de análisis de experiencias turísticas; el concurso Kaggle Traveler Insights 2026 es **una fuente de referencia**, no la definición del producto ni una prueba de generalización por sí misma.

### Fuentes posibles, sujetas a derechos y calidad

| Fuente | Finalidad | Condición previa |
|---|---|---|
| Archivos oficiales de la competencia | Benchmark comparable con sus reglas y métricas | Descargar con autorización; verificar esquema, mapeo y licencia |
| Corpus abiertos de reseñas turísticas | Validación externa y cambio de dominio | Verificar licencia, idioma, procedencia y términos de reutilización |
| Reseñas proporcionadas voluntariamente y con consentimiento | Evaluación local (por ejemplo, Cartagena) | Minimizar datos personales, documentar consentimiento, anonimización y retiro |
| Casos sintéticos | Pruebas unitarias y de integración | Señalización explícita; nunca presentarlos como evaluación científica |

**No recopilar reseñas de sitios comerciales mediante scraping sin verificar sus condiciones y el tratamiento legal de datos.** Ningún dataset se distribuye automáticamente en GitHub.

## 2. Contrato canónico de datos propuesto (todavía no implementado)

Un adaptador por fuente mapea sus columnas originales, sin alterar el raw, a:

- `review_id`: identificador seudónimo; nunca el identificador personal público.
- `text`: reseña original en memoria o almacenamiento local autorizado.
- `sentiment_label`: `positivo`, `neutral` o `negativo` **solo cuando el significado esté verificado**; en otro caso permanece sin mapear.
- `source_id`, `dataset_version`, `source_license`, `source_sha256`: trazabilidad.
- `language`: idioma detectado con método y posibilidad de `unknown`/código mixto.
- `place_group`, `date_bucket`: **opcionales**, sujetos a privacidad y procedencia.
- `split`: asignación controlada, nunca definida por el modelo.
- `label_provenance`: anotación humana, metadato, transformación o desconocida; evitar etiquetas construidas directamente desde features accesibles.
- `annotation_status`: calidad, discrepancias y adjudicación cuando corresponda.

Las etiquetas por aspecto y los spans de evidencia son **tablas de anotaciones separadas**, no valores inferidos tratados como verdad.

## 3. Protocolo de evaluación reproducible

1. Inventariar origen, URLs, SHA-256, fecha de acceso, licencia y restricciones de todos los archivos. Inspeccionar etiquetas antes de cualquier entrenamiento.
2. Auditar nulos, clases, repetición exacta y casi duplicados, textos cortos y largos, emojis/emoticonos, idiomas, metadatos y solapamientos entre fuentes.
3. Reservar un test por fuente antes de seleccionar hiperparámetros. Identificar solapamientos y, cuando proceda, separar por establecimiento/autor/fecha para evitar fugas. Eliminar identificadores y variables que codifiquen la etiqueta (por ejemplo, una puntuación de estrellas si originó el sentimiento).
4. Comparar bajo idénticos splits: *DummyClassifier* de mayoría/estratificado, TF-IDF palabras+caracteres con regresión logística y LinearSVC. Seleccionar parámetros únicamente sobre desarrollo/validación.
5. Añadir un modelo contextual multilingüe con ajuste fino **solo después** de establecer baselines y revisar licencia, recursos, coste y riesgos.
6. Evaluar accuracy, Macro-F1, precision/recall/F1 por clase, matriz de confusión, incertidumbre del intervalo y errores por negación, emojis, opiniones mixtas, idioma y dominio. Las métricas por subgrupo requieren soporte de muestras adecuado.
7. Estudiar **ablaciones de emojis**: (a) texto intacto, (b) texto sin emojis, (c) verbalización de emojis. No fijar polaridad a 🙃, 😂 o ❤️ sin contexto. Mantener test independiente y no optimizar sobre los cinco ejemplos manuales.
8. Calibrar puntajes y estudiar abstención solo en validación; reportar cobertura vs riesgo en test. No convertir `predict_proba` sin calibrar en certeza.
9. Evaluar ABSA mediante anotación humana por aspecto (polaridad y spans). Medir F1 de extracción por aspecto y acuerdo interevaluador; no confundir detección de palabras con ABSA supervisado.
10. Evaluación externa sobre nuevas reseñas de Cartagena u otro destino **si se obtienen legalmente**, sin entrenar sobre ellas antes de la primera medición de transferencia.

No comparar números obtenidos en conjuntos diferentes como si fueran una misma competición. Se publicarán también casos de error y limitaciones.

## 4. Computación en Paralelo y rendimiento

Primero medir auditoría y predicción **secuenciales**; evaluar paralelismo por archivos/lotes independientes cuando el tamaño y la memoria lo justifiquen. Registrar `T1`, `Tp`, speedup, eficiencia, núcleos, RAM, librerías y tamaño del corpus. Reglas invariables:

- mismos resultados y el mismo orden de salida (o orden canónico explícito);
- sin retrasos artificiales ni trabajo falso;
- evitar sobresuscripción de procesos y BLAS/OpenMP;
- los splits, las semillas y el test reservado no cambian entre implementaciones;
- mantener modo secuencial como opción cuando el paralelo sea más lento.

## 5. Arquitectura de agentes

- **Percepción:** texto, idioma, contexto verificable y validación.
- **Estado:** representación estructurada de sentimiento, aspectos, spans y calidad de evidencia.
- **Razonamiento:** estimación de incertidumbre, reglas explícitas, restricciones y política de abstención.
- **Acción:** recomendaciones por aspecto, revisión humana o ausencia de acción cuando corresponda.
- **Aprendizaje y evaluación:** actualización de modelos offline con dataset versionado, no aprendizaje oculto durante inferencia.

Prolog o búsqueda BFS/DFS se incluirán únicamente si existe una tarea concreta y una evaluación comparativa que los justifique; no son adornos de arquitectura. Recuperación semántica se contempla solo para búsqueda de reseñas similares y evidencia, separando texto recuperado de evidencia causal.

## 6. Criterios para cerrar el siguiente incremento

- [ ] Se conocen los nombres y procedencia de los CSV oficiales.
- [ ] Se revisan permisos de uso y no se publican raws en Git.
- [ ] Se verifica semántica de etiquetas (incluidos 0/1/2 si aparecen).
- [ ] Informe auditado localmente sin divulgar textos de terceros.
- [ ] Se documentan decisiones de split y control de fuga.
- [ ] CI y pruebas verdes en PR.
- [ ] Baselines científicos solo después de completar los controles anteriores.

**Límite actual:** este documento es un protocolo, no prueba de que se hayan descargado datos oficiales ni de que un método sea superior.
