# V3: cantidad, familias semánticas y riesgo selectivo

Base: `main@6f7334b90b5655535b3fac41687febf690781106`, PR #21 fusionado por el responsable. Este protocolo y los nuevos literales se publican antes de la primera inferencia sobre la evaluación V3. El informe identificará el commit. Los datos, código, resultados y política de V2 se conservan; sus 54 casos evaluados y nueve ambiguos no seleccionan ninguna técnica o umbral V3.

## Inspección previa y límites de V2

Se revisaron código y resultados agregados de #21. La cobertura previa a redactar V3 se calculó únicamente con metadatos de TRAIN y desarrollo V2: 13/15 contrastes y 6/7 opiniones de aspectos mixtos en TRAIN tenían intención negativa; solo 1/36 negaciones era neutral. El recurso no permite atribuir causalmente un beneficio a cada dominio/aspecto: no registraba esas dimensiones de forma controlada y no había un factorial.

El informe previo se conserva en [v2-coverage-before-authoring.json](research/v3/v2-coverage-before-authoring.json). No se ejecutó el benchmark V2 para preparar esta fase. Los textos V2 completos se usan mecánicamente como lista de hashes prohibidos, sin tomar sus etiquetas o ejemplos reservados para redactar, ajustar o puntuar V3. La suite histórica puede seguir verificando sus propios contratos de software; sus resultados no son evidencia nueva de aprendizaje V3.

Antes de congelar se examinó el primer borrador V3 con una sonda de metadatos, validación cruzada por familias dentro de TRAIN. La forma del discurso seguía demasiado asociada a la clase: neutrales descriptivos frente a opiniones narrativas. Se reescribieron **solo textos de TRAIN**, incorporando relatos neutrales, diálogos en las tres clases y observaciones breves de las tres intenciones. Las etiquetas de fenómenos se revisaron al reescribir. No se observaron predicciones de sentimiento de la evaluación V3 para esta corrección.

El [registro del control de redacción](research/v3/train-authoring-quality-audit.json) conserva hashes y diagnósticos del borrador y de la versión revisada. La sonda completa usa dominios, aspectos, región, longitud, forma y fenómenos; algunos fenómenos describen semántica relacionada con la propia intención (por ejemplo, palabras positivas en contexto negativo), de modo que su puntuación **no mide por sí sola un atajo textual**. Otra sonda usa únicamente longitud y puntuación. Ninguna de esas características se entrega a los clasificadores de sentimiento. Los cierres de cuatro palabras repetidos entre familias se auditan; no se afirma ausencia de todo atajo o paráfrasis.

El registro incluye exclusivamente el TRAIN descartado del borrador para reproducir ese diagnóstico; no se admite como fuente de entrenamiento del factorial. El diagnóstico no mejoró en todas sus dimensiones: en `broad_120`, la sonda completa pasó de 0.879 a 0.751, mientras la sonda de forma pasó de 0.400 a 0.561. Son señales residuales que se informarán, no un criterio superado de eliminación de atajos.

## Catálogo y particiones

Los textos de `research_data/synthetic_diversity_v3.json` son literales completos redactados individualmente por el asistente. La reproducción vuelve a materializar esos literales; una semilla no recrea la redacción del modelo. Se declara modelo exacto no identificado de forma fiable, muestreo no disponible, ausencia de autores humanos e intención del autor como etiqueta. No se inventan reseñistas reales.

| Rol | Familias | Filas | Uso permitido |
| --- | ---: | ---: | --- |
| TRAIN, pool | 20 | 180 | Ajuste de clasificadores, sondas de atajos y muestreo registrado |
| Tune | 10 | 30 | C de LR/SVM y ablaciones exploratorias |
| Calibration | 10 | 30 | Una temperatura por clasificador seleccionado |
| Policy development | 10 | 30 | Selección de política de revisión |
| Assessment | 15 | 45 | Evaluación después de congelar todas las decisiones |
| Ambigüedad de desarrollo | 10 | 10 | Expectativa de derivación a revisión, sin sentimiento gold |
| Ambigüedad de evaluación | 10 | 10 | Derivación prospectiva, sin sentimiento gold |

Son 335 textos, 315 con intención de sentimiento y 20 ambiguos. Las variantes relacionadas permanecen juntas por familia/componente. Cada familia etiquetada contiene las tres intenciones; cada variante se conserva dentro de TRAIN. Los cinco dominios son hotel, restaurante, museo, transporte y recorrido; los cinco aspectos son limpieza, seguridad, accesibilidad, precio y atención. Ambos se equilibran por clase y presupuesto en las cuatro condiciones principales. Hay español general, estilización colombiana y peruana; su naturalidad y las anotaciones requieren todavía revisión independiente de hablantes.

Se registran negaciones, dobles negaciones lógicas, concordancia negativa, contrastes explícitos/implícitos, aspectos mixtos, regionalismos, errores ortográficos y palabras positivas en sentido negativo. Los relatos, diálogos, observaciones y longitudes varían. No se fuerzan los casos ambiguos a neutral. Una intención neutral significa ausencia de evaluación global polar clara, no desacuerdo automático entre aspectos. Las etiquetas y familias son decisiones de autor, no gold adjudicado.

Se prohíben duplicados exactos normalizados (NFKC, casefold y espacios) dentro de V3 y coincidencias exactas con cualquier fila V2. La pantalla léxica existente se reutiliza para cada par de roles V3: Jaccard de caracteres de cinco posiciones y secuencias de tres palabras, umbral 0.70. Un candidato bloquea la preparación. No se aprende vocabulario del assessment para este filtro. La identidad de familia/componente y el origen sintético se conservan en cada registro; los lotes de redacción por rol no son fuentes humanas independientes. **El mismo asistente redactó todos los roles: estos controles no prueban independencia semántica o externa.**

## Factorial de cantidad y diversidad

| Condición | Familias | Variantes por intención/familia | Filas |
| --- | ---: | ---: | ---: |
| `narrow_60` | 10 | 2 | 60 |
| `narrow_120` | Las mismas 10 | 4 | 120 |
| `broad_60` | 20 | 1 | 60 |
| `broad_120` | Las mismas 20 | 2 | 120 |

Las diez familias del pool estrecho se declaran antes de medir; hay dos por dominio y dos por aspecto. Los incrementos de cantidad son anidados por ID dentro de cada ancho. Cada condición tiene un tercio de cada clase. La comparación a presupuesto fijo cambia el número de contextos y la repetición de variantes relacionadas; no identifica un efecto causal universal de toda forma posible de diversidad. Solo hay un catálogo y una elección del pool, no réplicas independientes de generación o múltiples semillas de muestreo.

Se reutilizan Dummy, TF-IDF palabra+carácter con LR y SVM calibrado del código existente. La parrilla fija es C={0.5,2}; gana Macro-F1 de tune con desempate por orden de parrilla. El vectorizador se aprende únicamente en TRAIN, dentro de cada fold de calibración interna SVM. Los folds separan familias/componentes y duplicados exactos. No se reajusta sobre TRAIN+tune.

Se calculan contrastes de cantidad dentro de cada ancho, diversidad a 60/120 filas e interacción; se remuestrean 1000 veces las 15 familias de assessment con muestras pareadas. Son intervalos de sensibilidad de este pequeño conjunto sintético, no intervalos de generalización humana. Todos los modelos se informan; no hay promoción automática del de mayor puntuación.

## Aporte exploratorio de dimensiones

Para cada uno de los cinco dominios y cinco aspectos se omiten sus familias de TRAIN y se repone el presupuesto hasta 120 usando variantes de las familias restantes, conservando 40 intenciones de cada clase. LR/SVM se seleccionan y puntúan **solo con tune**. Se reporta el cambio de Macro-F1 general y en la rebanada omitida frente a `broad_120`, sus soportes, parámetros y familias.

Estas diez ablaciones miden contrastes de regímenes de entrenamiento en desarrollo. La reposición modifica las variantes y la composición de otras dimensiones; no son contribuciones causales aisladas ni se trasladan a turistas humanos. No se seleccionan combinaciones adicionales de dominios/aspectos ni se evalúan esos modelos de ablación sobre assessment. Los fenómenos y longitudes tienen sus rebanadas con soportes: no se presentan muestras pequeñas o de una sola clase como Macro-F1 comparable de tres clases.

## Calibración y revisión como estudio separado

Cada modelo elegido conserva sus predicciones de clase. Se ajusta una temperatura escalar sobre el logaritmo de `predict_proba` usando únicamente calibration; parrilla {1,0.5,0.75,1.5,2,3}, mínimo NLL y desempate fijo. Esta transformación preserva argmax, no corrige por sí misma errores de sentimiento y no modifica el clasificador de la DEMO. La calibración sigmoide interna de SVM continúa dentro de TRAIN; esta temperatura posterior usa un conjunto separado.

Se miden NLL, Brier multiclase (promedio de la suma de tres errores cuadrados, rango 0–2), ECE de confianza máxima y tablas de fiabilidad por clase con cinco bins fijos. El ajuste de NLL en calibration no es una evaluación independiente; se informa además en policy development y assessment. NLL/Brier combinan fiabilidad, discriminación e incertidumbre: no equivalen a una medida pura de calibración. Referencia: [documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).

Se comparan tres políticas sobre las mismas predicciones:

1. Política DEMO original con puntuaciones originales, sin cambios (0.56; contraste explícito bajo 0.80).
2. La misma regla usando probabilidades transformadas por temperatura.
3. Una regla experimental asimétrica: umbral global en {0.50,0.56,0.65,0.75,0.85,0.95,1.01}; para predicción positiva exige además {0.65,0.80,0.90,1.01}; margen entre las dos primeras probabilidades ≥0.10. 1.01 significa no aceptar esa categoría.

La selección de la tercera política ocurre solo en policy development y la cohorte ambigua de desarrollo. Elegibilidad: al menos nueve aceptados, cobertura ≥25%, riesgo empírico entre aceptados ≤10%, cero negativos→positivos aceptados y derivación ≥80% de los diez ambiguos previstos por el autor. Gana cobertura, luego menor riesgo, luego orden fijo. Si no hay candidato elegible, se fija **REVIEW_ALL**; con cero aceptados el riesgo condicional es `null`. Estos criterios son empíricos, **no garantías probabilísticas de seguridad**, y pueden no sostenerse en assessment.

En evaluación se reportan cobertura, error entre aceptados, revisiones, negativos→positivos automáticos sobre todos los negativos y sus IDs, más derivación de ambiguos sin exactitud de sentimiento. La curva de umbrales predefinida es descriptiva; no devuelve un umbral ganador de assessment. La opinión mixta, un contraste implícito o una cifra alta de confianza no son por sí solos evidencia de corrección. La revisión del sistema real permanece intacta.

## Ejecución, costos y protección

El runner verifica hashes del catálogo, protocolo y código relevante contra un commit publicado de registro; valida correspondencia exacta de registros e impide sobrescritura. Un diario distingue primera evaluación de reproducción congelada. Todos los C, temperaturas, políticas y ablaciones se escriben/hashan antes de una inferencia de assessment por modelo. Las métricas reutilizan esas predicciones. Los artefactos de modelos quedan en una ruta ignorada de investigación.

Se acota el filtro de archivos del workflow V2 a sus dos recursos exactos para que añadir recursos V3 no dispare otra evaluación V2. No se cambian el comando, los datos, los estimadores o las verificaciones de aquella reproducción congelada.

Se guardan versiones, commit de ejecución, hashes, tiempos de cada candidato/etapa, tiempo total de pared/CPU, pico de RSS del proceso Linux, tamaño de modelos y latencia cargada de lote 1 en policy development (cinco calentamientos, tres pasadas; p50/p95). El coste de temperatura/política se mide aparte. No se mide UI, Whisper, concurrencia o GPU; no se inventa coste monetario sin tarifa del equipo.

```bash
python -m pip install -r research/requirements-synthetic-v2.txt -e '.[test]'
python -m traveler_agent.research_v3_data --out data/processed/synthetic-v3-registered
python -m traveler_agent.research_v3_benchmark --data data/processed/synthetic-v3-registered --out artifacts/synthetic-v3/first --registration-commit SHA_DEL_REGISTRO_V3
```

Para repetir, usar un destino nuevo, conservar el registro y añadir `--reproduction`. La repetición se marca como tal, nunca como nueva evaluación ciega. El diario y los hashes no detectan toda ejecución manual en otra carpeta: sigue siendo necesaria disciplina de investigación. Las pruebas de contratos usan pequeños fixtures y no ajustan los clasificadores sobre assessment real.

## BETO, validación humana y aceptación

Se conserva la prueba técnica BETO de #21, con checkpoint y pesos fijados, sin convertir el encoder en clasificador. En esta fase se aíslan el efecto de representación clásica/cobertura y el de calibración/revisión. No se ejecuta ajuste contextual: permanecen pendientes el protocolo específico de pooling/cabeza, la medición de recursos de entrenamiento y el control independiente de calidad de etiquetas. El pico de inferencia CPU de #21 no es presupuesto de ajuste completo. No se usa UNED para entrenar BETO automáticamente.

La vía humana continúa según PILOTO-01 de #20: descarga autorizada de `travelerinsights2026/train.csv`, con reglas, privacidad, procedencia y mapeo real de etiquetas verificados, o recogida original consentida con dos anotaciones independientes y adjudicación. No se accede al test oculto ni se solicita contraseña. Ningún corpus humano está admitido en V3; sus resultados no demuestran mayor precisión en reseñas reales.

Criterios de aceptación: factorial equilibrado y anidado; roles/componentes sin fugas; V2 intacto; filtros de duplicados; trazabilidad sintética; selección solo en desarrollo; temperatura/política separadas; resultados completos aunque empeoren; riesgos con denominadores y `null` correcto; costos medidos; pruebas y reproducción CI; ningún cambio en agentes, Gradio, ASR, lotes, exportaciones o modelo DEMO. BM25, Sentence-BERT y RAG siguen en la hoja de ruta posterior. El responsable revisará y decidirá manualmente cualquier fusión.
