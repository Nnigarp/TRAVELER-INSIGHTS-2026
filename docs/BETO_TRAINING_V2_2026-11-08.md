# BETO Training V2: entrenamiento real, evaluación independiente y demostración

Estado del 8 de noviembre de 2026; preparación para el lunes **9 de noviembre**. Investigación aislada, dependiente del adaptador opcional del PR #24. No cambia `main`, no fusiona PRs ni promueve BETO como modelo predeterminado. Los cinco agentes, Midnight Indigo, Whisper, transcripción editable, lotes, revisión humana, exportaciones y umbrales históricos conservan su comportamiento.

## Qué problema aborda

El diagnóstico de las predicciones congeladas de Contextual V1 identificó negativos→positivos cuando elogios locales o irónicos coexistían con un resultado insatisfactorio: promesa incumplida de un taller, desayuno ya pagado y no servido, pérdida de una experiencia con fauna. BETO congelado tuvo tres de esos errores; el ajuste parcial tuvo cuatro, dos aceptados por las reglas históricas. Son observaciones de ese experimento, no una identificación causal de pooling o de una palabra concreta. Su doble negación tenía soporte de un solo ejemplo: no permite estimar rendimiento de tres clases en ese fenómeno. [Diagnóstico con IDs y textos](research/beto-v2-prior-error-diagnosis.json).

Este incremento prueba una hipótesis distinta: mejorar composición y contexto con entrenamiento supervisado por etapas, comparar fuentes/cantidad controladas y estudiar calibración por separado. Se implementó un ajuste real de los dos últimos bloques y fine-tuning completo de BETO; todos los candidatos se seleccionan únicamente en desarrollo nuevo. El catálogo, protocolo y elecciones anteriores al assessment quedaron guardados en GitHub antes de evaluarlo.

## Datos, procedencia y protección del experimento

| Papel | Textos | Familias/componentes | Uso |
| --- | ---: | ---: | --- |
| TRAIN histórico | 120 | 20 | Solo variantes 0/1 de TRAIN V3; 40 por clase |
| TRAIN nuevo | 120 | 40 | Textos originales sintéticos; 40 por clase |
| Mezcla TRAIN | 240 | 60 | Fuentes identificadas; 80 por clase |
| Tune nuevo | 30 | 10 | C, representación, configuración y épocas |
| Calibración nueva | 60 | 20 | Temperatura por NLL |
| Desarrollo de política | 60 | 20 | Cobertura/riesgo y regla de revisión |
| Assessment nuevo | 90 | 30 | Comparación común; 30 por clase |
| Ambiguos de política / assessment | 8 / 8 | Disjuntos | Sin etiqueta de sentimiento; evaluación de derivación |

Son **376 textos nuevos sintéticos**, además de los 120 TRAIN históricos. Las etiquetas son intención de redacción del asistente, no anotación ni gold humano. Mezcla de aspectos es un fenómeno, no se convierte automáticamente en neutralidad. Los ambiguos no entrenan la clasificación de tres sentimientos.

Los nuevos textos cubren hoteles, restaurantes, museos, transporte y recorridos; limpieza, seguridad, accesibilidad, precio y atención. Incorporan narración de incidentes y resoluciones, información objetiva, expectativas incumplidas, contrastes explícitos/implícitos, negación, doble negación, opiniones mixtas, variación colombiana/peruana, errores y longitudes diferentes. No se genera volumen sustituyendo sustantivos en una plantilla. TRAIN tiene 6–104 palabras y assessment 28–108; no representan todavía reseñas extensas de cientos de tokens ni la distribución de una población turística.

TRAIN y desarrollo se redactaron con conocimiento del diagnóstico histórico, declarado en la procedencia. Assessment se redactó separadamente, sin acceso a los textos de TRAIN/desarrollo ni a sus resultados. Esa separación de redacción no demuestra independencia poblacional: comparten autoría artificial. Se controlan familias/componentes y duplicados exactos/cercanos en los 21 pares de roles, con Jaccard de caracteres/palabras ≥0.7. Los conjuntos históricos evaluados solo se consultan como lista de hashes para excluir copias, no para elegir técnicas o umbrales.

El diagnóstico OOF por familias de TRAIN detecta una Macro-F1 de **0.5741 usando solo forma/longitud/puntuación**. Fuente/dominio/región por sí solos producen 0.1667 y no hay cierres repetidos de cuatro palabras. Persisten correlaciones de estilo: no se afirma que los textos estén libres de atajos ni que cada dimensión haya demostrado una aportación causal. La comparación histórico/nuevo confunde diversidad y fuente; mezclar 240 también cambia cantidad y componentes.

- Registro previo: [`7dd51a7c689873b5e3b000e482bfd2bc481adc3e`](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/commit/7dd51a7c689873b5e3b000e482bfd2bc481adc3e).
- Selección completa guardada antes de assessment: [`d067c8aad31cf23a12ea2cd76f78c2f71f4cdcca`](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/commit/d067c8aad31cf23a12ea2cd76f78c2f71f4cdcca).
- Registros combinados SHA-256: `908850b47a21811df447cda83ab9d36fb75eab771ad4ed22270dee4459239766`.
- [Registro](research/beto-v2-registration.json), [protocolo](../src/traveler_agent/research_data/beto_training_v2_protocol.json), [diagnóstico TRAIN](research/beto-v2-training-diagnostics.json), [selección inmutable inicial](research/beto-v2-development-selection.json).

V2, V3 y Contextual V1 permanecen congelados. El assessment de este incremento **ya está observado**; tampoco puede anunciarse como test final intacto para seleccionar configuraciones futuras. La reproducción de CI comprueba software, no aporta otra muestra independiente.

## Entrenamiento y selección

Checkpoint `dccuchile/bert-base-spanish-wwm-cased`, revisión `c4d86612f51b4f46759c8390d1798c2febe71b93`; SHA-256 del peso base `e131a95091c777bbd45250fb647fec415010cb8cd1ad6e1d59babeb82a0be360`. Python 3.12, torch 2.9.1 CPU, transformers 4.57.6, dos hilos; dependencias científicas en `research/requirements-contextual-v1.txt`. No se necesita GPU para estos tamaños. El cargador verifica hashes, usa `weights_only=True`, rechaza código remoto y carga adaptaciones mediante safetensors; la predicción no descarga pesos.

| Candidato | Encoder | Pooling / tokens | LR encoder / cabeza | Épocas previstas |
| --- | --- | --- | --- | ---: |
| frozen_mean128 | Congelado | Media L2 / 128 | Solo cabeza | 0 |
| frozen_cls128 | Congelado | CLS L2 / 128 | Solo cabeza | 0 |
| top2_low128 | Bloques 10/11 | Media L2 / 128 | 1e-5 / 5e-4 | 4 |
| top2_high128 | Bloques 10/11 | Media L2 / 128 | 3e-5 / 5e-4 | 4 |
| full_low128 | Completo | CLS L2 / 128 | 1e-5 / 2e-4 | 3 |
| full_high128 | Completo | CLS L2 / 128 | 3e-5 / 2e-4 | 3 |
| full_long256 | Completo | CLS L2 / 256 | 1e-5 / 2e-4 | 3 |
| full_regularized128 | Completo | CLS L2 / 128 | 1e-5 / 2e-4 | 3 |

Cabeza de 2307 parámetros: AdamW, LR .01, dropout .1, weight decay .01; selección entre épocas 20/40/80. Fine-tuning con batch 4, warmup 10 %, decaimiento lineal, clip de gradiente 1.0, dropout activo en bloques entrenables. Variante regularizada: weight decay .1, dropout de cabeza .3 y label smoothing .05. Clases equilibradas por diseño, sin sobremuestreo; LR/SVM usan pesos equilibrados. Una sola pasada por esta rejilla no agota las alternativas de entrenamiento.

Se eligió **top2_low128**, cabeza época **80**, checkpoint de encoder época **1 de un plan de 4**. La réplica conserva el plan original de LR de cuatro épocas al detenerse en la primera; no reinicia un scheduler de una época. Se entrenan 14 175 744 parámetros del encoder + 2307 de cabeza. Gradiente inicial 1.31737 y cambio L2 del tensor de prueba 0.0647444 verifican actualización real; parámetros congelados sin gradientes. Las pruebas automatizadas verifican ambos alcances y la reproducción de los pesos con el plan de LR original.

Selección por Macro-F1 en tune; desempate por menos negativos→positivos y menor NLL. Semilla primaria **42**, fijada antes de evaluar. Réplicas **7 y 101** usan configuración y épocas elegidas, sin seleccionar la mejor semilla. El máximo TRAIN observado es 128 tokens y el máximo tune 46: la opción 256 no aportó contexto adicional en estas filas. No permite concluir que 128 sea suficiente para reseñas humanas largas. La [auditoría posterior de tokenización](research/beto-v2-sequence-length-audit.json) encuentra dos de 90 textos de assessment por encima de 128 (máximo 133): esos dos sufren truncamiento. No se ejecuta ahora el modelo de 256 sobre assessment ni se cambia la longitud elegida.

Dummy, TF-IDF+LR y TF-IDF+SVM usan exactamente el mismo assessment. C se elige entre .1/1/10 solo en tune. SVM calibra internamente con sigmoid en tres folds agrupados de TRAIN; TF-IDF se ajusta dentro de cada fold. No se incluyen IDs, fenómenos o metadatos como características de los clasificadores de sentimiento.

## Resultados exclusivamente sintéticos

Los resultados siguientes se observaron en la primera evaluación. La restauración del entorno reconstruye los artefactos con elecciones fijadas; el acta de reproducción enlazada abajo distingue esa ejecución de la original.

| Modelo, TRAIN mezcla 240 | Macro-F1 | Recall negativo | Negativo→positivo / 30 |
| --- | ---: | ---: | ---: |
| Dummy | 0.1667 | 1.0000 | 0 |
| TF-IDF + LR | 0.8461 | 0.8333 | 2 |
| TF-IDF + SVM | 0.8585 | 0.9000 | 2 |
| BETO congelado + cabeza | 0.8642 | 0.7333 | 5 |
| BETO dos bloques, semilla 42 | 0.8454 | 0.8667 | 1 |

| Entrenamiento | LR | SVM | BETO congelado |
| --- | ---: | ---: | ---: |
| Histórico 120 | 0.7941 | 0.7731 | 0.7929 |
| Nuevo 120 | 0.8447 | 0.7795 | 0.8870 |
| Mezcla 240 | 0.8461 | 0.8585 | 0.8642 |

Más datos no mejora todas las condiciones. BETO nuevo 120 supera a su mezcla en Macro-F1, pero ambos tienen cinco negativos→positivos. No se escoge ese brazo a posteriori para sustituir el candidato principal ni se extrapola a personas.

Las semillas 42/7/101 del ajuste elegido alcanzan Macro-F1 **0.8454 / 0.8962 / 0.8488** con **1 / 5 / 8** negativos→positivos. Media 0.8635, desviación estándar muestral 0.0284: la sensibilidad de los errores críticos impide afirmar robustez. El bootstrap pareado por 30 familias para BETO primario menos LR mezcla abarca aproximadamente **−0.0966 a +0.0978**. Es descriptivo de textos sintéticos, no un intervalo poblacional humano ni evidencia de superioridad.

Recall por clase (negativo / neutral / positivo): LR mezcla **0.8333 / 0.9000 / 0.8000**; SVM mezcla **0.9000 / 0.8667 / 0.8000**; BETO primario **0.8667 / 0.9333 / 0.7333**. En los 50 casos con negación y tres clases presentes, Macro-F1 es **0.8527 / 0.8985 / 0.8820**, respectivamente; los negativos→positivos son **1 / 1 / 0**. Es una descripción sintética condicionada al soporte, no prueba causal de comprensión.

Matriz primaria, orden negativo/neutral/positivo, filas intención y columnas predicción: `[[26,3,1],[2,28,0],[8,0,22]]`. El error negativo→positivo es el taxi cuyo conductor cobra deprisa y no devuelve el cambio (`b2_e9ae66de2f27d94efea7`). El elogio local no representa la evaluación global. También hay ocho positivos→negativos, incluidas resoluciones satisfactorias con vocabulario de incidentes o negación; reducir una dirección de error no resuelve la composición lingüística completa.

Soportes lingüísticos del assessment: negación **50** (20 negativos, 4 neutrales, 26 positivos), contraste **38** (25 negativos, 13 positivos), doble negación **6** (4 negativos, 2 positivos) y aspectos mixtos **4** (3 negativos, 1 positivo). Las tres últimas rebanadas no tienen las tres clases: su Macro-F1 de tres clases no es comparable con la global y el JSON lo marca. No se acredita robustez en opiniones mixtas con cuatro ejemplos. Tune no tiene ejemplos marcados como mixtos o doble negación; la elección fue global y no una optimización validada de esos fenómenos. Se conserva esa limitación sin rediseñar los textos después de conocer resultados.

[Informe, matrices y métricas por clase](research/beto-v2-assessment.json) · [1170 predicciones reales](research/beto-v2-predictions.json) · [Análisis pareado/semillas](research/beto-v2-analysis.json) · [Figura](research/beto-v2-comparison.png).

## Calibración y abstención: estudio separado

Temperatura seleccionada por NLL únicamente en 60 ejemplos de calibración; ECE con cinco bins, Brier y NLL guardados. Doce políticas preespecificadas combinan umbrales globales .6/.7/.8, positivos .85/.95, margen .1 y guardia textual de negación/contraste activada o no. La guardia usa texto observado y predicción, no las etiquetas del fenómeno.

Desarrollo exige ≥20 aceptados, ≥5 negativos aceptados, ≥20 negativos totales, ≥10 componentes aceptados, cobertura ≥.25, riesgo empírico ≤.10, negativos→positivos automáticos / todos los negativos ≤.05 y derivación de ambiguos ≥.80. Estos son criterios exploratorios de admisión, no garantía estadística de seguridad.

La regla histórica, conservada sin ajustes, acepta **80/90** predicciones del BETO primario, con **ocho errores**, incluido **un negativo→positivo automático**. Mayor cobertura/confianza no acredita seguridad. Esa regla no se transforma en política validada por la aplicación experimental.

El BETO primario no admite política en desarrollo. No se presenta «revisar todo» como solución. En evaluación, las candidatas admitidas de SVM mezcla y BETO congelado nuevo solo derivan **5/8 ambiguos**, por debajo del criterio. SVM acepta 49/90 con tres errores; BETO congelado nuevo 45/90 con tres errores y un negativo→positivo automático. **Ninguna política satisface todos los criterios independientes**. No se traslada ningún umbral a la DEMO. [Selección y resultados de calibración/política](research/beto-v2-calibration-policy.json).

## Recursos, recuperación y evidencia

La búsqueda original ejecutó los ocho candidatos: **988.76 s** de pared y pico de proceso **2945.37 MiB**, CPU/dos hilos. Assessment con réplicas: **263.75 s**, sin incluir instalación, descarga, redacción o un intento inicial incompleto cuyo costo/causa no quedó registrado de forma fiable. No se atribuye ese intento a OOM ni se oculta en el costo. El costo monetario y GPU no se midieron.

La sesión local se restauró a una instantánea anterior el 8 de noviembre. Datos, código, historial de entrenamiento y elecciones previas a la evaluación permanecieron en GitHub (`7dd51a7`, `d067c8a`). Se vuelve a ejecutar la rejilla original, sin nuevos datos/configuraciones, y se exige coincidencia de todos los C, épocas de cabeza, candidato y semilla antes de repetir assessment. Las métricas originales se conocían, pero las probabilidades originales no publicadas se perdieron: **no se inventaron ni se afirma una comparación bit a bit con esos arrays perdidos**. Los hashes de los once manifiestos/pipelines principales y las seis adaptaciones del encoder coinciden exactamente con los registrados antes del assessment. Las predicciones publicadas corresponden a la reconstrucción real y sirven como referencia explícita de CI; la coincidencia con la primera ejecución solo puede comprobarse sobre los agregados conservados. [Acta de reproducción](research/beto-v2-reproduction.json).

La primera instrumentación lanzó hijos desde un proceso con optimizador cargado; Linux les heredó el pico y dio 1161.70 MiB incluso al Dummy. Esas cifras se descartaron como comparación entre modelos. `scripts/measure_beto_v2_memory.py` mide en procesos nuevos lanzados desde un padre que importa solo stdlib; incluye imports, carga y nueve inferencias tune, sin repetir assessment. Rechaza una medición idéntica para todos. [Memoria corregida de inferencia](research/beto-v2-inference-memory.json). Latencia mide ocho solicitudes tune con batch 1 y modelo ya cargado; no incluye arranque ni ASR. Son mediciones del host indicado, no garantías del portátil del usuario.

[Gradio HTTP local con el artefacto V2](research/beto-v2-http.json): frontend/configuración 200; LR, SVM y BETO ejecutados efectivamente, diez eventos de traza por análisis y ninguna violación del contrato. No acredita exactitud semántica, voz real ni navegador Windows.

Validación local de la rama: **294 passed, 1 xfailed** en 68.98 s. Se combinaron solo fuentes rastreadas de PR #23 (`f18a6b0`) y de este incremento en un directorio ignorado, sin fusionar ramas ni copiar datos privados: **324 passed, 1 xfailed** en 71.54 s. El xfail estricto histórico `test_negation_expected_positive_is_known_limitation` se conserva. Las 18 pruebas nuevas (18/18 aprobadas en ejecución enfocada) cubren gradientes, alcance de bloques, scheduler, procedencia/roles, drift de selección y telemetría inválida.

La reconstrucción medida tarda **1361.15 s** en desarrollo (pico **2913.79 MiB**) y **236.54 s** en assessment/réplicas. La medición limpia de inferencia da **159.53 MiB LR**, **166.21 MiB SVM** y **779.67 MiB BETO primario**. Sus latencias p50/p95 observadas son **1.82/3.64 ms**, **7.63/8.80 ms** y **91.56/153.48 ms**, respectivamente, sobre ocho solicitudes warm por modelo; ese tamaño no estima una cola de producción ni garantiza latencia en otro equipo.

El [aviso original del checkpoint](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/blob/c4d86612f51b4f46759c8390d1798c2febe71b93/README.md) declara intención CC BY 4.0 y conserva incertidumbre sobre derechos de sus corpus de preentrenamiento. No se publican pesos, cachés, corpus privados, secretos o consentimientos.

## Windows: ruta corta para la presentación

Desde un clon con la rama del incremento y sin cambiar `main`; ejecutar antes del día de la presentación para dejar el checkpoint local preparado. Usar directorios nuevos si ya existen resultados.

```powershell
git fetch origin
git switch feat/beto-training-v2-20261107
py -3.12 -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-research\Scripts\python.exe -m pip install -r research/requirements-contextual-v1.txt -e '.[ui,test]'
.\.venv-research\Scripts\python.exe -m traveler_agent.beto_training_v2 register --output artifacts/presentacion-registro-v2
.\.venv-research\Scripts\python.exe -m traveler_agent.beto_training_v2 replay --download --registration artifacts/presentacion-registro-v2 --registration-commit 7dd51a7c689873b5e3b000e482bfd2bc481adc3e --selection docs/research/beto-v2-selected-configuration.json --seed 42 --output artifacts/presentacion-beto-v2
$env:TRAVELER_BETO_ARTIFACT = 'artifacts/presentacion-beto-v2/model'
.\.venv-research\Scripts\python.exe scripts/smoke_gradio_http.py --beto-artifact $env:TRAVELER_BETO_ARTIFACT --out artifacts/presentacion-beto-v2/http.json
.\.venv-research\Scripts\python.exe app.py
```

Seleccionar BETO manualmente en Gradio. La aplicación inicia con LR. Un error al cargar/ejecutar BETO activa respaldo LR explícito y trazable, sin adjudicar esa predicción a BETO. La comparación en la UI ilustra funcionamiento: los modelos clásicos allí usan DEMO, por lo que no es el benchmark científico común de este documento. Para volver a la configuración clásica, quitar la variable y reiniciar.

Para reproducir el experimento completo: `register`, `develop --download --registration-commit 7dd51a7c689873b5e3b000e482bfd2bc481adc3e`, verificador de selección, `assess` con el mismo commit, `beto_training_v2_analysis`, verificador de predicciones y medidor de memoria. La receta exacta está en [CI CPU/Windows](../.github/workflows/beto-training-v2.yml). La figura se reconstruye desde el informe con `scripts/plot_beto_v2.py` y matplotlib; no interviene en selección.

El job Windows entrena de verdad el candidato fijo y llama al servidor HTTP/cola Gradio. Esa evidencia no acredita micrófono, navegador ni voz real en el equipo del responsable. Para el lunes aún hay que ensayar allí la transcripción propia autorizada, editarla, comparar texto/lotes, probar corrección humana y descargar JSON/CSV. La [guía anterior](AVANCE_EJECUTABLE_BETO_Y_DEMO_2026-11-07.md) contiene el comando de audio real; no se publica la grabación. No se acredita un porcentaje académico sin rúbrica.

## Datos humanos y decisión siguiente

No hay corpus humano admitido ni resultados humanos en esta fase. La vía de aportaciones consentidas del PR #23 sigue disponible y requiere integración manual independiente. Reutiliza formularios privados, códigos seudónimos, dos anotaciones ciegas y tercera persona para adjudicar. El JSON de prueba del formulario no es una reseña admitida. [Fuentes revisadas y pasos del responsable](DATOS_HUMANOS_PARA_BETO_V2_2026-11-07.md).

Kaggle requiere el acceso personal del responsable, reglas aceptadas, solo `train.csv`, procedencia/derechos y diccionario auténtico de etiquetas; no contraseña, tokens, cookies ni test oculto. Se verificó además el depósito Zenodo citado por la competencia, con metadatos de acceso abierto/CC BY 4.0 y procedencia de plataformas, pero sin gold global ni permisos textuales suficientemente aclarados para admitirlo. No se descargó o entrenó con ese XLSX. UNED queda como auditoría.

La siguiente mejora de datos debe cubrir con personas independientes la distinción entre incidente resuelto, elogio local y juicio global, además de ambigüedad. No se genera otro corpus masivo ni se ajusta ahora el entrenamiento a los errores de este assessment. Las decisiones futuras requieren desarrollo/evaluación nuevos y derechos/procedencia resueltos. BM25, Sentence-BERT y RAG permanecen posteriores y no sustituyen la clasificación.

## Criterios de aceptación del incremento

- Datos/protocolo y selección fijados antes del assessment; roles y componentes sin fuga y datasets históricos congelados.
- Entrenamiento real verificado por gradientes/cambio de pesos y réplicas con scheduler original.
- Misma evaluación para todos; métricas por clase, matrices, errores, latencia, memoria y límites publicados.
- CI reproduce los resultados sintéticos con tolerancias numéricas preespecificadas, verifica que no cambien las elecciones y ejecuta el candidato/HTTP en Windows.
- Suite existente en verde, xfail histórico de negación conservado; fallback y selección opcional probados. Sin sustitución automática de DEMO ni política admitida para producción.
