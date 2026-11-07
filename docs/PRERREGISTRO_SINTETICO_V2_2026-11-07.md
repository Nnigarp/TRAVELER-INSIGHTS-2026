# Investigación sintética V2 — registro previo a la primera evaluación

Base verificada: `main` en `cb3cb27fe43c4dc298c387f3e067c1f3f779a007` (PR #20). Este incremento es una línea experimental independiente. No admite datos humanos, no modifica la DEMO, sus cinco agentes, UI, Whisper, revisión humana ni exportaciones. El primer commit de esta rama fija catálogo y protocolo **antes de ejecutar los modelos sobre el nuevo desafío**. El informe posterior identifica ese SHA.

## Diagnóstico y alcance de la hipótesis

El generador histórico combina 56 familias de establecimiento/aspecto, tres clases y dos variantes (336 textos); el split original conserva 204 filas de TRAIN. La separación por familia evita compartir el escenario completo, pero varias familias reutilizan cláusulas de aspecto, tres ordenamientos de discurso y cierres/negaciones/contrastes asociados a una clase. Un TF-IDF puede aprender esas pistas sin resolver el alcance de una negación o el peso de aspectos contradictorios.

El desafío anterior ya mostró Macro-F1 de 0.3926 para LR y 0.3882 para SVM, y 232/420 y 236/420 negativos clasificados como positivos, respectivamente, según `docs/GENERALIZACION_FUERA_GENERADOR_2026.md`. Es evidencia histórica, **no una nueva medición**. Los tests históricos y su revisión influyeron en las hipótesis; no se presentan como un test final intacto ni se puntúan aquí para seleccionar modelos. La repetición de frases apoya una hipótesis de atajos léxicos; este experimento no demuestra causalmente que explique toda la caída. Puede coexistir ambigüedad de intención y cambio de distribución.

Se reutilizan el vectorizador palabra+carácter, Dummy, LR, SVM y la calibración por grupos del evaluador existente. No se cambian los modelos de aplicación ni sus artefactos. La reconstrucción histórica materializa el split original en memoria y retorna exclusivamente TRAIN, sin invocar el evaluador o escritor histórico.

## Corpus y trazabilidad

El recurso `src/traveler_agent/research_data/synthetic_diversity_v2.json` contiene 180 textos completos redactados individualmente, agrupados en 60 familias semánticas declaradas. No es un sistema de sustitución de sustantivos/adjetivos. Los escenarios varían en el problema central, tipo de evidencia, orden temporal, voz, extensión y relación entre aspectos; se incluyen alojamientos, recorridos, museos, gastronomía y experiencias naturales/culturales. Cada texto conserva intención del autor, explicación, discurso, fenómenos y región estilística declarada.

| Rol | Familias | Textos | Por clase |
| --- | ---: | ---: | ---: |
| Entrenamiento | 30 | 90 | 30 |
| Desarrollo común | 12 | 36 | 12 |
| Desafío nuevo | 18 | 54 | 18 |
| Ambigüedad, sin sentimiento gold | 9 | 9 | — |
| Ejemplos ilustrativos del usuario | 4 | 4 | — |

Cada rol etiquetado equilibra `es`, `es-CO` y `es-PE` por clase. Los regionalismos son una apreciación del autor; falta revisión de hablantes nativos. El catálogo incluye negaciones, dobles negaciones lógicas, concordancia negativa, contrastes, aspectos mixtos, palabras positivas con evaluación negativa, errores ortográficos y textos breves/extensos. Las etiquetas de fenómenos también son juicios del autor.

**No existen autores humanos, reseñas reales ni etiquetas humanas gold en este recurso.** La procedencia declara redacción asistida por Codex, modelo exacto no identificado de manera fiable y parámetros de muestreo no disponibles. La reproducción consiste en materializar el catálogo literal fijado, no en prometer que una semilla reproduce la redacción de un LLM. La semilla 42 controla únicamente selección y modelos. Los cuatro ejemplos del usuario nunca reciben una etiqueta gold ni entran en ajuste o Macro-F1. Los nueve ambiguos miden derivación a revisión, sin exactitud de sentimiento.

Las familias se asignan a roles antes de medir. Todos los textos se normalizan con NFKC, casefold y espacios para impedir duplicados exactos. Se calcula Jaccard de caracteres de longitud 5 y secuencias de tres palabras entre roles nuevos, sin aprender vocabulario sobre el desafío; un candidato ≥0.70 impide preparar el corpus. Se comprueba además que ningún texto de TRAIN histórico/nuevo coincida exactamente con roles nuevos reservados. Este filtro no detecta toda paráfrasis ni prueba independencia semántica. **El mismo asistente redactó los tres roles: el desafío no es una evaluación externa ciega o humana.**

## Comparación fijada

| Estrategia | Filas de entrenamiento | Uso |
| --- | ---: | --- |
| A histórica, presupuesto igualado | 90 de TRAIN histórico | Comparación principal |
| B nueva | 90 de TRAIN nuevo | Comparación principal |
| C combinación controlada | 45 históricas + 45 nuevas | Comparación principal |
| A histórica completa | 204 de TRAIN histórico | Sensibilidad al tamaño; informar por separado |
| D humana + sintética | Pendiente | Requiere corpus humano autorizado y admitido |

La selección de familias utiliza hash estable; las filas históricas igualadas eligen una variante `narrative_0` por clase/familia. A se vuelve a ajustar y selecciona sus hiperparámetros con el nuevo desarrollo común: no es una evaluación de los pesos históricos guardados ni una reproducción de su antigua puntuación.

Dummy no tiene búsqueda; LR/SVM prueban exclusivamente `C ∈ {0.5, 2.0}`. Gana Macro-F1 de desarrollo, con desempate por orden fijo. SVM usa tres folds disjuntos por grupos de TRAIN y vectorizadores ajustados dentro de cada fold. Las cuatro estrategias terminan toda la selección y escriben/hashan `selection_manifest.json` antes de una única inferencia de desafío por estimador seleccionado. Las mismas predicciones alimentan todas las métricas posteriores. No se revisarán datos, parámetros o umbrales para mejorar este desafío tras abrirlo.

El catálogo, protocolo y registros tienen SHA-256. El benchmark exige un commit de registro que contenga el catálogo/protocolo exactos. Un diario impide una segunda primera evaluación sobre el mismo directorio; `--reproduction` permite exclusivamente repetir la especificación congelada y se identifica como reproducción, no como una nueva prueba ciega. Este control no sustituye la disciplina de investigación: no detecta evaluaciones manuales en otro directorio.

## Métricas, riesgos y criterios

Se reportan Macro-F1 de las tres clases, recall/F1/support por clase, matriz de confusión, negativos→positivos y negativos→neutrales con denominador explícito. Las rebanadas por fenómeno, región y longitud incluyen soportes y advierten si falta una clase. Los intervalos remuestrean las 18 familias 1000 veces, con semillas y muestras pareadas para diferencias B−A y C−A; expresan sensibilidad del pequeño catálogo sintético, no inferencia sobre turistas humanos.

La política de revisión se mide sin cambios: confianza máxima <0.56, o contraste detectado (`pero`, `aunque`, `sin embargo`) y confianza <0.80. Se informa cobertura, tasa de revisión, error entre aceptados y negativos→positivos aceptados automáticamente sobre todos los negativos. Si no se acepta nada, el riesgo condicional es `null`, nunca cero. La fiabilidad probabilística de LR/SVM sigue sin validación humana; SVM calibrado no implica calibración externa demostrada.

La latencia mide desarrollo nuevo, modelos cargados, lote de un texto, vectorización incluida, cinco llamadas de calentamiento y diez pasadas. Se reportan mediana/p95 y versiones del entorno; no es carga real ni la latencia de toda la aplicación.

Criterios de aceptación: materialización determinista; procedencia inequívoca; separación por familia y duplicados verificada; ejemplos sin gold excluidos del ajuste; presupuesto A/B/C de 90 filas y clases equilibradas; selección antes del desafío; resultados y fallos conservados sin exigir una mejora; pruebas de contratos; reproducción en CI; DEMO e históricos sin cambios. Un resultado peor sigue siendo un resultado admisible y debe informarse.

## Ejecución y BETO

Desde la raíz del repositorio, con Python 3.12:

```bash
python -m pip install -e '.[test]'
python -m traveler_agent.synthetic_diversity --out data/processed/synthetic-diversity-v2-registered
python -m traveler_agent.synthetic_diversity_benchmark --data data/processed/synthetic-diversity-v2-registered --out artifacts/synthetic-diversity-v2/first --registration-commit SHA_DEL_PRIMER_COMMIT
```

Usar rutas nuevas y no sobrescribir evidencia. Para reproducir tras la primera apertura, mantener el mismo commit/catálogo/protocolo y añadir `--reproduction`, con otro destino de resultados. La ejecución del test suite verifica software; no crea nuevas conclusiones científicas.

BETO se prueba en otro entorno, sin instalar Torch/Transformers en la DEMO:

```bash
python -m venv .venv-research
.venv-research/bin/python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu
.venv-research/bin/python -m pip install -e '.[encoder-research]'
.venv-research/bin/python -m traveler_agent.beto_encoder_probe --download --output artifacts/beto-encoder-smoke-v1/report-new.json
```

En Windows, sustituir `.venv-research/bin/python` por `.venv-research/Scripts/python.exe`. La descarga es explícita; posteriores ejecuciones usan la caché verificada sin red y requieren otro archivo de salida. Se fija `dccuchile/bert-base-spanish-wwm-cased` en `c4d86612f51b4f46759c8390d1798c2febe71b93`, pesos SHA-256 `e131a95091c777bbd45250fb647fec415010cb8cd1ad6e1d59babeb82a0be360`, Transformers 4.57.6, Torch 2.9.1 y psutil 7.2.2. Carga local, `trust_remote_code=False`, `weights_only=True`; no se publican pesos.

La [ficha oficial fijada](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/blob/c4d86612f51b4f46759c8390d1798c2febe71b93/README.md) expresa intención CC BY 4.0 y reconoce incertidumbre sobre compatibilidad de los textos de preentrenamiento, en particular para uso comercial. La prueba técnica académica no equivale a una autorización integral sobre esos textos.

El probe comprueba tokenización, truncamiento, salida finita, dimensiones, tiempos y memoria de inferencia CPU. Un encoder sin ajuste supervisado no produce tres sentimientos. Una futura cabeza lineal de 768→3 sumaría 2307 parámetros y requeriría entrenamiento/selección supervisados y evaluación independiente. GPU, ajuste completo y sus recursos reales permanecen pendientes. El mínimo teórico de parámetros+gradientes+momentos de Adam no es una recomendación de RAM total.

## Línea humana y hoja de ruta

PILOTO-01 humano continúa según `docs/PILOTO01_ACCESO_Y_BENCHMARK_HUMANO_2026-11-06.md`: acceso autorizado a `travelerinsights2026/train.csv` o alternativa con derechos/procedencia/etiquetas auditados. No se accede al test oculto ni se supone un mapeo de etiquetas. UNED permanece como evidencia de auditoría, sin entrenamiento BETO automático. Sin datos humanos admitidos no se medirá D ni se afirmará robustez humana. BM25, Sentence-BERT y RAG conservan su posición posterior en la hoja de ruta.
