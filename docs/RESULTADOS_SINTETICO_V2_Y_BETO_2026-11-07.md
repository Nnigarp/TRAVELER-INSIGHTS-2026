# Resultados registrados: diversidad sintética V2 y encoder BETO

La comparación se ejecutó después de publicar el catálogo y protocolo en el commit [`e3420606c721622d4c9134b0a1c82cef78982744`](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/commit/e3420606c721622d4c9134b0a1c82cef78982744). La base es `main@cb3cb27fe43c4dc298c387f3e067c1f3f779a007`. **Todos los resultados de sentimiento son sintéticos y usan intención del autor, no gold humano.** No se incorpora ningún modelo a la DEMO.

El [prerregistro](PRERREGISTRO_SINTETICO_V2_2026-11-07.md) detalla hipótesis, controles, presupuestos y limitaciones. El catálogo/protocolo permanecieron intactos después de abrir el desafío. Se reutilizaron los estimadores y la calibración por grupos existentes. La selección completa precedió a la inferencia de desafío y quedó en un manifiesto independiente.

## Comparación principal y sensibilidad al tamaño

Desafío: 54 textos, 18 familias, 18 intenciones de cada clase. Desarrollo común: 36 textos de otras 12 familias. A/B/C: 90 filas y 30 intenciones por clase; C conserva 45 históricas y 45 nuevas. La fila histórica completa tiene 204 filas y se interpreta por separado.

| Entrenamiento | Dummy Macro-F1 | LR Macro-F1 | SVM Macro-F1 | LR negativos→positivos | SVM negativos→positivos |
| --- | ---: | ---: | ---: | ---: | ---: |
| A histórico igualado, 90 | 0.1667 | 0.4247 | 0.4657 | 6/18 | 10/18 |
| B nuevo, 90 | 0.1667 | 0.7729 | 0.7751 | 5/18 | 5/18 |
| C combinación 45+45 | 0.1667 | 0.7165 | 0.7165 | 6/18 | 6/18 |
| A histórico completo, 204; sensibilidad | 0.1667 | 0.4792 | 0.5550 | 9/18 | 9/18 |

No se puntúan los antiguos tests. A se reajustó usando exclusivamente TRAIN histórico y selección en el nuevo desarrollo: no corresponde a los pesos históricos guardados ni a una reproducción de su antiguo resultado. La comparación igualada elige una variante por clase/familia, por lo que A90 y A204 difieren también en diversidad de variantes; no aísla causalmente el tamaño.

| Modelo / estrategia | C elegido solo en desarrollo | Recall negativo | Recall neutral | Recall positivo |
| --- | ---: | ---: | ---: | ---: |
| LR A90 | 0.5 | 0.3333 | 0.4444 | 0.5000 |
| LR B90 | 0.5 | 0.6667 | 1.0000 | 0.6667 |
| LR C90 | 2.0 | 0.5556 | 0.9444 | 0.6667 |
| SVM A90 | 0.5 | 0.3333 | 0.4444 | 0.6111 |
| SVM B90 | 2.0 | 0.6667 | 1.0000 | 0.6667 |
| SVM C90 | 0.5 | 0.5556 | 0.9444 | 0.6667 |

B mejora frente a A90 en este catálogo: ΔMacro-F1 LR **+0.3482**, intervalo por remuestreo pareado de familias [0.1865, 0.4994]; SVM **+0.3094**, [0.1612, 0.4651]. C mejora frente a A90 en LR +0.2918 [0.1201, 0.4662] y SVM +0.2508 [0.0901, 0.4235]. Son 1000 remuestreos de solo 18 familias sintéticas, no intervalos poblacionales humanos ni una demostración de superioridad general. No se probó la significación de B frente a C.

C obtuvo mejor Macro-F1 de desarrollo que B (LR 0.7996 vs 0.7721; SVM 0.8279 vs 0.7721), pero rindió peor en el desafío. Esto ilustra incertidumbre de selección con desarrollo pequeño y cambio de familias. No se cambió la proporción de mezcla ni se volvió a buscar C usando esta observación. La DEMO y su modelo histórico siguen sin cambios.

## Errores, negaciones y contrastes

La auditoría de TRAIN histórico encontró cierres asociados a clase repetidos entre 19 y 26 veces y cláusulas de negación/contraste repetidas 9 y 21 veces por clase. Esto refuerza la hipótesis de atajos léxicos. El experimento no manipula aisladamente esas frases y no permite atribuirles causalmente toda la caída anterior.

SVM B todavía comete 12 errores sobre 54 textos: cinco negativos→positivos, uno negativo→neutral y seis positivos→negativos. Su recall neutral perfecto puede reflejar que los neutrales sintéticos siguen siendo mayormente descriptivos. No acredita comprensión general de neutralidad.

| Fenómeno de autor, SVM B | Aciertos / soporte | Limitación |
| --- | ---: | --- |
| Negación | 15/20 | Solo un neutral en la rebanada |
| Contraste | 5/8 | Sin neutrales; dos negativos→positivos |
| Doble negación | 2/3 | Los tres casos tienen intención positiva |
| Concordancia negativa | 3/5 | Sin neutrales |
| Aspectos mixtos | 2/2 | Muestra demasiado pequeña |
| Palabras positivas con sentido negativo | 5/5 | Solo negativos; no es Macro-F1 comparable de tres clases |
| Regionalismo | 1/2 | No valida dialectos o poblaciones regionales |
| Error ortográfico | 2/3 | Cobertura escasa de errores posibles |

Las rebanadas se solapan. Se informa exactitud con denominadores porque varias no contienen las tres clases; el JSON conserva Macro-F1 de clases fijas y el indicador de comparabilidad. Las extensiones larga/breve tienen apenas dos casos cada una en desafío. Falta validación independiente de las etiquetas, naturalidad regional y familias semánticas; Jaccard no garantiza ausencia de paráfrasis. El máximo Jaccard entre roles nuevos fue 0.2368 frente al umbral fijado 0.70, sin candidatos ni duplicados exactos normalizados.

Tres fallos ilustrativos nuevos, conservados sin modificar el catálogo:

- `v2_40be67d48dd3e6b78020`: queja por información meteorológica desactualizada, con «No me molestó cambiar el plan, me molestó enterarme…». Intención negativa, predicción positiva y aceptación automática. La política actual no detecta esta forma de contraste sin sus conectores explícitos.
- `v2_40dac3f904b048ae1d51`: «No hubo una foto que no pudiéramos repetir…». Intención positiva, predicción negativa y aceptación automática; persiste un fallo de alcance de doble negación.
- `v2_7cd4549daaf08321b338`: problema de fichas/colas que quita las ganas de probar comida. Intención negativa, predicción positiva y aceptación automática; cuesta integrar la consecuencia de la experiencia.

Estos ejemplos explican errores después de la evaluación. No se convierten en reglas, ajustes o nuevos ejemplos de entrenamiento para este experimento congelado.

## Revisión humana, abstención y latencia

Se aplicó exactamente la política actual: confianza máxima <0.56, o `pero/aunque/sin embargo` y confianza <0.80. No se optimizaron umbrales. Las puntuaciones probabilísticas no tienen fiabilidad externa validada.

| Modelo | Aceptados / 54 | Errores entre aceptados | Negativos→positivos automáticos / 18 | Ambiguos derivados / 9 |
| --- | ---: | ---: | ---: | ---: |
| LR A90 | 0 | No definido (`null`) | 0 | 9 |
| LR B90 | 0 | No definido (`null`) | 0 | 9 |
| LR C90 | 4 | 0/4 | 0 | 9 |
| SVM A90 | 4 | 1/4 | 1 | 7 |
| SVM B90 | 34 | 5/34 (14.7%) | 2 | 2 |
| SVM C90 | 24 | 3/24 (12.5%) | 2 | 6 |

El aumento de cobertura de SVM B (63.0%) viene acompañado de errores aceptados y solo 2/9 casos ambiguos derivados. LR B deriva todo; esto no equivale a un sistema operativo resuelto ni a riesgo cero. Los nueve casos ambiguos representan expectativa de revisión del autor, sin sentimiento gold ni evaluación de acuerdo entre humanos. Dummy predice siempre negativo, acepta todo con su confianza degenerada y se equivoca en 36/54: cero negativos→positivos por sí solo no es una garantía de utilidad o seguridad.

La latencia cargada, lote 1, en desarrollo nuevo fue LR B mediana **1.73 ms**, p95 **2.70 ms**; SVM B mediana **7.65 ms**, p95 **10.92 ms** (360 llamadas después de cinco calentamientos). Incluye vectorización; no mide UI, orquestación, audio, concurrencia o tráfico real. El entorno científico fue Python 3.12.14, sklearn 1.9.1, NumPy 2.5.4, SciPy 1.18.1 y joblib 1.6.0; las versiones se guardan y fijan para reproducción en CI.

## BETO: viabilidad técnica medida

La prueba aislada descargó y verificó el checkpoint oficial `dccuchile/bert-base-spanish-wwm-cased`, revisión `c4d86612f51b4f46759c8390d1798c2febe71b93`. Los pesos son 439621341 bytes, SHA-256 `e131a95091c777bbd45250fb647fec415010cb8cd1ad6e1d59babeb82a0be360`. La caché, pesos y modelos ajustados experimentales permanecen ignorados; el PR publica solo metadatos e informes.

| Comprobación | Resultado observado |
| --- | --- |
| Entorno separado | Python 3.12.14, Torch 2.9.1+cpu, Transformers 4.57.6 |
| Dispositivo | CPU, dos hilos; CUDA no disponible |
| Tokenización de dos entradas ilustrativas | 13 y 11 tokens con especiales; cero UNK |
| Salida | `[2, 13, 768]`, todos los valores finitos |
| Truncamiento técnico | 128 tokens |
| Parámetros del encoder sin pooler | 109260288; FP32 416.8 MiB |
| Pico de RSS del proceso | 802.6 MiB, Linux `getrusage` |
| Tokenización + inferencia, lote 2 | Cinco pasadas: aproximadamente 61.7–88.0 ms |
| Ajuste completo con Adam | Mínimo teórico de parámetros/gradientes/momentos 1667.2 MiB; RAM total pendiente |
| GPU y ajuste supervisado | No medidos |

Una lectura de psutil inferior al tamaño de los propios parámetros se descartó por inconsistente; el informe conserva el diagnóstico y utiliza la marca máxima de memoria del proceso de Linux. No se recomienda RAM a partir de la lectura inválida. Los 802.6 MiB son un consumo observado con dos entradas cortas, no un presupuesto general para ajuste, secuencias largas o concurrencia.

La ficha oficial expresa intención de licencia CC BY 4.0, con incertidumbre reconocida sobre la compatibilidad de los textos originales de preentrenamiento, especialmente para uso comercial. Se conserva la [fuente exacta](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/blob/c4d86612f51b4f46759c8390d1798c2febe71b93/README.md). No se declara autorización integral de los corpus originales.

El encoder no entrega clases o exactitud de sentimiento. Una futura cabeza `Linear(768, 3)` tendría 2307 parámetros, se inicializaría sin conocimiento de sentimientos y necesitaría ajuste supervisado. Antes de ejecutarlo se deberá fijar corpus admitido, particiones, pooling, presupuesto de memoria y selección en desarrollo. Este PR no crea ni evalúa esa cabeza y no modifica a LR/SVM de la aplicación.

## Evidencia y siguiente ejecución

- [Informe científico completo](research/synthetic-diversity-v2-first-report.json): métricas, soportes, confusiones, intervalos, errores por ID, latencia, versiones y limitaciones.
- [Selección previa al desafío](research/synthetic-diversity-v2-selection.json): presupuestos, fuentes, hashes de IDs y parámetros.
- [Manifiesto del corpus](research/synthetic-diversity-v2-manifest.json): procedencia, SHA-256 y auditoría léxica.
- [Probe BETO real](research/beto-encoder-probe-2026-11-07.json): dimensiones, hashes, memoria y tiempos.
- Pruebas locales: **230 passed, 1 xfailed** conocido; las 28 nuevas pasan sin descargar BETO ni ajustar un modelo sobre el desafío real. El wheel se construyó con ambos recursos JSON incluidos.
- `.github/workflows/synthetic-diversity-v2.yml` repite la especificación congelada y verifica hashes, parámetros, matrices de confusión y Macro-F1 contra la primera ejecución. Publica predicciones sintéticas por ID y diarios como artefactos de 30 días; no publica pesos. El resultado efectivo de CI debe comprobarse en el PR, no suponerse a partir de esta configuración.

Para una reproducción local, usar un directorio nuevo:

```bash
python -m pip install -r research/requirements-synthetic-v2.txt -e '.[test]'
python -m traveler_agent.synthetic_diversity --out data/processed/synthetic-v2-local-01
python -m traveler_agent.synthetic_diversity_benchmark --data data/processed/synthetic-v2-local-01 --out artifacts/synthetic-v2-local-01 --registration-commit e3420606c721622d4c9134b0a1c82cef78982744 --reproduction
```

**Siguiente hito humano:** admitir `travelerinsights2026/train.csv` obtenido con acceso autorizado, o una alternativa humana con licencia, procedencia y tres clases verificadas, según el protocolo de PILOTO-01 del PR #20. Siguen pendientes los datos humanos y D; el corpus UNED se conserva como evidencia de auditoría y no entra automáticamente en BETO. No hay todavía benchmark humano ni afirmación de robustez en turistas reales.
