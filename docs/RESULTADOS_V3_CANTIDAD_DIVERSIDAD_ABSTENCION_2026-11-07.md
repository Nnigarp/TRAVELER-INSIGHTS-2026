# V3: resultados de cantidad, diversidad y riesgo selectivo

**Resultado:** duplicar textos ayudó en los cuatro contrastes puntuales dentro de este catálogo, pero ampliar de 10 a 20 familias al mismo presupuesto no produjo una mejora consistente. La calibración mejoró algunas medidas probabilísticas, sin resolver la revisión humana. No se cambia el modelo o política de la DEMO.

La ejecución se realizó después del registro público [`72e400124d0c6e3004976d4e81ad5636eb862841`](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/commit/72e400124d0c6e3004976d4e81ad5636eb862841), partiendo de `main@6f7334b90b5655535b3fac41687febf690781106`. Catálogo, protocolo y código relevante se verifican por hash; los C, temperaturas, políticas y ablaciones terminaron y se guardaron antes de una inferencia de assessment por estimador. No se modificó nada de ello después de ver los resultados. **Todos los sentimientos y fenómenos son intención del autor sintético, no gold humano.**

## Qué se midió

Se reutilizaron Dummy, TF-IDF de palabra+carácter con LR y SVM, selección C={0.5,2} y calibración interna SVM por familias de TRAIN. Todo el pool es V3, sin reutilizar reseñas V2 para entrenamiento o selección. Los 335 textos nuevos se distribuyen en 180 de pool TRAIN, 30 de tune, 30 de calibración, 30 de política, 45 de assessment y dos cohortes ambiguas de diez. Las variantes relacionadas se agrupan; hoteles, restaurantes, museos, transporte y recorridos, y los cinco aspectos, están equilibrados por clase en las cuatro condiciones principales.

El [prerregistro](PRERREGISTRO_V3_CANTIDAD_DIVERSIDAD_ABSTENCION_2026-11-07.md) describe todos los controles. [El manifiesto](research/v3/corpus-manifest.json) conserva distribución, hashes y pantalla léxica entre cada par de roles. No hubo duplicados exactos con V2 ni candidatos entre roles nuevos al umbral Jaccard 0.70. El mismo asistente redactó los lotes: familias y filtros no prueban independencia semántica, humana o externa.

## Clasificación: cantidad frente a familias

Assessment nuevo: 45 textos de 15 familias; 15 por clase. Cada condición tiene 20 o 40 textos por clase y usa tune común de 30, separado de calibración/política/evaluación.

| Condición | Familias | Variantes por clase/familia | Filas | LR Macro-F1 | SVM Macro-F1 | SVM negativos→positivos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Estrecha 60 | 10 | 2 | 60 | 0.6659 | 0.6868 | 2/15 |
| Estrecha 120 | 10 | 4 | 120 | 0.7932 | 0.8182 | 2/15 |
| Amplia 60 | 20 | 1 | 60 | 0.6683 | 0.6248 | 5/15 |
| Amplia 120 | 20 | 2 | 120 | 0.7577 | 0.7346 | 2/15 |

Dummy obtuvo 0.1667 y accuracy 1/3 en las cuatro condiciones, prediciendo siempre negativo. Cero negativos→positivos en ese modelo no representa utilidad o seguridad: se equivoca en todos los neutrales y positivos.

| Modelo | Condición | C de tune | Recall negativo | Recall neutral | Recall positivo |
| --- | --- | ---: | ---: | ---: | ---: |
| LR | Estrecha 60 | 0.5 | 0.6000 | 0.8000 | 0.6000 |
| LR | Estrecha 120 | 0.5 | 0.6000 | 0.9333 | 0.8667 |
| LR | Amplia 60 | 2.0 | 0.7333 | 0.5333 | 0.7333 |
| LR | Amplia 120 | 0.5 | 0.7333 | 0.8000 | 0.7333 |
| SVM | Estrecha 60 | 0.5 | 0.6000 | 0.8000 | 0.6667 |
| SVM | Estrecha 120 | 0.5 | 0.6667 | 0.9333 | 0.8667 |
| SVM | Amplia 60 | 2.0 | 0.6667 | 0.4667 | 0.7333 |
| SVM | Amplia 120 | 0.5 | 0.6667 | 0.8000 | 0.7333 |

Matrices de dos condiciones SVM, filas=intención de autor y columnas=predicción, orden negativo/neutral/positivo:

| Condición | Negativo real | Neutral real | Positivo real |
| --- | --- | --- | --- |
| Estrecha 120 | [10, 3, 2] | [0, 14, 1] | [2, 0, 13] |
| Amplia 120 | [10, 3, 2] | [2, 12, 1] | [4, 0, 11] |

Todas las matrices, F1 y soportes por clase se conservan en el [informe completo](research/v3/first-experiment-report.json). Ampliar familias a 120 no mejoró el recall negativo de SVM y redujo recall neutral/positivo en esta realización. LR amplia 120 tiene solo un negativo→positivo, frente a dos de LR estrecha 120; ningún modelo domina todas las métricas.

### Contrastes pareados, no conclusiones poblacionales

| Contraste de Macro-F1 | LR Δ [intervalo 95%] | SVM Δ [intervalo 95%] |
| --- | --- | --- |
| Cantidad, 10 familias: 120−60 | +0.1273 [−0.0032, +0.2602] | +0.1314 [+0.0384, +0.2446] |
| Cantidad, 20 familias: 120−60 | +0.0894 [−0.0469, +0.2374] | +0.1098 [−0.0352, +0.2625] |
| Diversidad a 60: 20−10 familias | +0.0024 [−0.2131, +0.2066] | −0.0620 [−0.2579, +0.1140] |
| Diversidad a 120: 20−10 familias | −0.0354 [−0.1743, +0.1029] | −0.0836 [−0.2166, +0.0224] |

Se remuestrearon 1000 veces las mismas 15 familias para cada contraste. Solo el contraste de cantidad estrecha de SVM excluye cero en este análisis exploratorio. No hay corrección por multiplicidad, réplicas independientes de generación ni inferencia humana. Los intervalos de interacción también cruzan cero. No se concluye que más diversidad perjudique en general: el presupuesto fijo reduce variantes por familia y el diseño no separa todos los factores de estilo/contenido.

En tune la condición amplia 120 parecía más favorable (LR y SVM 0.8993), frente a estrecha 120 (LR 0.8011, SVM 0.7716), pero el orden se invirtió en assessment. No se cambió el pool ni se eligió otra semilla tras verlo. **No se comparan las cifras V3 con el Macro-F1 V2 como si fueran el mismo test.** V3 ya fue observado y tampoco seguirá considerándose un test intacto para futuras optimizaciones.

## Aporte de dominios y aspectos, solo desarrollo

Se retiró una dimensión de TRAIN y se repusieron variantes hasta 120 textos equilibrados. Las siguientes diferencias son de Macro-F1 de tune frente a amplia 120. Cada rebanada del dominio/aspecto omitido tiene solo seis textos, dos por clase.

| Dimensión omitida | LR Δ tune general | SVM Δ tune general |
| --- | ---: | ---: |
| Hotel | −0.0738 | −0.0738 |
| Restaurante | −0.0673 | −0.0673 |
| Museo | −0.0350 | −0.0753 |
| Transporte | −0.1127 | −0.1127 |
| Recorrido | −0.1027 | −0.1359 |
| Limpieza | −0.0370 | −0.0350 |
| Seguridad | −0.1355 | −0.1355 |
| Accesibilidad | −0.0753 | −0.0738 |
| Precio | −0.0361 | −0.0361 |
| Atención | −0.1347 | −0.0997 |

Señales exploratorias: seguridad/atención y transporte/recorridos parecen aportar a ese desarrollo. No todas las rebanadas omitidas empeoraron: varias conservan la puntuación local aunque cambie la general. El [informe de desarrollo](research/v3/development-report.json) contiene ambas diferencias y parámetros. No son efectos causales aislados: la reposición altera composición/variantes y vuelve a seleccionar C. Ninguno de estos modelos de ablación se puntúa en assessment ni provoca una nueva combinación de entrenamiento.

## Calibración: mejor puntuación probabilística no significa revisión resuelta

La temperatura se ajustó con 30 casos separados y se eligió 0.5 en LR y casi todos los SVM; SVM amplia 60 mantuvo 1.0. El argmax no cambia. Las siguientes cifras corresponden al assessment independiente, después de congelar la temperatura.

| Modelo | NLL original → temperatura | Brier suma original → temperatura | ECE original → temperatura |
| --- | --- | --- | --- |
| SVM estrecha 120 | 0.6060 → 0.4902 | 0.3423 → 0.2850 | 0.2030 → 0.0517 |
| SVM amplia 120 | 0.6732 → 0.6218 | 0.3872 → 0.3508 | 0.1758 → 0.0729 |
| LR amplia 120 | 0.9593 → 0.8502 | 0.5716 → 0.4963 | 0.3536 → 0.2971 |

Se guardan bins de confianza máxima y fiabilidad de cada clase. NLL/Brier incluyen discriminación e incertidumbre, y ECE depende de bins y muestras pequeñas. Una ECE menor no prueba calibración humana ni seguridad selectiva; la referencia técnica es la [documentación oficial](https://scikit-learn.org/stable/modules/calibration.html).

### Cobertura, riesgo y ambiguos

La política original se mide sin cambios. La política experimental se eligió únicamente con 30 casos de desarrollo y diez ambiguos de desarrollo: riesgo empírico ≤10%, cero negativos→positivos automáticos, al menos nueve aceptados/cobertura ≥25% y ≥80% de ambiguos derivados. Si no hay candidato, el resultado fijado es revisar todo. Es un criterio empírico, no una garantía.

| Modelo / política | Aceptados / 45 | Errores aceptados | Negativos→positivos automáticos / 15 | Ambiguos derivados / 10 |
| --- | ---: | ---: | ---: | ---: |
| SVM estrecha 120, original | 29 | 2/29 (6.9%) | 0 | 3 |
| SVM estrecha 120, temperatura + misma regla | 38 | 6/38 (15.8%) | 1 | 1 |
| SVM estrecha 120, seleccionada en desarrollo | 19 | 2/19 (10.5%) | 0 | 6 |
| SVM amplia 120, original | 31 | 4/31 (12.9%) | 1 | 3 |
| SVM amplia 120, temperatura + misma regla | 40 | 9/40 (22.5%) | 1 | 2 |
| SVM amplia 120, seleccionada: revisar todo | 0 | No definido (`null`) | 0 | 10 |
| LR amplia 60, seleccionada en desarrollo | 9 | 3/9 (33.3%) | 0 | 6 |

La seleccionada SVM estrecha 120 (`tau_global=0.85`, `tau_positivo=0.65`, margen 0.10, T=0.5) había aceptado 13/30 sin errores y derivado 8/10 ambiguos en desarrollo. En evaluación **no mantuvo** ni el riesgo ≤10% ni la derivación ≥80%. Además pierde cobertura frente a la regla original y no mejora su número de negativos→positivos automáticos, que ya era cero en esa condición. La política LR amplia 60 también falla al trasladarse: pasa de 0/10 errores aceptados en desarrollo a 3/9, con cobertura 20%, inferior al mínimo previsto.

Solo esas dos políticas LR/SVM resultaron elegibles en desarrollo. Las otras seis recurrieron a revisar todo. Reducir negativos→positivos automáticos a cero absteniéndose en todo **no constituye una solución operativa**. No se relajaron restricciones, alteraron umbrales o eligieron otras políticas usando estos fallos. Las curvas de umbrales predefinidos se conservan como descripciones sin selección final.

## Errores concretos y huecos de cobertura

Se conservaron IDs/predicciones y se analizaron tras abrir la evaluación, sin reetiquetar ni ajustar el generador/modelo:

- `v3_5854c0fd200a88f32ce8`: el precio del pescado se explica por peso solo al pagar. Intención negativa, predicción positiva en SVM estrecha y amplia 120. La regla original de amplia 120 lo acepta automáticamente; en estrecha 120 aparece como aceptado al aplicar temperatura con los umbrales anteriores. La consecuencia económica implícita sigue costando.
- `v3_d22c850a94ed35e87442`: la compra de pasaje no puede completarse con teclado. SVM estrecha 120 predice positivo; persiste un fallo al integrar una negación con una barrera de accesibilidad.
- `v3_550b680b7dae056f3f2f`: se permite participar por escrito sin exigir habla. Ambos SVM120 predicen negativo aunque la intención del autor es favorable; la negación de una exigencia se interpreta mal.
- `v3_e48fb7311a4ff51340b9`: el hotel sustituye una cuna defectuosa antes de usarla. Ambos SVM120 predicen negativo: el fallo inicial eclipsa la solución que sustenta la valoración favorable.
- `v3_31101e83677c3f614d35`: el relato del conteo de nadadores se etiquetó neutral por su forma descriptiva y SVM estrecha 120 predice positivo. Es también una posible ambigüedad de anotación: un humano podría leer aprobación implícita. La intención sintética no permite atribuir todos los desacuerdos al modelo.

En SVM estrecha 120: negaciones 15/17 aciertos; contrastes implícitos 10/16. Las dos dobles negaciones y las dos concordancias negativas fueron acertadas, pero son únicamente neutrales. Los seis casos marcados como regionalismo tienen intención positiva; seis aciertos no validan el español de países o dialectos.

**Huecos que permanecen:** assessment no tiene casos marcados de error ortográfico, contraste explícito, aspectos mixtos o palabras positivas en contexto negativo, ni textos de ≤12 palabras. Hay solo tres textos largos, uno por clase. Esos fenómenos aparecen en el pool TRAIN o en las cohortes ambiguas, pero no se inventa una medida de robustez donde el assessment no tiene soporte. No se añaden ahora al test observado; requerirían una nueva evaluación independiente.

El control de redacción tampoco eliminó todos los atajos. En amplia 120, la sonda completa de metadatos de TRAIN bajó de 0.879 a 0.751 tras la revisión, mientras la sonda de longitud/puntuación subió de 0.400 a 0.561. La primera incorpora etiquetas semánticas ligadas a intención; la segunda indica que siguen diferencias formales aprovechables. No hubo cierres idénticos de cuatro palabras compartidos entre familias en TRAIN, pero eso no descarta otras señales repetidas. Se publican el TRAIN descartado del primer borrador y los diagnósticos como evidencia de autoría; el cargador solo admite el catálogo final y no mezcla ese borrador en el factorial.

## Costos medidos y reproducción

| Medida, ejecución local Linux | Resultado |
| --- | --- |
| Tiempo de pared del runner completo | 46.18 s |
| CPU acumulada del proceso | 251.17 s, puede sumar varios hilos |
| Pico RSS del proceso completo | 195.0 MiB, `getrusage` |
| CPU lógicas visibles | 9 |
| Ajuste de candidatos principales, suma | 6.91 s |
| Ajuste de candidatos de ablación, suma | 17.82 s |
| LR estrecha 120, latencia cargada p50/p95 | 2.04 / 2.98 ms |
| SVM estrecha 120, latencia cargada p50/p95 | 9.32 / 16.53 ms |
| LR amplia 120, latencia cargada p50/p95 | 2.08 / 3.14 ms |
| SVM amplia 120, latencia cargada p50/p95 | 8.80 / 13.21 ms |
| Artefactos LR/SVM estrecha 120 | 719239 / 1613346 bytes |

Se ajustaron 60 candidatos externos: 20 principales y 40 de ablación; cada SVM incluye sus folds internos. El total también contiene sondas de TRAIN, predicciones, selección de política, 90 medidas de latencia por modelo, escritura y verificación. No se incluye redacción, instalación, CI, UI, ASR, concurrencia o GPU. No hay tarifa del equipo para expresar costo monetario. Se reutilizan las versiones fijadas de #21: Python 3.12.14, sklearn 1.9.1, NumPy 2.5.4, SciPy 1.18.1, joblib 1.6.0, pandas 3.1.1 y threadpoolctl 3.8.0.

El [informe de selección](research/v3/selection-manifest.json) y el de desarrollo son copias byte a byte de lo escrito antes de assessment, de modo que pueden comprobarse los hashes referidos por el informe completo. Los modelos experimentales quedan en `artifacts/synthetic-v3/first/models/`, ignorados y sin integración en la aplicación. CI publica registros/predicciones sintéticos y diarios, no pesos.

Reproducción en un checkout con líneas LF (Linux/WSL; en Windows usar un clon separado con `core.autocrlf=false` y la ruta `Scripts/python.exe` del entorno). Usar directorios nuevos:

```bash
python -m pip install -r research/requirements-synthetic-v2.txt -e '.[test]'
python -m traveler_agent.research_v3_data --out data/processed/synthetic-v3-local-01
python -m traveler_agent.research_v3_benchmark --data data/processed/synthetic-v3-local-01 --out artifacts/synthetic-v3-local-01 --registration-commit 72e400124d0c6e3004976d4e81ad5636eb862841 --reproduction
```

Pruebas locales: **254 passed, 1 xfailed** conocido; 24 pruebas nuevas, incluidas garantías de roles, presupuestos, calibración, fallback y orden de acceso, con fixtures que no entrenan sobre assessment real. El workflow V3 compara contra la primera ejecución hashes de código/datos, C, temperaturas, políticas, matrices y recuentos de riesgo. Sus tiempos y memoria pueden variar. La CI efectiva debe verificarse en el PR; una configuración no se presenta como ejecución exitosa.

## Estado del producto, BETO y datos humanos

Se preservaron TravelerOrchestrator, PerceptionAgent, SemanticsAgent, ControlAgent, ActionAgent, Midnight Indigo, Whisper, transcripción editable, lotes, revisión, exportaciones y trazabilidad. No se promovió ningún clasificador o umbral nuevo. V2 conserva código, datos e informes; el único ajuste de su workflow acota el filtro a los recursos exactos de V2 para que futuras adiciones V3 no lo disparen. Cambiar ese workflow puede activar su propia reproducción congelada de validación, cuyos resultados no seleccionan modelos V3.

BETO sigue siendo la prueba de encoder de #21, técnicamente viable para inferencia CPU, sin ajuste de sentimientos. Esta fase no añadió pooling, cabeza o ajuste supervisado; el consumo de entrenamiento completo no está medido. Se mantiene pendiente un protocolo contextual específico con revisión de calidad de etiquetas y presupuesto de entrenamiento. El resultado clásico actual no justifica afirmar que un Transformer resolverá las negaciones o el riesgo selectivo. UNED permanece como evidencia de auditoría, sin entrenamiento BETO automático.

La ruta humana de PILOTO-01 sigue lista: Kaggle mediante acceso del responsable, aceptación personal de reglas y descarga solo de `train.csv`, o contribuciones originales consentidas. Continúan pendientes la evidencia de autorización/procedencia, mapeo auténtico de etiquetas o dos anotaciones humanas independientes y adjudicación, y revisión de privacidad/lengua/grupos. No hay todavía corpus humano admitido ni resultado humano que medir.

**Decisión derivada:** conservar este experimento y sus fallos; no hacer otra búsqueda sobre assessment V2/V3. La siguiente evidencia útil es revisar con humanos las intenciones y ambigüedades, ampliar desarrollo y validar prospectivamente cobertura/riesgo antes de una integración. La revisión y eventual fusión de este incremento son manuales.
