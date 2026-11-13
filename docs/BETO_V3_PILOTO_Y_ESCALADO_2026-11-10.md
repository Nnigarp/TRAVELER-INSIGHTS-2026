# BETO V3: piloto de calidad y plan de ampliación a 5.000

Investigación del **10 de noviembre de 2026**. Se construyó un primer corpus operativo de **564 borradores sintéticos**, antes de comprometer la generación y el entrenamiento de 5.000. Dos casos ambiguos se excluyeron tras adjudicación: quedan **562 muestras admitidas**, de las cuales 540 tienen sentimiento previsto y 22 solo expectativa de revisión. No hay turistas reales, anotación humana ni métricas humanas en esta fase.


Actualización operativa del **13 de noviembre de 2026**: PR #26 ya está fusionado; main verificado `1628df6fe9a1c9ee68ab34096444177ba4c4a580`. PR #27 se ha reorientado a main. El entorno recuperado no conserva los archivos raw de la primera evaluación que Work informó; sí conserva en Git el corpus, protocolo y selección previa. Los [agregados informados](research/beto-v3-prior-reported-aggregates.json) se preservan sin inventar probabilidades. El [protocolo de reconstrucción](research/beto-v3-recovery-protocol-2026-11-13.json) fija las recetas y el presupuesto antes de volver a ejecutarlas. Las próximas mediciones son **reproducción de una cohorte ya observada**, no un test nuevo ni recuperación byte a byte de los archivos perdidos.

## Estado de GitHub y aislamiento

Se verificó `main@9e3f0d8d9c1591921a7103952dbe83de234be906`: #23 y #24 están fusionados en main. #25 se fusionó en la antigua rama del #24, no en main; #26 propone incorporar ese resultado y seguía abierto al iniciar este incremento. V3 se prepara en `feat/beto-v3-quality-pilot-20261110`, dependiente de #26, sin duplicar sus cambios ni fusionar PRs.

TravelerOrchestrator, PerceptionAgent, SemanticsAgent, ControlAgent y ActionAgent permanecen intactos. También Midnight Indigo, Whisper, transcripción editable, lotes, trazas, revisión humana, exportaciones y respaldo explícito a LR. V2/V3 sintéticas históricas, Contextual V1 y BETO V2 no cambian ni se usan para elegir nuevas configuraciones sobre sus evaluaciones. Se reutilizan sus helpers y el formato seguro de artefactos supervisados; usar un formato llamado `supervised-v2` no convierte el experimento V3 en el experimento anterior.

## Registro anterior a entrenamiento y evaluación

| Registro | Evidencia |
| --- | --- |
| Protocolo antes de redactar reseñas/entrenar | `e4c52c18c146355ab955debc1255e680af8df2a4` |
| Primer corpus original y auditoría fallida de estilo | `d05c9bd1883efc1e5a91f068449e378f0e04fd3b` |
| Única revisión de estilo de TRAIN, preinscrita | `d0bde7e50dc7a101cd696a67f8fe0979bd881397` |
| Corpus admitido, código y pruebas antes de entrenar clasificadores del piloto | `07935b6dc5eee6d62d2f62c6d67ce47b21cc58b9` |

Los baselines de forma y los fixtures de optimización forman parte de la auditoría/pruebas anteriores al entrenamiento de sentimiento. La [inscripción](research/beto-v3-registration.json) conserva hashes, presupuesto, versiones, tolerancia numérica e intervención de estilo. La selección de modelos se publica y verifica mediante otro commit **antes** de la primera predicción de assessment.

## Corpus y procedencia

| Papel | Generadas | Admitidas | Uso exclusivo |
| --- | ---: | ---: | --- |
| TRAIN | 300 | 300 | Optimización; 100 por clase/100 familias |
| Tune | 60 | 60 | C, configuración y épocas |
| Calibración | 60 | 60 | Temperatura por NLL |
| Política | 60 | 60 | Selección de cobertura/riesgo |
| Assessment | 60 | 60 | Comparación común; 20 por clase/20 familias |
| Ambigüedad de política | 12 | 11 | Revisión, sin sentimiento gold |
| Ambigüedad de assessment | 12 | 11 | Evaluación de revisión, sin sentimiento gold |

Los 180 escenarios etiquetados contienen tres experiencias distintas, una por sentimiento, dentro de la misma familia y partición. Hay además 24 escenarios de ambigüedad. Las familias y sus componentes se mantienen juntos; no se distribuyen variantes relacionadas aleatoriamente entre train y test. Los controles exactos/cercanos abarcan también copias dentro de una familia; un cambio de adjetivo no obtiene una excepción automática.

Se cubren Colombia, Perú, México, España y otros contextos hispanohablantes. Diez dominios canónicos incluyen hotel, restaurante, taxi, aeropuerto, transporte, playa, museo, tour, reserva y excursión natural; el subtipo distingue transporte local/interurbano. Los objetivos de longitud son 15–30, 50–85 y 140–180 palabras. Una familia comparte banda de longitud entre sus tres clases. En TRAIN, **99/300 textos superan 128 tokens** después de la edición; el máximo es 201 y ninguno supera 256. Esto permite estudiar el truncamiento a 128 sin justificar ventanas mayores que 256 en este piloto.

Los textos fueron redactados individualmente por agentes reales separados: un planificador de escenarios, cinco autores regionales y un autor de evaluación con conversación inicial vacía, sin acceso a TRAIN/desarrollo o resultados históricos. No se generaron reseñas mediante sustituciones de sustantivos, sufijos de polaridad o multiplicación de paráfrasis. La [instrucción de autoría](research/beto-v3-writer-brief.md), los lotes y los metadatos identifican las tareas efectivamente ejecutadas. La versión exacta y semilla del modelo generador no están expuestas por la plataforma: se declaran desconocidas. Los textos versionados permiten reproducir el entrenamiento, no prometer regeneración determinista de un LLM.

La independencia es procedimental: las tareas no constituyen modelos base distintos ni prueban independencia poblacional. La naturalidad regional no ha sido validada por hablantes humanos. No se inventan autores humanos ni se publican los textos como reseñas de negocios reales.

## Auditoría y revisión independiente

El primer baseline OOF por familias, usando solo longitud y puntuación de TRAIN, alcanzó **Macro-F1 0,4986**, superior al límite preinscrito **0,45**. Por ello se detuvo la admisión. Se conservó la versión fallida y se registró una única corrección editorial de TRAIN: estilos fluidos, fragmentados y mixtos contrabalanceados entre clases. Los editores recibieron texto y perfil, sin etiqueta propuesta. No se añadieron filas ni se cambiaron hechos, etiquetas o escenarios; no se borró mecánicamente la puntuación ni se usó relleno.

El resultado fue **0,3854**, bajo el mismo límite. Los hashes normalizados de tune, calibración, política, assessment y ambos grupos ambiguos permanecen idénticos. La [recepción de la revisión](research/beto-v3-style-revision-receipt.json) y el [brief editorial](research/beto-v3-style-brief.md) documentan el cambio. También se refrescaron las etiquetas de fenómenos sobre el texto editado: no se conservan automáticamente tags que la nueva redacción ya no expresa.

Dos revisores AI independientes vieron únicamente texto e identificador. Revisaron una muestra estratificada de **90 etiquetadas** y los 24 ambiguos. Tras la revisión de estilo, ambos coinciden con las 90 intenciones propuestas y entre sí (kappa 1,0 en esa muestra). Esto no acredita perfección en las 540 ni gold humano. Tres diferencias en el grupo ambiguo se adjudicaron; dos expresaban valoración positiva identificable y se excluyeron del grupo de ambigüedad, conservando su redacción original en Git.

La admisión exige hashes, criterios numéricos automáticos, dos revisiones distintas y adjudicaciones completas; una bandera `admitted:true` por sí sola no permite entrenar. Se registran diversidad léxica, indicadores estructurales, metadatos, cierres repetidos, soporte de fenómenos y truncamientos. Son proxies y anotaciones AI, no análisis sintáctico humano. La corrección editorial redujo el ruido ortográfico en TRAIN: las rebanadas con typos son pequeñas y no sostienen una conclusión de robustez ortográfica.

El [diagnóstico léxico de TRAIN](research/beto-v3-train-lexical.json), añadido como análisis descriptivo durante el entrenamiento y sin cambiar decisiones, muestra asociaciones residuales: «pero» aparece en 51 negativos, 1 neutral y 12 positivos; «pude», en 25 positivos y 1 negativo. Hay 23 negativos frente a 4 positivos con opiniones mixtas, y la ironía anotada aparece solo en negativos. Toda la neutralidad de TRAIN incluye cronología objetiva. Estas asociaciones pueden contener evidencia legítima y convenciones del generador; no demuestran causalidad ni ausencia de otros atajos. El clasificador de metadatos obtiene Macro-F1 0,2077 OOF. No se editarán estos datos a partir de resultados del assessment. Una futura ampliación deberá diversificar también neutralidad, contrastes positivos y opiniones mixtas con polaridad global inequívoca, antes de congelar otra evaluación.

[Auditoría automática](research/beto-v3-quality.json) · [Revisión/adjudicación AI](research/beto-v3-ai-review.json) · [Auditoría inicial conservada](research/beto-v3-initial-quality.json).

Una comprobación adicional final mediante otro agente no terminó por cuota de la plataforma; no se presenta como auditoría completada. Los hallazgos independientes anteriores sí motivaron correcciones probadas de admisión, presupuestos y bloqueo de assessment. La inspección y ejecución restantes corresponden al coordinador y a CI.

## Experimento y presupuesto

BETO `dccuchile/bert-base-spanish-wwm-cased`, revisión `c4d86612f51b4f46759c8390d1798c2febe71b93`, peso SHA-256 `e131a95091c777bbd45250fb647fec415010cb8cd1ad6e1d59babeb82a0be360`. Python 3.12, torch 2.9.1 CPU, dos hilos. [Dependencias fijadas](../research/requirements-beto-v3.txt). El [aviso del checkpoint fijado](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/blob/c4d86612f51b4f46759c8390d1798c2febe71b93/README.md) expresa intención CC BY 4.0 y dudas sobre compatibilidad de los corpus de preentrenamiento; se conserva al exportar. No se publican pesos.

- Dummy, TF-IDF+LR y TF-IDF+SVM usan exactamente las mismas filas admitidas. C en .1/1/10 se selecciona solo en tune: Macro-F1, menos negativos→positivos y menor NLL. SVM calibra con tres folds agrupados de TRAIN y TF-IDF dentro de cada fold.
- Cabeza supervisada: AdamW, LR .01, decay .01, dropout .1, hasta 80 épocas; selección20/40/80. Encoder congelado y media normalizada L2, con128 y256 tokens.
- Ajuste parcial de los dos últimos bloques,256 tokens: LR encoder1e-5/cabeza5e-4, decay.01 y dropout.1. Variante regularizada: LR2e-5, decay.1, dropout.2 y smoothing.05. Ese paquete no aísla el efecto de cada cambio.
- Variante de casos difíciles: mismos hiperparámetros estándar, muestreo con reemplazo y multiplicador máximo1.5 para tags TRAIN preinscritos. Normalización por clase, exactamente N extracciones por época y exposiciones guardadas. Los tags no entran en inferencia.
- Ajustes parciales: máximo 3 épocas y parada temprana real con paciencia1; mejora material de Macro-F1≥1e-6, o menos negativos→positivos, o NLL inferior por≥1e-4 en empate. Se restaura cabeza y encoder del mismo checkpoint.
- Fine-tuning completo,256 tokens, una época, solo si la proyección predefinida cabe en el presupuesto restante. Un candidato omitido/interrumpido se declara sin medir.
- Desarrollo: máximo 2400 s; réplicas, calibración y assessment comparten 900 s. Las llamadas síncronas pueden terminar tarde, pero la comprobación posterior detiene el resto y registra interrupción. No se publica un resultado completo después de exceder el límite.

El mejor ajustado se elige en tune antes de assessment; también se registra el mejor candidato global de desarrollo. La semilla primaria es 42. Las réplicas7/101, cuando caben, conservan configuración y épocas elegidas, sin consultar tune y manteniendo el scheduler del plan original. No se elige la mejor semilla después.

Las referencias de recetas V1/V2, si caben, se reentrenan sobre el **nuevo TRAIN común**, con hiperparámetros históricos fijados: no son los binarios antiguos ni una comparación directa con sus antiguas puntuaciones. No se leen sus evaluaciones para escoger V3.

La temperatura usa únicamente calibración 60. Las políticas usan política 60 + ambiguos 11. Criterios empíricos: ≥20 aceptados, ≥5 negativos aceptados, ≥20 negativos totales, ≥10 componentes aceptados, cobertura≥.25, riesgo≤.10, negativos→positivos automáticos/todos los negativos≤.05 y derivación de ambiguos≥.80. Incluso superar estas reglas sintéticas no certifica seguridad poblacional. No admitir una política no equivale a anunciar «revisar todo» como solución satisfactoria.

El assessment se reserva antes de predecir. Sus locks son exclusivos y también se identifican por hash de corpus para que otra carpeta no aparente un test nuevo. Una vez observado, `develop` lo rechaza; la reproducción usa decisiones fijas y conserva su condición de evaluación ya observada. Las tolerancias CI se fijaron antes: clases exactas y diferencia raw máxima .003.

## Resultados medidos

PENDIENTE_RESULTADOS_V3

## Reproducción y Gradio en Windows

Desde la raíz del clon, con el incremento disponible y destinos nuevos:

```powershell
git fetch origin
git switch feat/beto-v3-quality-pilot-20261110
py -3.12 -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-research\Scripts\python.exe -m pip install -r research/requirements-beto-v3.txt -e '.[ui,test]'
.\.venv-research\Scripts\python.exe scripts/reproduce_beto_v3.py --download --primary-only --output artifacts/beto-v3-demo
$env:TRAVELER_BETO_ARTIFACT = 'artifacts/beto-v3-demo/selected-replay/model'
.\.venv-research\Scripts\python.exe scripts/smoke_gradio_http.py --beto-artifact $env:TRAVELER_BETO_ARTIFACT --out artifacts/beto-v3-demo/http.json
.\.venv-research\Scripts\python.exe app.py
```

Seleccionar BETO expresamente. La app inicia con LR y conserva su respaldo explícito. La comparación de la UI no sustituye al benchmark: sus clásicos usan DEMO y V3 usa TRAIN experimental. La receta anterior comprueba el modelo ya observado, no lo selecciona otra vez. Para reproducir también clásicos y congelados quitar `--primary-only`; se verifican seis modelos fijados.

La descarga de dependencias/checkpoint es preparación explícita; no ocurre durante la predicción. Para restaurar la configuración clásica quitar la variable y reiniciar. El job Windows verifica entrenamiento real y HTTP/cola, pero no sustituye el ensayo del micrófono, voz propia autorizada y navegador del portátil. ASR y sus transcripciones editables conservan la guía existente; no se atribuye una prueba con fixtures a voz humana real.

## Escalado a 5.000

El plan preinscrito suma exactamente 5.000: TRAIN 3501, tune 450, calibración 300, política 249, assessment 450 y 25 + 25 ambiguos. Se reservaron más filas para calibración/política que en la propuesta4000/500/500. Cada rol mantiene escenarios/componentes separados.

**Ese plan no está generado ni entrenado.** El futuro assessment 450 debe ser nuevo y congelado antes de sus modelos: los 60 de este piloto no vuelven a anunciarse como final intacto. Solo TRAIN del piloto puede incorporarse a TRAIN ampliado; el resto de cohortes nuevas requiere separación explícita.

La proyección de coste desde el piloto es una aproximación lineal por filas, no una medición de5000 y no incorpora todos los costes de redacción/revisión. La decisión de ampliar usa calidad y recursos, no seleccionar nuevas técnicas sobre el score final del piloto. El límite de desarrollo para la ampliación es 3600 s CPU en el protocolo actual; si la proyección lo supera, se requiere otro presupuesto/entorno antes de emprender una búsqueda de horas.

No se reemplaza LR automáticamente. El piloto ayuda a identificar cobertura, errores y coste; no acredita desempeño de turistas reales, ni autoriza reutilizar este assessment observado para optimización futura. BM25, Sentence-BERT y RAG no se incorporan en este incremento.
