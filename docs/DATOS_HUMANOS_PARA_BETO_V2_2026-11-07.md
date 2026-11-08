# Datos humanos: vía operativa paralela al entrenamiento BETO V2

Comprobación documental: **7 de noviembre de 2026, America/Bogota**. Se revisaron fuentes primarias y metadatos de repositorios; **no se descargaron corpus, no se abrieron archivos privados, no se contactó a participantes y no se admitió una nueva reseña humana**. El JSON del ensayo del formulario Windows no se considera una reseña admitida.

## Decisión y fuentes comprobadas

No se identificó en esta revisión un corpus externo que reúna a la vez permiso suficientemente documentado para los textos, español turístico y gold global independiente de negativo/neutral/positivo. Esto describe las comprobaciones realizadas; no afirma que esos recursos sean ilegales o carezcan de valor.

| Fuente primaria | Evidencia observada | Decisión para este benchmark |
| --- | --- | --- |
| [OpeNER original español](https://github.com/opener-project/opinion_annotations_es/tree/334e44ca51ca4e44f3e412025caedd39cf7f6b2c), [archivo CLTL](https://cltl.github.io/research/sentiment/) y [SemEval 2022, documentación](https://github.com/jerbarnes/semeval22_structured_sentiment/blob/aa65f2c9fd65a1b7f31b96878da8912b8ff3f67c/README.md) | Opiniones de hoteles, con anotaciones de expresiones, objetivos y emisores. SemEval describe unidades oracionales y polaridades en cada opinión estructurada. En README y árboles revisados no se localizó una licencia explícita para los textos españoles. | Mejor candidata para investigar anotación contextual/aspectos, pero no sustituto directo del gold global. No convertir ausencia de expresiones en neutral ni mezclar polaridades por mayoría. Requiere aclarar permisos y definir/anotar la etiqueta global. |
| [SemEval 2016 Task 5, datos oficiales](https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools) y [guía oficial, p. 1](https://alt.qcri.org/semeval2016/task5/data/uploads/absa2016_annotationguidelines.pdf) | Confirma restaurantes españoles en subtareas 1 y 2. Su neutralidad incluye sentimiento levemente positivo/negativo y no significa objetividad. No se identificó en esa página una autorización explícita para reutilizar los textos. | Investigar permiso y unidad exactos, incluso si se usa `RESTAURANT#GENERAL` de la subtarea 2. Hay una diferencia taxonómica con nuestra neutralidad descriptiva: requeriría justificación y posiblemente reanotación; no un mapeo directo. |
| [COAH, SINAI–Universidad de Jaén](https://sinai.ujaen.es/en/research/resources/coah) | Describe 1816 reseñas de TripAdvisor puntuadas de 1 a 5 y ofrece contacto y enlace de acceso. La página consultada no establece licencia precisa ni acredita una anotación textual global independiente. | Los ratings solo sostendrían un objetivo proxy declarado. No inferir que el nivel intermedio equivale a neutralidad textual ni admitirlo como gold humano de tres clases. |
| [TourPedia, CNR](https://wafi.iit.cnr.it/openervm/about/datasets.html) | El publicador declara CCZero. La propia documentación distingue ratings del autor y `polarity` extraída automáticamente por el tagger OpeNER; el esquema incluye `authorName` y procedencia de plataformas. | La declaración de licencia es evidencia positiva, pero falta documentar su alcance/procedencia para textos de terceros y auditar idioma/privacidad. Su polaridad automática no es anotación humana ni validación independiente de BETO. |
| [Kaggle exacto](https://www.kaggle.com/competitions/travelerinsights2026/data) | Sin nueva sesión autorizada ni `train.csv` verificado en esta revisión. | Se mantiene la ruta personal de acceso; no se cambia su estado a admitido. |

Alcance de la inspección OpeNER: árboles completos de `opener-project/opinion_annotations_es` (334e44c), `cltl/OpeNER_corpus` (abe5e1b) y `jerbarnes/semeval22_structured_sentiment` (aa65f2c), más sus README pertinentes. La única ruta `LICENSE` encontrada en el último árbol pertenece a un baseline de código (`baselines/graph_parser/src/awd`): **no se extiende al corpus**. No se leyeron train/dev/test de esas fuentes.

UNED se conserva como evidencia de auditoría según [la revisión previa](PILOTO01_REVISION_WORK_2026-11-06.md); no se incorpora automáticamente a BETO español. Trip-MAML, Hopinion y Rest-Mex mantienen las condiciones pendientes ya descritas en [el protocolo de acceso](PILOTO01_ACCESO_Y_BENCHMARK_HUMANO_2026-11-06.md).

## Próximo paso sin depender de una plataforma

Usar el flujo privado existente de **PR #23**, sin construir otra aplicación. Los comandos siguientes requieren esa rama, o `main` después de su integración manual; el PR de entrenamiento no fusiona #23 automáticamente:

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli init
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli participant-form --out data/private/delivery/participant-001.html
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli status
```

El responsable entrega un formulario/código por persona mediante un canal privado acordado. La persona relata voluntariamente una experiencia turística vivida, descarga su consentimiento y texto, y los devuelve por ese canal. No solicitar opiniones inventadas ni copiar plataformas. Buscar cobertura real de hoteles, restaurantes, museos, transporte y recorridos; no forzar experiencias o etiquetas para completar cuotas. Registrar la invitación y el sesgo de selección: son aportaciones elicitadas, no una muestra representativa de turistas.

Para admitir cada texto hacen falta: aceptación efectiva del aviso y finalidades; procedencia/originalidad/experiencia verificadas por el coordinador; revisión de privacidad; códigos estables de autor y lugar; dos anotadores distintos que trabajen sin etiquetas ajenas; adjudicación por tercera persona. Neutralidad, mezcla y ambigüedad permanecen separadas. Conservar registro privado, distribución de clases y acuerdo; no contar plantillas o pruebas como personas. Agrupar por autor **o** lugar **o** relación/duplicación antes de separar entrenamiento, desarrollo y evaluación.

La [guía ya implementada de recogida y anotación](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/blob/f18a6b026445a0d8cf68ae9e2536d5d7518766d2/docs/PILOTO01_RECOGIDA_HUMANA_Y_ANOTACION_CIEGA_2026-11-07.md) contiene registro, paquetes ciegos, importación, adjudicación, congelado y retirada. La referencia de 150 reseñas/30 componentes sigue siendo planificación; no es un resultado ni un tamaño muestral validado.

## Acceso Kaggle y evidencia que falta

El responsable inicia sesión personalmente, revisa y acepta las reglas si está de acuerdo y descarga solo `travelerinsights2026/train.csv`. No compartir contraseña, token, cookies o `kaggle.json`; no descargar el test oculto. La receta oficial de OAuth y descarga selectiva ya está en el protocolo enlazado. Tras el acceso se necesitan hash del archivo, reglas/fecha, procedencia y derechos de los textos, cabeceras reales y diccionario auténtico/origen de etiquetas. Un código numérico o rating no establece por sí solo el sentimiento textual.

Para una fuente externa alternativa, la evidencia habilitante es una licencia o autorización concreta aplicable a los textos y al uso académico previsto, además del esquema/unidad/origen de etiquetas, privacidad y posibilidad de agrupar reseñas relacionadas. No se envió ninguna solicitud a terceros desde este trabajo. Hasta resolver esas condiciones, los resultados de BETO V2 siguen siendo sintéticos y la validación humana permanece pendiente.
