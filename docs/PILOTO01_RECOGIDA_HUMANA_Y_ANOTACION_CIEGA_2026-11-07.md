# PILOTO-01: recogida original y anotación humana independiente

Base verificada: `main@188f9f7159c38810a4f228e7e467a3e1cfeb1a70`, después de la fusión manual de #22. Este incremento habilita recogida y anotación; **no entrena, no genera un corpus sintético de investigación y no declara rendimiento humano**. V2/V3, agentes, modelo DEMO, Midnight Indigo, Whisper, lotes y exportaciones permanecen intactos.

## Inspección y reutilización

| Componente existente | Reutilización / hueco resuelto |
| --- | --- |
| `corpus.py` / `corpus_cli.py` | Se conservan los cuatro CSV, validadores, contribuciones, anotaciones y adjudicaciones. La nueva capa comprueba el aviso/consentimiento y exige dos dictámenes antes de adjudicar/congelar. El gestor genérico antiguo no garantiza esas condiciones por sí solo. |
| `corpus_annotation_app.py` | Se mantiene el modo privado anterior y se añade modo de paquetes ciegos. La construcción es diferida para que el modo ciego no lea el corpus maestro al importar la UI. |
| `corpus_review.py` | Se reutilizan spans Unicode, kappa y candidatos a casi duplicados. Se añade acuerdo bruto y se separan etiquetas globales de mezcla/ambigüedad/fenómenos. |
| `corpus_splits.py` | Se conserva el split existente. Antes se calculan componentes por autor **o** lugar **o** relación declarada **o** duplicación exacta/cercanía literal. Se exporta una copia con esos componentes como `group_id`. |
| `scientific_benchmark.py` | Puede verificar las particiones de esa copia. No se invoca entrenamiento en este incremento. |

El código nuevo es un adaptador local del gestor, no otro servicio de publicación. Los permisos, originales, paquetes, respuestas y snapshots solo aceptan rutas resueltas bajo `data/private` desde la raíz del repositorio. Se rechazan escapes por enlaces simbólicos. No se suben archivos humanos como artefactos de CI.

## Estado real comprobado

El [registro de disponibilidad](human_intake_readiness_2026-11-07.json) informa **0 consentimientos registrados, 0 reseñas humanas, 0 anotaciones y 0 adjudicaciones** en la raíz de recogida. Solo hay plantillas vacías. La generación de un formulario o de fixtures de prueba no registra una persona ni constituye consentimiento.

`train.csv` de Kaggle no está disponible en su ruta autorizada local. El lector público de Data/Rules no devolvió contenido auditable en esta comprobación. No se usó una sesión autenticada, no se descargó entrenamiento ni se accedió al test oculto. El HTTP 401 registrado el 6 de noviembre es evidencia histórica, no un resultado HTTP nuevo que se afirme aquí.

## 1. Preparación del coordinador

Desde la raíz del clon, con Python 3.12 y las dependencias existentes. En PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[ui,test]'
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli init
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli status
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli participant-form --out data/private/delivery/participant-001.html
```

`init` reutiliza `data/private/piloto01-human-v1` y no reemplaza CSV existentes. `--root` permite otra raíz privada y se coloca **antes** del subcomando. En Linux/WSL usar `.venv/bin/python`. El HTML se abre como archivo local; no necesita servidor ni conexión, no envía formularios, no usa analítica ni guarda borradores. Genera un código de autor y muestra el aviso completo. El participante decide marcar cada aceptación y descargar un JSON de consentimiento y su texto TXT; cerrar el formulario no registra nada.

El coordinador entrega personalmente el aviso/formulario por un canal privado acordado y recibe ambos archivos por ese canal. No se ha contactado a participantes desde este incremento. No publicar los enlaces o archivos en el repositorio. Entregar **un formulario/código por persona**, conservar el código estable y mantener la correspondencia bajo custodia privada; no asignar otro código al mismo autor para simular independencia. Para el piloto inicial se recomienda una contribución por autor, como en el protocolo previo; si se admiten varias, deben estar cubiertas por autorización efectiva y conservar el mismo código.

El aviso versionado `PILOTO01-CONSENT-1` cubre lectura/anotación humana, entrenamiento y evaluación académicos, y resultados agregados. No autoriza publicar los textos por defecto. No pide nombre, correo, teléfono, edad o dirección. El canal del coordinador y el código permiten consultas/retirada. El responsable debe explicar custodia y acceso reales antes de recoger; las declaraciones no autentican identidad, autoría o derechos automáticamente. No admitir un texto copiado de una plataforma como una contribución original.

Alternativa sin HTML: `consent-template --out data/private/delivery/consent.json` crea un formulario JSON con todas las aceptaciones en `false`, sin aprobación. `notice` muestra el aviso y la guía. El JSON debe reflejar la aceptación efectiva del autor, no marcarse por conveniencia del operador.

## 2. Registrar consentimiento y texto auténticos

Guardar los archivos efectivamente recibidos en `data/private/incoming`. Sustituir el código por el del consentimiento real. `coord01` y `place_001` son códigos de ejemplo, no personas/lugares ya registrados:

```powershell
$authorCode = 'COPIAR_CODIGO_REAL_DEL_AUTOR'
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli register-consent --consent-file "data/private/incoming/${authorCode}-consent.json" --reviewed-by coord01
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli contribute --text-file "data/private/incoming/${authorCode}-review.txt" --author-id $authorCode --place-id place_001 --reviewed-by coord01 --privacy-reviewed --spanish-verified --original-human --real-experience
```

El coordinador debe comprobar antes de marcar las opciones: consentimiento correspondiente al aviso y finalidades, autoría original, experiencia vivida, español y ausencia de datos personales innecesarios. Si hacen falta cambios por privacidad, pedir una versión al autor; no corregir automáticamente su estilo, faltas, negaciones o valoración. El gestor recorta espacio externo como antes y conserva el contenido para anotarlo; la normalización se usa solo para claves derivadas de duplicados.

Se archiva una copia JSON normalizada de las aceptaciones en `permissions/`, su hash y referencia, con código del revisor. El texto entra mediante el comando de contribución ya existente y se añade un enlace privado autor/lugar/hash. No se imprimen textos o permisos en la respuesta del comando. Un formulario incompleto o de otra versión no registra la fuente.

Mantener un código estable para **el mismo establecimiento/lugar** en todas las aportaciones. No usar una clave compuesta autor+lugar como sustituto de ambas dimensiones: el grafo las une por separado. Si se sabe que dos reseñas son variantes, coautorías, parte de la misma experiencia o textos relacionados:

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli link-related --left r_CODIGO_REAL_1 --right r_CODIGO_REAL_2
```

También existe `contribute --related-to r_CODIGO_REAL`. Las relaciones se revisan manualmente antes del congelado; no se infiere toda paráfrasis o identidad real del establecimiento.

## 3. Dos anotadores separados

Asignar dos personas distintas, sin consultar predicciones, estrellas o etiquetas ajenas. No usar al autor de la reseña como anotador de su propia experiencia. Los códigos no prueban por sí mismos que sean dos personas: el coordinador debe organizar esa independencia.

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli packet --annotator ann01 --limit 25 --out data/private/delivery/ann01-batch01.json
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli packet --annotator ann02 --limit 25 --out data/private/delivery/ann02-batch01.json
```

Repetir por lotes según pendientes. El límite es 1–100. Cada paquete contiene únicamente guía, código del anotador e IDs opacos con texto; no contiene autor, lugar, fuente, calificación, predicciones o anotaciones previas. Cada persona recibe solo su paquete y usa una copia/directorio propios. El coordinador conserva el índice de correspondencias. Los IDs y el orden de items difieren entre paquetes. Solo se reservan dos plazas por reseña; una tarea pendiente puede cancelarse con `cancel-task --task-id task_CODIGO_REAL`, liberando la plaza. Una tarea ya recibida no se cancela para borrar dictámenes.

En el equipo de cada anotador, copiar su paquete bajo `data/private` en el clon y ejecutar:

```powershell
.\.venv\Scripts\python.exe corpus_annotation_app.py --packet data/private/annotation-work/ann01-batch01.json --responses data/private/annotation-work/ann01-responses01.json
```

Se abre Gradio en `127.0.0.1`, `share=False`, sin acceso al maestro desde ese modo. Las respuestas son borradores propios editables hasta pulsar **Finalizar lote**, tras completar todos los items y declarar anotación humana independiente. Entregar solo el archivo finalizado al coordinador por el canal privado. No abrir la UI antigua sin `--packet` para este proceso: sigue existiendo como herramienta genérica, no como garantía de cegamiento.

| Campo | Criterio |
| --- | --- |
| `positivo` / `negativo` | Evaluación global favorable/desfavorable sustentada por el contexto, no por una palabra aislada. |
| `neutral` | Descripción sin evaluación polar global clara. |
| `mixed=true` | Opiniones opuestas sobre uno o varios aspectos. Puede coexistir con positivo/negativo si hay predominio justificable. |
| `ambiguous=true`, `desconocido` | No se puede resolver valoración global; la mezcla sin predominio no se fuerza a neutral. |
| `negation` / `contrast` | `true` presente, `false` ausente, `null` incierto. Se anotan semánticamente; su incertidumbre no obliga a cambiar el sentimiento global. |
| Aspectos / spans | Opcionales, usando el vocabulario existente. Offsets Unicode Python `[start,end)` del texto almacenado, sin cambiarlo. |

La UI no preselecciona sentimiento y comienza los fenómenos como inciertos. Las respuestas no contienen el texto completo, aunque los paquetes sí lo contienen y siguen siendo privados. La guía, el hash del paquete, su identidad y su correspondencia con cada reseña se verifican al importar. No se aceptan paquetes alterados, respuestas parciales, no finalizadas o adjudicación con un solo anotador.

Alternativa CLI: `judgment-template --out ...`, editar un dictamen local, `answer --packet ... --responses ... --item-id ... --judgment-file ...` y después `seal --packet ... --responses ... --independent-human`. No introducir textos/justificaciones sensibles como argumentos de shell; se emplean archivos privados.

## 4. Importación y adjudicación

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli import-answers --responses data/private/returned/ann01-responses01.json
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli import-answers --responses data/private/returned/ann02-responses01.json
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli status
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli adjudication-packet --review-id r_CODIGO_REAL --out data/private/adjudication/review01.json
```

La importación valida todo el lote antes de escribir anotaciones, reutiliza el registrador existente y conserva las marcas de fenómeno. Reimportar las mismas respuestas finalizadas es idempotente y puede completar una importación interrumpida; otra versión no sobrescribe dictámenes.

La tercera persona recibe el archivo de adjudicación solo después de los dos dictámenes. Completa `adjudicator_id`, `judgment` y `reason`; no altera texto ni anotaciones originales. Debe justificar la decisión, también si acepta el acuerdo del par. Puede conservar `desconocido` y ambigüedad; no hay voto o adjudicación automática.

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli adjudicate --decision-file data/private/adjudication/review01.json
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli status
```

El estado informa clases adjudicadas, casos mixtos/ambiguos, fenómenos presentes/inciertos, acuerdo bruto y kappa por pares sobre sus dictámenes iniciales compartidos. Kappa es `null` cuando es indefinido; ni acuerdo alto ni mayoría demuestran gold correcto. No se publican esos resúmenes automáticamente, sobre todo con grupos pequeños.

## 5. Congelado y futura comparación justa

Solo con consentimiento vigente, dos dictámenes finalizados por reseña activa y adjudicación completa:

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli freeze --out data/private/piloto01-human-snapshots/v1 --reviewed-by coord01 --links-reviewed
.\.venv\Scripts\python.exe -m traveler_agent.human_intake_cli verify-snapshot --snapshot data/private/piloto01-human-snapshots/v1
```

`--links-reviewed` declara revisión real de identidad de autor/lugar y textos relacionados. Se conectan por **OR** esas dimensiones, hashes exactos normalizados y todos los candidatos literales del detector existente (umbral 0.88). Agrupar candidatos conservadoramente no los elimina ni prueba independencia semántica; falsos positivos pueden producir componentes grandes. El algoritmo existente limita la auditoría a 2000 reseñas por ejecución.

La copia privada `corpus/` mantiene los esquemas del gestor; solo cambia `group_id` por componente. `splits/` usa el algoritmo existente 60/20/20 por componentes, semilla 42, con al menos nueve componentes y tres clases por partición. Son requisitos técnicos, **no validación del tamaño muestral**. La meta de 150 reseñas/30 componentes sigue siendo planificación, no dato logrado ni cálculo de potencia. No buscar una semilla por rendimiento. `desconocido` se conserva en la copia y se excluye del benchmark de tres clases con recuento explícito.

El snapshot preserva hashes de consentimientos, entradas, pruebas de anotación/adjudicación, particiones y fenómenos humanos. La verificación falla si se retira consentimiento o cambia el corpus/prueba después. Si se interrumpe el congelado no se entrega un snapshot parcial. No se entrena nada al congelar o verificar.

El runner existente `scientific_benchmark` puede consumir `snapshot/corpus` y `snapshot/splits`, pero **todavía no debe ejecutarse para obtener resultados humanos inexistentes ni abrir un test antes de fijar la comparación**. Dummy/LR/SVM y, si es viable, BETO deben usar las mismas filas/particiones, etiquetas, texto como única característica y evaluación. Ni autor, lugar, ID o rating son variables predictoras. Para BETO falta fijar cabeza/pooling, entrenamiento, presupuesto y recursos reales; la inferencia CPU demostrada en #21 no mide ajuste supervisado. UNED no entra automáticamente en BETO.

Se conserva el [protocolo humano propuesto](piloto01_experiment_protocol_v1.json) como antecedente. Esta capa implementa recogida, cegamiento y relaciones adicionales; no convierte sus apartados pendientes de latencia, costos, calibración y BETO en resultados ya realizados. La futura instrumentación reutilizará predicciones guardadas para Macro-F1, clase/confusión, recall negativo, negativos→positivos, negaciones/contrastes humanos, p50/p95 y costos de pared/CPU/RSS/modelo. Los fenómenos inciertos no contarán como ausencia ni como gold de una rebanada.

## 6. Abstención: investigación pendiente y condiciones de admisión

No se modifica ningún umbral de la DEMO ni se usa assessment V2/V3 para seleccionar técnicas. Se revisaron [Selective Classification](https://arxiv.org/abs/1705.08500v2) y [Learn then Test](https://arxiv.org/abs/2110.01052v5): riesgo/cobertura y validación estadística de una familia de decisiones son problemas distintos de tomar la confianza como verdad. No se implementan aquí sus garantías ni se trasladan automáticamente a reseñas agrupadas.

Antes de cualquier candidato: fijar la familia de políticas, pérdidas, criterios y tratamiento de múltiples comparaciones; separar por componentes tune, calibración, elección de política y evaluación independiente. Las particiones externas actuales son solo la base del benchmark: no proporcionan por sí solas tres nuevos subconjuntos de desarrollo con soporte suficiente. Si no pueden subdividirse sin perder clases/grupos, recoger desarrollo adicional o fijar un diseño anidado por grupos antes de abrir evaluación.

Como objetivos **propuestos**, no parámetros desplegados: cobertura mínima 25%, riesgo entre aceptados ≤10%, tasa de negativos→positivos automáticos ≤5% sobre todos los negativos y derivación ≥80% de la cohorte adjudicada ambigua. Se informarán también números de aceptados, negativos y componentes, intervalos apropiados y errores por ID. Los valores y el tamaño necesario deben quedar fijados con datos de desarrollo y precisión estadística justificada, no con V2/V3 o el futuro test.

Ejemplo puramente matemático: con decisiones Bernoulli independientes y una política ya fijada, 0/10 errores aún admite un límite superior unilateral del 95% de aproximadamente 25.9%; para bajar de 5% con cero errores harían falta 59 observaciones independientes bajo esos supuestos. No son 59 reseñas cualesquiera: opiniones del mismo autor/lugar no deben contarse como independientes, y seleccionar entre muchas políticas requiere control adicional. Esto **no es un cálculo de potencia para nuestra muestra**. Con datos actuales inexistentes no hay política candidata evaluable ni soporte que certificar.

«Revisar todo» puede impedir automatizar, pero no satisface una meta operativa de cobertura; se declarará **NO APTA PARA AUTOMATIZAR** si no hay política con cobertura/riesgo respaldados. Una ECE menor tampoco bastará: se medirán NLL/Brier/fiabilidad y riesgo selectivo por separado. La validación humana sigue pendiente.

## 7. Kaggle: acción personal y evidencia requerida

Abrir [Data de la competencia exacta](https://www.kaggle.com/competitions/travelerinsights2026/data), iniciar sesión personalmente y revisar [Rules](https://www.kaggle.com/competitions/travelerinsights2026/rules). Si se aceptan las condiciones, realizar esa aceptación personalmente y descargar **solo `train.csv`**. No se solicitaron contraseñas, tokens o cookies. No se necesita instalar Kaggle dentro de la aplicación.

La documentación oficial vigente permite [OAuth local](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) y [descarga de un archivo](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md):

```powershell
kaggle auth login
kaggle competitions download travelerinsights2026 -f train.csv -p data/private/piloto01-kaggle
```

No omitir `-f`: la descarga conjunta puede incluir datos que no necesitamos. Si llega un ZIP, comprobar sus entradas antes de extraer únicamente entrenamiento. Si el acceso falla, conservar privadamente el mensaje preciso y compartir solo una descripción sin datos de cuenta; no buscar copias que eludan permisos.

Tras descargar: conservar hash SHA-256, versión CLI, referencia/fecha de reglas y documentos de origen/etiquetado/derechos. Leer primero solo cabeceras y ejecutar la receta de [auditoría preparada](PILOTO01_ACCESO_Y_BENCHMARK_HUMANO_2026-11-06.md), adaptada al esquema real. No inferir 0/1/2 observando resultados. La admisión exige diccionario auténtico, origen de etiquetas, permiso de reutilización de los textos, privacidad/español y agrupación por autor/lugar/duplicados. Un proxy de estrellas no es automáticamente gold textual humano. No registrar Kaggle como `voluntary` para saltar esa auditoría.

## 8. Límites operativos y pruebas

El sistema es local y para un coordinador. Usa bloqueo de escritor y reemplazos atómicos por fichero, no una transacción global de todos los CSV. Una interrupción de contribución puede dejar trazabilidad incompleta: la validación bloquea hasta recuperar una copia privada coherente o reconciliar los archivos con revisión. No reintentar creando otra reseña ni borrar el original para ocultarlo. Las respuestas sí permiten reimportación idempotente para completar escrituras interrumpidas.

Proteger acceso al equipo, copias y canal privado: Git ignorado no es cifrado ni autenticación de personas. Los códigos, consentimientos y límites declarados no prueban autoría, experiencia real o anonimato total. Revisar PII antes de entregar textos a anotadores. No publicar la UI, archivos o hashes derivados sin revisión. El coordinador debe mantener copias privadas coherentes y gestionar su retención.

`withdraw --author-id author_CODIGO_REAL` bloquea nuevos paquetes/importaciones y futuros snapshots de ese autor; `verify-snapshot` rechaza los consentimientos cambiados. No borra automáticamente copias ya entregadas, snapshots anteriores, respaldos o modelos: deben localizarse y revisarse con el responsable. El aviso no promete borrado retroactivo automático.

Las pruebas usan exclusivamente fixtures artificiales y cubren consentimiento incompleto/alterado, rutas y symlinks, paquetes sin etiquetas ajenas, independencia declarada, asociación persona/texto, neutralidad/mezcla/ambigüedad, importación parcial, tercera adjudicación, acuerdo, spans Unicode, componentes y snapshot/retirada. La CI ejecuta además los handlers JavaScript reales del formulario con un DOM simulado; no usa personas ni cuentas reales y no sube archivos privados. La limitación histórica `test_negation_expected_positive_is_known_limitation` sigue como **xfail estricto**, sin tocar el modelo DEMO. Consultar el resultado efectivo de CI en el PR.
