# PILOTO-01: acceso autorizado y primer benchmark humano

**Decisión operativa, 6 de noviembre de 2026:** intentar Kaggle mediante la cuenta del responsable y preparar en paralelo un corpus de contribuciones originales consentidas. Ninguna fuente está admitida todavía. No se espera indefinidamente una respuesta de una plataforma ni se sustituyen permisos por disponibilidad pública.

Base comprobada: `main=291f6a78e67aba22b9fa44b2c0851214f576617d`; #19 ya fusionado por el responsable. Este incremento propone una opción de salida privada de auditoría y documentación; no entrena modelos, cambia agentes, modifica la DEMO ni repite experimentos históricos.

## 1. Kaggle: intervención exacta del responsable

Abrir [la página Data de la competencia exacta](https://www.kaggle.com/competitions/travelerinsights2026/data) y comprobar que el slug es `travelerinsights2026`.

1. Iniciar sesión personalmente en Kaggle. No enviar contraseña, token, cookies ni `kaggle.json` al proyecto o al agente.
2. Leer [Rules](https://www.kaggle.com/competitions/travelerinsights2026/rules) y, si el responsable acepta sus condiciones, realizar él mismo la aceptación. No se ha aceptado un acuerdo en su nombre. La intervención anterior leyó las reglas renderizadas de esta competencia: contemplan uso académico/no comercial y condiciones adicionales de compartir datos/código. Su lectura no acredita derechos de contenido de terceros.
3. En el explorador de archivos seleccionar **`train.csv`** y usar la descarga de ese archivo. Evitar la descarga conjunta. No descargar, abrir, anotar ni utilizar el test de competencia o entregas ajenas como gold.
4. Guardar el archivo bajo `data/private/piloto01-kaggle/train.csv` en el clon local. Si se recibe un ZIP de ese archivo, verificar que contiene solo la entrada de entrenamiento antes de extraerla; conservar su hash como procedencia.
5. Conservar privadamente la fecha de aceptación, referencia de reglas y documentación disponible de origen y anotación. Si la cuenta no permite aceptar/descargar por cierre de competencia, invitación o permisos, registrar el mensaje concreto y pasar a la ruta voluntaria. No buscar copias que eludan ese acceso.

La consulta de listado de archivos sin credenciales volvió a dar **HTTP 401** en esta intervención. Una petición HEAD al endpoint de `train.csv` dio 404; no permite concluir que el archivo no exista ni sustituye una descarga autenticada. No se leyó su contenido. La ficha web de Data ya confirmó nombres y esquema **declarados**, pero aún no cabeceras o tamaños de un CSV real.

### Opción CLI oficial

La documentación vigente de [autenticación](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) permite OAuth local y la de [descargas](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) permite seleccionar un archivo con `-f`.

Si ya se dispone de la CLI oficial, el responsable puede ejecutar en su propio equipo:

```powershell
kaggle auth login
kaggle competitions download travelerinsights2026 -f train.csv -p data/private/piloto01-kaggle
```

OAuth no sustituye la aceptación de reglas. Comprobar `kaggle --version` y guardar esa versión en el registro local. Para instalar la herramienta, seguir el método aislado `uv tool install kaggle` o `pipx install kaggle` indicado por Kaggle si esas herramientas están disponibles; no se agrega Kaggle a las dependencias del agente. La descarga web permite continuar sin instalar esta CLI.

## 2. Auditoría preparada para `train.csv`

Ejecutar desde la raíz del clon local con Python 3.12 y el entorno del proyecto. No se supone una ruta concreta de Windows. Confirmar primero ubicación y exclusión del archivo:

```powershell
git rev-parse --show-toplevel
git check-ignore data/private/piloto01-kaggle/train.csv
Get-FileHash -Algorithm SHA256 data/private/piloto01-kaggle/train.csv
```

Leer únicamente cabeceras, sin imprimir filas:

```powershell
.\.venv\Scripts\python.exe -c "import csv,json; from pathlib import Path; p=Path('data/private/piloto01-kaggle/train.csv'); f=p.open(encoding='utf-8-sig',newline=''); print(json.dumps(next(csv.reader(f,strict=True)),ensure_ascii=False)); f.close()"
```

**Usar los argumentos siguientes solo si las cabeceras reales coinciden exactamente.** Estos nombres están en la descripción web, no se han verificado dentro del archivo. Adaptar nombres y espacios al esquema comprobado.

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.corpus_metadata_probe `
  --train data/private/piloto01-kaggle/train.csv `
  --text-column "Comentario" `
  --label-column "Sentimiento" `
  --id-column "ID" `
  --group-column "Índice del lugar" `
  --allow-label-value 0 `
  --allow-label-value 1 `
  --allow-label-value 2 `
  --output data/private/piloto01-kaggle/audit_train_v1.json
```

`--output` es la opción nueva de este PR: escribe UTF-8 directamente, no sobrescribe un informe previo y acepta destinos JSON únicamente bajo `data/private` o `reports` desde la raíz actual. En `main` anterior a este incremento, omitir esa opción muestra el informe en la consola local. No hace falta integrar este PR para iniciar sesión o descargar `train.csv`.

Las tres listas de valores autorizan mostrar **códigos crudos**, no asignan significados. Un `0` no se convierte en negativo por esa opción. Si hay otros valores, sus frecuencias permanecen ocultas por valor y se requiere revisar el diccionario original. También puede omitirse toda opción `--allow-label-value` para ocultar todos los valores. Las cabeceras e informes siguen sujetos a revisión antes de compartir.

Esta es una auditoría de entrenamiento solamente: no usa `--test`. La agrupación por índice de lugar es preliminar; antes de dividir datos debe comprobarse si el identificador es global o necesita un espacio de nombres por plataforma. Una clave compuesta `(plataforma,lugar)` identifica una entidad; hotel y autor, en cambio, son dimensiones independientes que se unen por componentes. No agregar `Sitio` como dimensión independiente si eso une todas las reseñas de una plataforma en un solo grupo.

### Evidencia necesaria para admitir Kaggle

| Bloqueo | Evidencia que lo resuelve |
| --- | --- |
| Acceso al archivo | Descarga autorizada, SHA-256 y auditoría del CSV real |
| Significado de códigos | Diccionario o aclaración de la organización para esta competencia; no elegir mapeo observando métricas |
| Origen de etiquetas | Descripción de anotación humana, conversión desde valoraciones o generación automática; distinguir los tres casos |
| Reutilización | Reglas aplicables más base documentada para el uso de los textos originales; no inferir redistribución desde la ficha CC |
| Privacidad y lengua | Revisión local de texto/metadatos, idioma verificado y tratamiento documentado de posibles datos personales |
| Independencia | Identidad de lugar/autor disponible, duplicados/casi duplicados y componentes revisados antes de congelar particiones |

No admitir etiquetas de estrellas o sintéticas como gold humano independiente. Un corpus auténtico con etiquetas proxy puede servir para un experimento proxy separado si sus permisos lo permiten; para el benchmark humano se necesita adjudicación válida del sentimiento textual. Solo el texto de la reseña será una característica: no usar rating, etiquetas, ID o claves de grupo como variables predictoras.

## 3. Alternativas revisadas: disponibilidad no equivale a admisión

| Fuente primaria | Encaje y bloqueo | Decisión actual |
| --- | --- | --- |
| [Trip-MAML](https://github.com/diegma/trip-maml), incluido [README de anotación](https://github.com/diegma/trip-maml/blob/main/EN/README.md) | Declara 500 reseñas españolas anotadas por aspecto/oración. La guía utilizada distingue positivo, negativo y **neutral/mixto**. En el árbol/README revisados no se localizó una autorización precisa de reutilización del corpus. No se descargaron sus archivos de reseñas. | Candidata condicionada; no convertir neutral/mixto en neutral global ni asumir licencia por GitHub público. |
| [OpeNER en SemEval 2022](https://github.com/jerbarnes/semeval22_structured_sentiment) y [corpus original CLTL](https://github.com/cltl/OpeNER_corpus) | Reseñas españolas con expresiones, objetivos y polaridades estructuradas. No es una etiqueta global de tres clases por reseña. No se localizó una licencia precisa para los textos en los árboles/documentación revisados; no se descargaron train/dev/test. | No es sustituto directo. Una futura adaptación exigiría permiso, unidad de análisis y gold global definido. |
| [Rest-Mex](https://sites.google.com/cimat.mx/rest-mex2023/data-and-evaluation) | La página oficial exige registro para recibir acceso. La tarea de polaridad es ordinal de cinco niveles; la licencia CC que aparece para artículos de participantes no autoriza por sí misma el dataset. | No elegirlo como salida inmediata a otro bloqueo de acceso. |
| [COAR](https://huggingface.co/datasets/SINAI/COAR) | Sigue condicionado a aceptación y compartir información de la cuenta; cinco niveles y origen de etiquetas aún no verificado en archivos. | Mantener como candidata, sin aceptar condiciones en nombre del usuario. |

No se declara que estos corpus sean ilegales o inútiles; se identifican condiciones no resueltas. **La alternativa operativa elegida para no depender de esas condiciones es una recogida original consentida**, apoyada en infraestructura ya existente.

UNED se conserva como evidencia de auditoría en el informe anterior; no entra automáticamente en entrenamiento de BETO español.

## 4. Ruta B ejecutable: contribuciones humanas originales en español

Se recoge un texto original sobre una experiencia turística realmente vivida, ofrecido por su autor para entrenamiento/evaluación académica y revisión humana. No se copian reseñas de plataformas ni se generan ejemplos con IA. No se contactó a posibles participantes ni se envió una invitación en esta intervención.

**Meta inicial de planificación:** 150 reseñas de al menos 30 componentes de lugar, una contribución por autor. Son metas de recogida, **no datos ya disponibles ni un cálculo de potencia**. No solicitar una clase concreta al escribir. Revisar cobertura de clases antes del congelado y ampliar la recogida si faltan clases/grupos; una muestra pequeña se describirá como piloto exploratorio.

### Material que debe existir antes de contribuir

- Autorización real del autor, con código seudónimo, versión de las condiciones y referencia de su registro privado. Debe cubrir entrenamiento, evaluación, lectura por anotadores y comunicación de resultados agregados; no publicación de textos por defecto.
- Confirmación de autoría y experiencia real; revisión para evitar nombres de terceros, contactos y otros datos personales innecesarios.
- Una fuente `voluntary` por código de autor, para mantener trazabilidad del permiso; código estable del establecimiento/lugar para agrupar visitas a la misma entidad.
- Una sola contribución por autor en esta primera versión. Si se permiten varias después, unir componentes por autor **o** lugar **o** texto duplicado antes de dividir.

La solicitud de contribución puede explicar: «Aporta, si deseas, una reseña original en español de una visita real a un lugar, alojamiento o restaurante. Describe tu experiencia con tus propias palabras, sin nombres de personas ni contactos. Se utilizará para un proyecto académico de clasificación de sentimiento, con entrenamiento y evaluación de modelos y revisión por anotadores. Se comunicarán resultados agregados; tu texto no se publicará por defecto». Esta descripción no es evidencia de aceptación: conservar la autorización efectiva y la posibilidad de retirar el texto antes del congelado, identificándolo por su código.

### Preparación y entrada en el gestor existente

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli --root data/private/piloto01-human-v1 init
```

Se crean cuatro CSV vacíos sin inventar reseñas: `sources`, `reviews`, `annotations`, `adjudications`. Registrar cada fuente en `sources.csv` con el esquema existente. Ejemplo de **estructura**, que no debe marcarse revisada hasta disponer del permiso real:

| Campo | Contenido esperado |
| --- | --- |
| `source_id` | Código de autor, por ejemplo `author_001` |
| `source_type` | `voluntary` |
| `source_reference` | Referencia a contribución original de ese código |
| `reuse_basis` | `consent` |
| `permission_reference` | Referencia al registro privado de autorización del autor |
| `authorization_reviewed` | `yes` únicamente después de comprobar la autorización efectiva |

Para una contribución ya autorizada guardada en UTF-8, reemplazando los códigos por los reales:

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli --root data/private/piloto01-human-v1 contribute --text-file data/private/piloto01-human-v1/incoming/author_001.txt --source-id author_001 --group-id place_001 --language es --consent-confirmed
```

El comando devuelve un ID de reseña. Dos personas anotan ese mismo ID mediante `corpus_cli ... annotate`, con IDs distintos, sin consultar la otra etiqueta ni una predicción del modelo. Un adjudicador registra `... adjudicate` con justificación. Estos comandos ya existen; ver [el gestor de corpus](CORPUS_TURISTICO_2026.md). Usar `--root data/private/piloto01-human-v1` **antes del subcomando** en todos los pasos. La interfaz privada actual usa otra raíz fija; esta receta aislada emplea la CLI y no modifica esa interfaz.

### Guía de las tres clases y fenómenos

| Etiqueta | Criterio humano |
| --- | --- |
| `positivo` | Evaluación global favorable, explícita o claramente sustentada en el contexto |
| `negativo` | Evaluación global desfavorable, considerando el alcance de negaciones y contrastes |
| `neutral` | Experiencia principalmente descriptiva o sin evaluación global polar clara |
| `desconocido` | Ambigüedad o balance sin predominio global justificable; no forzar a neutral |

Registrar por separado negación semántica, contraste real y opinión mixta en una tabla privada por ID. Las marcas automáticas de `linguistic.py` sirven como indicadores exploratorios, no como gold. Conservar acuerdos/desacuerdos, calcular acuerdo bruto y kappa por pares, y documentar exclusiones. Los ambiguos pueden formar una cohorte de revisión humana separada, sin inventarles una verdad de tres clases.

Esta ruta produce **reseñas humanas elicitadas**, no una muestra representativa de TripAdvisor, Kaggle o Cartagena. No prometer acierto poblacional a partir de ella.

## 5. Ejecución del primer benchmark tras admitir y anotar

El protocolo versionado está en [piloto01_experiment_protocol_v1.json](piloto01_experiment_protocol_v1.json). Su estado es propuesto; no contiene resultados.

1. Completar la admisión, auditar privacidad/duplicados, garantizar dos anotaciones independientes y adjudicación, y fijar componentes. Excluir `desconocido` de clasificación con recuento explícito.
2. Congelar un split 60/20/20 por componentes, semilla 42. El software exige al menos nueve componentes y tres clases por partición; para un piloto útil la meta es mayor, con soporte suficiente por clase. No buscar una semilla por rendimiento.
3. Desarrollar Dummy/LR/SVM en TRAIN/VALIDATION, con TF-IDF aprendido dentro del entrenamiento y calibración SVM en folds de TRAIN separados por grupos. La parrilla existente usa `C={0.5,2.0}` y selección por Macro-F1 de validación.
4. Preparar después el candidato BETO con checkpoint/revisión, tokenizador, truncamientos, presupuesto y selección por validación fijados. Mantener el test propio reservado hasta fijar la comparación. No descargar ni entrenar BETO en esta fase documental.
5. Abrir el test propio una vez para el informe congelado. Si se decide abrir un informe final de baselines antes de fijar BETO, la comparación posterior debe declararse descriptiva o usar un nuevo test independiente; no seleccionar BETO observando errores del test.

Para la raíz nueva, utilizar la función de partición existente con destino aislado. La CLI `split` vigente tiene un destino fijo; esta llamada evita sobrescribirlo. Ejecutar **después** de la admisión y del congelado de anotaciones, desde la raíz del repositorio:

```powershell
if (Test-Path data/processed/piloto01-human-v1) { throw "El destino ya existe; usar una versión nueva y conservar la anterior." }
.\.venv\Scripts\python.exe -c "from pathlib import Path; from traveler_agent.corpus_splits import prepare_splits; r=Path('data/private/piloto01-human-v1'); prepare_splits(r/'reviews.csv',r/'sources.csv',r/'annotations.csv',r/'adjudications.csv',Path('data/processed/piloto01-human-v1'),seed=42)"
```

El runner existente realiza selección y evaluación del test en la misma ejecución. Por ello, el comando siguiente se reserva para el momento de apertura final acordado en el protocolo, no para explorar datos ni decidir particiones:

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.scientific_benchmark --corpus-dir data/private/piloto01-human-v1 --split-dir data/processed/piloto01-human-v1 --output-dir artifacts/piloto01-human-baselines-v1 --seed 42
```

Esta llamada producirá el informe clásico existente de Dummy/LR/SVM; **no produce todavía** todos los análisis adicionales de latencia, fenómenos y abstención descritos abajo. Antes del primer informe final habrá que instrumentarlos sobre las predicciones guardadas, sin cambiar la DEMO ni reconstruir el benchmark. No se presenta esa instrumentación como implementada en este PR.

## 6. Medidas fijadas desde el inicio

| Medida | Procedimiento |
| --- | --- |
| Macro-F1 | Promedio de las tres F1, orden fijo negativo/neutral/positivo; reportar soporte y matriz de confusión. Accuracy secundaria. |
| Errores negativos | Recall negativo; negativos→positivos y negativos→neutrales divididos por el total de negativos reales; además su versión restringida a casos aceptados automáticamente. |
| Negaciones/contrastes | Subconjuntos definidos por anotación humana; reportar tamaño, soporte por clase y errores. No comparar Macro-F1 de tres clases sin advertir clases ausentes en un subconjunto. |
| Latencia | Misma muestra de validación por ID estable, hasta 100 reseñas; lote 1, modelo cargado, cinco pasadas de calentamiento y veinte medidas. Incluir preprocesamiento/vectorización o tokenización e inferencia; p50/p95, equipo, versiones y longitudes. Separar carga, entrenamiento, ASR e interfaz. |
| Abstención | Cobertura = aceptados/N; riesgo = errores entre aceptados/aceptados; tasa de revisión = 1−cobertura. Con cero aceptados, riesgo `null`, no cero. Medir negativos reales enviados como positivos sin revisión. |
| Comparación de políticas | Medir la política actual fija (0.56 y contraste bajo 0.80) sin modificarla; curva offline por umbrales prefijados. Elegir cualquier punto operativo solo con validación y congelarlo antes de test. No asumir probabilidades fiables. |
| Incertidumbre | Remuestreo pareado por componentes independientes de predicciones guardadas; indicar cuando faltan grupos para una conclusión estable. No contar paráfrasis como participantes independientes. |

Guardar hashes de entradas, permisos/consentimientos referenciados, anotaciones y splits; commit, versiones exactas, equipo, semillas, parámetros y luego revisión de BETO. Los resultados de tests de software no son resultados predictivos humanos.

## 7. Estado real y criterio de continuación

- Kaggle: método de acceso identificado; pendiente de sesión/aceptación y archivo autorizado, diccionario de etiquetas y derechos originales.
- Ruta voluntaria: plantillas vacías y comandos preparados; pendientes contribuciones reales, autorizaciones y anotaciones. No se declaran datos recogidos.
- Auditoría/CLI: comandos probados con un fixture sintético separado y 22 pruebas dirigidas; suite integral **202 passed, 1 xfailed** en Python 3.12.14. La CI del nuevo commit se consulta en el PR.
- Ningún corpus admitido, ningún entrenamiento científico nuevo, ningún test de Kaggle abierto y ningún experimento histórico modificado.

La próxima evidencia útil es el informe del `train.csv` autorizado **o** la primera contribución original con permiso documentado y dos anotaciones. Cualquiera de esas vías permite avanzar sin esperar a la otra; no se puede emitir un benchmark humano antes de tener datos y gold reales.
