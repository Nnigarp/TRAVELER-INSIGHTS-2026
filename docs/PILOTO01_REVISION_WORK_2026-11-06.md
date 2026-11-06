# PILOTO-01 — Revisión técnica de PR #18 y acceso real a fuentes

**Fecha:** 6 de noviembre de 2026. **Dictamen del cambio original: CORREGIBLE.**

Esta es una revisión del código y de fuentes inspeccionadas, no una validación predictiva. El PR #18 ya estaba fusionado cuando comenzó la revisión. Las correcciones de este incremento se proponen en una rama independiente; no se modifica `main` ni se fusionan PR.

## A. Estado comprobado de GitHub

| Elemento | Resultado observado |
| --- | --- |
| `main` | `06a14454d9b34ba122ef7094a3912e70c669ed76` |
| PR #17 | Fusionado; acta de aceptación funcional reportada por el estudiante |
| PR #18 | Fusionado el 2026-11-06 a las 06:10:03 America/Bogota |
| Head revisado del PR #18 | `60f88a3f8c62a142d489daddbe2ac25d1763d7c9` |
| Base del PR #18 | `8a3576d0cfba8e0af910587820b91616f2236fc2` |
| Diff original | 4 archivos nuevos, 468 líneas añadidas, 0 eliminadas |
| PR abiertos al iniciar | 0; los PR #12, #13 y #15–#18 estaban fusionados |
| CI de `main` | [Run 42208199342](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/42208199342), `success`, Python 3.12: 185 passed, 1 xfailed |
| CI del PR #18 | [Run 42207317646](https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/42207317646), `success`: 185 passed, 1 xfailed |
| Conflictos de #18 | Ya fusionado; GitHub devuelve `mergeable=null`, no una comprobación pendiente de un PR abierto |

Los conteos se comprobaron en logs de jobs, no se dedujeron del color del workflow. La reproducción local del estado original dio **185 passed, 1 xfailed** en Python 3.12.14. El `xfail` es el caso conocido de negación del modelo DEMO. La CI configurada usa Ubuntu y Python 3.12: no demuestra una ejecución nueva en Windows. En este entorno hubo que instalar `socksio` únicamente en el entorno virtual para el proxy usado por `httpx`; no se cambió una dependencia del proyecto por ese detalle del entorno.

## B. Arquitectura y módulos conservados

| Responsabilidad | Código existente | Comportamiento verificado |
| --- | --- | --- |
| Interfaz y sesión | `app.py`, `multiagent_view.py` | Gradio Midnight Indigo, entradas individual/lote/audio, corrección y traza |
| Coordinación | `multiagent.py`, `agent.py` | `TravelerOrchestrator` delega un plan fijo a cuatro especialistas, valida contratos y corta fallos |
| Percepción | `PerceptionAgent`, `audio_asr.py` | Valida texto/modalidad; ASR local ocurre antes de la delegación y permite edición |
| Semántica | `SemanticsAgent`, `model.py`, `aspects.py`, `linguistic.py` | TF-IDF de palabras/caracteres con LR/SVM; aspectos y señales por reglas |
| Control y acción | `ControlAgent`, `ActionAgent`, `policy.py` | Revisión humana por puntuación/contraste y recomendaciones informativas |
| Lotes y revisión | `batch.py`, `corpus_review.py`, `corpus_annotation_app.py` | Exportaciones y correcciones; no entrenamiento continuo |
| Investigación | `corpus.py`, `corpus_splits.py`, `scientific_benchmark.py`, adaptadores de fuentes | Procedencia, anotación, particiones y Dummy/LR/SVM separados de la DEMO |
| Experimentos históricos | `synthetic_*`, `out_of_generator_challenge.py`, `evaluate_out_of_generator.py` | Se conservan sin cambios ni ajuste contra sus tests |

La memoria es temporal de ejecución/sesión. El orquestador no genera planes generales ni actúa sobre servicios externos. El SVM contiene calibración interna, pero no hay fiabilidad probabilística demostrada sobre turismo auténtico. La revisión humana en la interfaz no reentrena modelos.

## C. Revisión de los cuatro archivos del PR #18

| Archivo | Evaluación |
| --- | --- |
| `docs/PILOTO01_AUDITORIA_DATOS_2026-11-06.md` | Distingue metadatos de fichas, archivos no inspeccionados y propuesta de partición. Su afirmación de salida sin textos es demasiado fuerte para el código original. La evidencia de acceso era un registro de su sesión; hay nuevos accesos en esta revisión. |
| `docs/piloto01_candidate_manifest.json` | Mantiene `NO_CORPUS_ADMITTED`, selección y hashes nulos, mapeo no resuelto y partición `NOT_EXECUTED`. Es un manifiesto de candidatos, no una autorización ni un dataset. Se conserva como registro inicial. |
| `src/traveler_agent/corpus_metadata_probe.py` | Solo biblioteca estándar, lectura local y salida JSON; sin red, entrenamiento, pickle/joblib, subprocess, escritura de reseñas ni integración con la app. Se encontraron los fallos siguientes. |
| `tests/test_corpus_metadata_probe.py` | Cinco funciones originales con fixtures sintéticos. Cubren casos habituales, pero omiten roles de columnas incorrectos, agrupación por varias dimensiones, filas sin texto y comillas sin cerrar. |

Referencias de línea en el **head original** de #18, no en el archivo corregido:

| Prioridad | Hallazgo y evidencia reproducida | Efecto | Corrección propuesta |
| --- | --- | --- | --- |
| P1 | Líneas 71–80 y 102–103: seleccionar `label_column=text_column` devuelve la reseña como clave del histograma; lo mismo puede suceder con idioma. Reproducción con texto sintético privado. | Una salida prometida como metadatos puede revelar el texto en terminal o reportes. No se observó exposición de una reseña auténtica. | Rechazar roles texto/etiqueta/idioma coincidentes; ocultar valores categóricos por defecto y exigir listas explícitas de códigos revisados para mostrarlos. |
| P2 | Líneas 86–93: con `hotel,author` se calcula una pareja; un hotel compartido con autores diferentes arroja solapamiento 0. | La auditoría puede pasar por alto dependencia por hotel o autor. | Contar cada dimensión por separado, con nombres de columna en el hash, y reportar solapamientos por dimensión y su unión. No afirmar que son componentes conectadas. |
| P2 | Líneas 65–67: `continue` ocurre antes de contar etiquetas, IDs, idiomas y grupos. Una fila sin texto y otra con el mismo ID arrojan 0 duplicados; una etiqueta vacía en la primera tampoco se cuenta. | Conteos incompletos que pueden ocultar problemas de calidad. | Contar metadatos en todas las filas y restringir únicamente fingerprints/longitud a textos no vacíos. |
| P2 | Línea 54: `DictReader` sin modo estricto acepta un campo entrecomillado sin cierre. | Se presenta como inspeccionado un CSV malformado. | `strict=True`, conservando soporte para multilínea válido y UTF-8 BOM. |
| P2 | Línea 169: el mensaje de `OSError` se imprime completo. | Puede mostrar una ruta privada, nombre de cuenta o montaje. Hallazgo por inspección; prueba de regresión con error simulado. | Emitir solo el tipo de error y una instrucción genérica. |

También se comprueba el hash antes y después de la lectura para rechazar un archivo que cambie durante la inspección. No es una certificación criptográfica de anonimato. Cabeceras, valores autorizados y reportes locales requieren revisión antes de compartir.

**Compatibilidad del informe:** `piloto01_metadata_probe_v2`. La firma conserva parámetros anteriores y añade listas opcionales `allowed_label_values` y `allowed_language_values`. Los consumidores que necesiten códigos crudos deben proporcionar esas listas. Sin ellas se muestran frecuencias sin sus valores. La agrupación múltiple cambia expresamente de pareja compuesta a dimensiones independientes; con una sola columna conserva su interpretación.

**Pruebas de la corrección:** 16 casos aprobados en `test_corpus_metadata_probe.py`, frente a cinco casos originales. Incluyen pruebas de CLI, ocultación de celdas arbitrarias, un hotel compartido con autor distinto/vacío, ausencia de coincidencias espurias entre dimensiones, conteos de filas vacías, CSV truncado, hash cambiante, UTF-8 BOM y texto multilínea válido. La suite integral local dio **196 passed, 1 xfailed**, Python 3.12.14 (scikit-learn 1.9.1, Gradio 6.31.0, pytest 9.2.0). El resultado de CI del nuevo commit debe consultarse en el PR; no se infiere de la ejecución local.

## D. Dictamen

**CORREGIBLE.** El aislamiento del incremento es adecuado y su política de no admisión es honesta. Los fallos de salida y de conteo impiden recomendar la herramienta original sin corrección para reseñas privadas. Como #18 ya estaba fusionado, se propone un PR posterior; no hay motivo para reescribir su historia ni alterar la DEMO.

El diff original no toca `app.py`, orquestador, LR/SVM, política, ASR, `pyproject.toml`, workflows ni experimentos históricos. La corrección conserva ese aislamiento. Que los tests pasen no demuestra Macro-F1 sobre reseñas humanas.

## E. Acceso y bloqueadores de los corpus

La evidencia agregada y hashes están en [piloto01_source_access_2026-11-06.json](piloto01_source_access_2026-11-06.json). No se incluyen reseñas, XML/ZIP, CSV, IDs de huéspedes ni datos demográficos.

### Kaggle: reglas y descripción ahora verificadas en la página exacta

Se leyó en navegador la página [Rules de travelerinsights2026](https://www.kaggle.com/competitions/travelerinsights2026/rules), sin sustituir reglas de otra competencia. Sus apartados 1.7 y 2.4 contemplan uso académico y no comercial. Las reglas fundacionales incluyen condiciones de compartir datos/código y prohíben anotación manual de registros de validación/test para entregas. No se aceptaron condiciones ni se inició sesión.

En [Data](https://www.kaggle.com/competitions/travelerinsights2026/data) se vieron `train.csv` y el esquema **declarado**, con `Comentario`, `Sentimiento` e identificador del lugar. La página exige sesión y aceptación para leer el contenido; la API pública de listado devolvió HTTP 401. Por tanto, no hay conteos, cabeceras **de archivo** ni hashes Kaggle verificados. La ficha CC BY-SA no sustituye esas condiciones ni acredita derechos originales. El mapeo negativo sigue contradictorio en Overview. No se accedió a registros o etiquetas del test de Kaggle.

### UNED: archivo obtenido y auditado realmente

Se obtuvo el ZIP del enlace ofrecido por la [página oficial](https://nlp.uned.es/~lplaza/resources/hotel-review-corpus/). La página y el README ofrecen uso para investigación y piden citar el trabajo; no se encontró autorización explícita de redistribución de textos ni se acreditaron los derechos sobre las reseñas originales de Booking.com. La inspección privada del paquete no lo admite automáticamente como fuente de entrenamiento.

| Propiedad medida | Resultado |
| --- | --- |
| ZIP | 122.221 bytes; SHA-256 `5f1996860bc475363b684b59b4a61ac8a5c3a9bb3861be040bc23ba724b895bb` |
| XML original | 580.048 bytes; SHA-256 `14d6711dcab104f292b6880ce4f9b185c9bafef9250dacadf38ecda1c8a17eed` |
| Formato original | No válido para parser XML estricto: atributos `3_classes_intensity` y `5_classes_intensity` empiezan por número |
| Transformación de inspección | Renombrar solo esos atributos de apertura, 1.000 apariciones de cada uno, en memoria; ZIP/XML originales intactos |
| Unidad | Una reseña = segmentos `PositiveOpinion` y `NegativeOpinion` concatenados con salto de línea; no se convierten los segmentos en etiquetas de sentimiento |
| Conteo real | 1.000 registros; 88 claves `hotelID` |
| Etiquetas originales de tres categorías | `Fair=292`, `Good=359`, `Poor=349`; sin mapear a español |
| Cinco categorías | 200 de cada una: `Excellent`, `Fair`, `Good`, `Poor`, `Very_Poor` |
| Inspección mediante herramienta corregida | 0 textos vacíos, 0 etiquetas vacías, 0 IDs duplicados, 0 duplicados exactos normalizados, 0 conflictos de etiqueta para el mismo texto |
| Idioma | No existe campo de idioma. Un diagnóstico por listas de palabras da predominio inglés en 994 registros y empate en seis; **no es certificación de idioma ni distribución lingüística validada** |
| Privacidad | Se descartaron de la conversión nacionalidad, categoría de viajero, ubicación, fecha y score. El texto libre todavía necesita revisión de PII. |

El README y la página afirman anotación manual global; esa procedencia declarada se distingue de una revisión independiente de su calidad. Los campos de segmentos positivos/negativos no equivalen por sí mismos a un gold global de tres clases. No se entrenó ni se dividió el corpus. La sospecha de idioma inglés impide asumir que resuelva el piloto español.

### Otras candidatas

| Fuente | Evidencia y bloqueo actual |
| --- | --- |
| [SINAI/COAR](https://huggingface.co/datasets/SINAI/COAR) | Ficha primaria: 2.202 opiniones españolas de restaurantes, cinco niveles y CC BY-NC-SA. El acceso requiere aceptar condiciones que comparten información de la cuenta. No se aceptaron; no se leyeron CSV ni se verificó si las etiquetas son anotaciones independientes o ratings. |
| [Hopinion UB](https://clic.ub.edu/corpus/es/hopinion) y [página del autor](https://johnrbto.com/hopinion/) | Descripción de 17.934 opiniones y metadatos personales; la página del autor enlaza descarga en Mega. No se identificó una licencia precisa ni un gold global independiente. No se descargó; la consulta directa a UB dio HTTP 503. |

**Estado vigente: `NO_CORPUS_ADMITTED`.** Conseguir un archivo no elimina los bloqueos de derechos originales, idioma, etiquetas, privacidad e independencia de grupos.

## F. Siguiente acción ejecutable y protocolo propuesto

1. Resolver acceso autorizado a **solo `train.csv`** de Kaggle y disponer de documentación fiable del origen de etiquetas y del mapeo. El inicio de sesión y la aceptación legal quedan pendientes del usuario. No hacen falta credenciales en GitHub ni enviar correos a autores.
2. En un entorno autorizado, confirmar cabeceras y ejecutar la herramienta corregida sobre la copia privada. Sin valores autorizados, los códigos quedan ocultos. Elegir las listas de códigos solo controla su visualización; no adivina significado. No utilizar test de competencia.
3. Registrar una decisión de admisión por fuente con evidencia de derechos, anotación, idioma y privacidad. Si Kaggle sigue sin gold fiable, la alternativa operativa ya disponible es el corpus propio consentido con `corpus_cli.py` y `corpus_annotation_app.py`; no hace falta construir otra app. Una fuente con ratings solo puede sostener una evaluación proxy claramente identificada, o requiere anotación humana independiente de datos autorizados.

Tras la admisión, fijar antes de entrenar:

| Aspecto | Diseño propuesto, todavía no ejecutado |
| --- | --- |
| Pregunta | Comparar representaciones sobre sentimiento global de reseñas turísticas españolas y riesgo al automatizar su salida |
| Gold | Guía de tres clases que distinga neutralidad de ambivalencia; dos anotaciones independientes y adjudicación cuando se produzcan nuevas etiquetas; acuerdo y desacuerdos registrados |
| Independencia | Unir por hotel/lugar, autor cuando sea legítimo y esté disponible, y duplicados exactos; revisar casi duplicados. No agrupar todo un corpus como un único `source_id`. Para evaluación entre fuentes, reservar fuentes completas en un análisis separado. |
| Partición | Propuesta 60/20/20 por componentes, semilla 42; verificar viabilidad de clases/grupos antes de abrir test. El splitter existente reparte componentes, no garantiza esos porcentajes de filas ni estratificación. |
| Baselines | Dummy mayoritario, TF-IDF+LR y TF-IDF+SVM con las mismas particiones. Vocabulario y cualquier ajuste se aprenden solo en TRAIN. |
| Selección/calibración | Hiperparámetros y umbrales se seleccionan con validación. Calibración dentro de folds de TRAIN separados por grupos; mismos criterios para candidatos contextuales. No tratar scores como probabilidades validadas sin evaluarlas. |
| Métricas | Macro-F1 principal; accuracy secundaria; precisión/recall/F1 y soporte por clase; matriz con orden fijo [negativo, neutral, positivo]; tasa negativo→positivo |
| Robustez | Subconjuntos humanos anotados de negación, contraste, ambivalencia y variantes lingüísticas; ejemplos mínimos creados después del sellado van a un conjunto separado y no se cuentan como reseñas auténticas independientes |
| Revisión humana | Comparar política DEMO fija con una política ajustada solo en validación; riesgo/cobertura, errores entre casos automáticos y negativos enviados erróneamente como positivos; ECE/Brier si se evalúa calibración |
| Incertidumbre | Intervalos y comparaciones pareadas mediante remuestreo de grupos, no de paráfrasis o filas relacionadas; reportar si hay pocos grupos para una conclusión estable |
| Costos | Tiempo de entrenamiento e inferencia, p50/p95 por reseña, CPU/GPU, RAM y tamaño del modelo; mismo equipo, lotes y calentamiento; longitud y truncamientos para Transformers |
| Reproducibilidad | SHA de entradas/particiones, versión de anotaciones, commit, dependencias exactas, revisión del checkpoint/tokenizador, semillas y parámetros; presupuesto de ensayos fijado antes de comparar |
| Test | Sellar pertenencia y etiquetas, predefinir candidatos/configuraciones y hacer una comparación final congelada; no mirar errores del test para volver a ajustar. Un estudio adaptativo posterior necesita un test nuevo. |

Se reutiliza `scientific_benchmark.py` después de verificar el adaptador/esquema y la admisión; no se ejecutan los comandos simples de train sobre la totalidad de datos antes de reservar el test. No hay porcentajes ejecutados ni una mejora predictiva nueva.

## G. Hoja de ruta priorizada

| Orden | Técnica | Condición de entrada y utilidad comprobable |
| --- | --- | --- |
| 1 | Datos y Dummy/LR/SVM | Fuente admitida, protocolo sellado y evaluación auténtica reproducible |
| 2 | BETO | Candidato contextual con checkpoint fijado; comenzar por una configuración acotada y comparar en el mismo gold, incluyendo negativos, contrastes, latencia y cobertura. No asumir superioridad. |
| 3 | Word2Vec opcional | Baseline estático solo si responde a una comparación útil; vectores con procedencia/licencia conocida, política de OOV y agregación fijada. No entrenar vocabulario sobre validation/test. Puede omitirse si diluye la pregunta o el presupuesto. |
| 4 | BM25 | Nueva línea de recuperación con documentos turísticos autorizados, consultas y juicios de relevancia; Recall@k, MRR, nDCG@k y latencia |
| 5 | Sentence-BERT e híbrido | Mismo corpus y juicios de recuperación; comparar contra BM25 antes de proponer fusión o re-ranking |
| 6 | RAG eventual | Solo con base autorizada y necesidad de respuestas sustentadas; medir relevancia de recuperación, apoyo de afirmaciones, citas y abstención antes de integrar generación |

Se conservan los cinco agentes. Los candidatos de sentimiento se evalúan fuera de la DEMO y solo se integrarían en `SemanticsAgent` tras una decisión experimental; recuperar documentos podría ser una capacidad interna posterior. No se cambian Midnight Indigo, recomendaciones actuales ni benchmarks históricos.

## Apéndice: reproducción privada de la inspección UNED

El siguiente script es el utilizado en esta sesión. Requiere una copia del ZIP oficial bajo la ruta privada indicada y la versión corregida de la herramienta. Crea y elimina un CSV temporal, conserva el archivo original y escribe solo el informe agregado en una carpeta ignorada por Git. No descarga archivos, no entrena y no crea particiones. Antes de repetir, comprobar el SHA-256 del ZIP de la tabla anterior. La metodología de palabras inglesas/españolas es diagnóstica y se conserva completa para no atribuirle validez de un identificador de idioma.

```python
from pathlib import Path
from zipfile import ZipFile
import csv, hashlib, json, re, tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from traveler_agent.corpus_metadata_probe import inspect_csv

archive = Path("data/private/source-audit-20261106/uned_zip.zip")
with ZipFile(archive) as package:
    xml_name = "HotelReview Corpus v0.1/HotelReview_Corpus_v0.0.xml"
    xml_bytes = package.read(xml_name)
    readme_bytes = package.read("HotelReview Corpus v0.1/README.txt")
xml = xml_bytes.decode("utf-8")
renames = Counter()
def fix_attribute_names(match):
    tag = match.group(0)
    for old, new in (("3_classes_intensity", "three_classes_intensity"),
                     ("5_classes_intensity", "five_classes_intensity")):
        tag, count = re.subn(r"(?<=\s)" + old + r"(?=\s*=)", new, tag)
        renames[old] += count
    return tag
# Only opening HotelReview tags are changed; the source archive stays intact.
fixed = re.sub(r"<HotelReview\b[^>]*>", fix_attribute_names, xml)
root = ET.fromstring(fixed)
assert root.tag == "HotelReviewCorpus"
assert all(row.tag == "HotelReview" for row in root)
rows = []
for row in root:
    rows.append({"text": "\n".join(x for x in (
        row.findtext("PositiveOpinion") or "", row.findtext("NegativeOpinion") or ""
    ) if x.strip()), "label": row.attrib["three_classes_intensity"],
        "id": row.attrib["idDoc"], "hotel": row.attrib["hotelID"]})
with tempfile.TemporaryDirectory(prefix="uned-probe-") as directory:
    csv_path = Path(directory) / "candidate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "label", "id", "hotel"),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata, _ = inspect_csv(csv_path, text_column="text", label_column="label",
        id_column="id", group_columns=("hotel",),
        allowed_label_values=("Good", "Fair", "Poor"))
# Diagnostic only, not a trained/validated language identifier.
en = set("the and was were very staff room breakfast hotel with clean good location not for to of".split())
es = set("el la los las muy fue eran personal habitación desayuno con limpio bueno ubicación para".split())
clues = Counter()
for row in rows:
    words = set(re.findall(r"\b\w+\b", row["text"].lower()))
    left, right = len(words & en), len(words & es)
    clues["english_dominant" if left > right else "spanish_dominant" if right > left else "tie"] += 1
result = {
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "xml_sha256": hashlib.sha256(xml_bytes).hexdigest(),
    "readme_sha256": hashlib.sha256(readme_bytes).hexdigest(),
    "archive_bytes": archive.stat().st_size,
    "xml_bytes": len(xml_bytes), "readme_bytes": len(readme_bytes),
    "research_notice_present": b"available for research purposes" in readme_bytes,
    "source_xml_strict_parse": "FAIL_NUMERIC_ATTRIBUTE_NAMES_LINE_3_COLUMN_176",
    "attribute_names_renamed_in_memory": dict(renames),
    "source_attribute_names": ["idDoc", "hotelID", "hotelLocation", "reviewerCategory", "reviewerNationality", "date", "score", "5_classes_intensity", "3_classes_intensity"],
    "text_elements": ["PositiveOpinion", "NegativeOpinion"],
    "five_class_counts_raw": dict(sorted(Counter(row.attrib["five_classes_intensity"] for row in root).items())),
    "three_class_csv_probe": metadata,
    "language_wordlist_diagnostic_NOT_CERTIFICATION": dict(clues),
    "language_distribution": "UNVERIFIED_NO_LANGUAGE_FIELD",
    "admission": "NO_CORPUS_ADMITTED",
    "rights": "RESEARCH_PURPOSE_NOTICE_PRESENT_ORIGINAL_REVIEW_RIGHTS_AND_REDISTRIBUTION_NOT_VERIFIED",
    "training_runs": 0, "test_split_created": False,
}
Path("data/private/source-audit-20261106/uned_audit.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
```
