# Auditoría comparativa y viabilidad — 2026-11-05

Estado: auditoría de fichas y campos visibles, código adaptador ejecutable. NO se ha importado corpus auténtico ni hay benchmark real publicado. Condiciones sobre texto original todavía pendientes de confirmación.

## Fuentes

| Dataset | Cantidad | Estructura efectivamente visible | Tarea | Licencia declarada | Riesgo pendiente |
| --- | ---: | --- | --- | --- | --- |
| Argilla/TripAdvisor | 20.491 | Parquet: text, id, prediction con lista {label,score}, annotation null, inputs, metadata | Rating 1–5, mapeable a sentimiento WEAK (1–2 negativo, 3 neutral, 4–5 positivo), NO etiqueta humana gold | CC BY-NC 4.0 | Texto obtenido originalmente de TripAdvisor; autorización sobre textos originales no verificada; hotel_id ausente; inglés y texto aparentemente preprocesado |
| HAMOS v4 | 8.796 reseñas, 23.995 cuádruplas, 3.399 hoteles | Organización declarada annotations/ metadata/ splits/ taxonomy/; JSON JSONL CSV. Nombres de columnas de cada archivo SIN verificar | ABSA, 6 categorías y 31 subaspectos, particiones oficiales por hotel | CC BY 4.0 | Texto originado de Booking.com y permisos sobre contenido original requieren revisión |
| CrawlFeeds/Booking | ~4.370 | CSV con uniq_id, hotel_name, reviewer_name, hotel_address, rating, positive_review_text, negative_review_text, language y otros | Segmentos positivo/negativo, no etiqueta global automática | CC BY-NC 4.0 | Revisor y datos privados, origen Booking.com, derechos de reutilización original pendientes |

Fuentes:
- https://huggingface.co/datasets/argilla/tripadvisor-hotel-reviews
- https://huggingface.co/datasets/argilla/tripadvisor-hotel-reviews/tree/main/data
- https://data.mendeley.com/datasets/x46f4y4fpr/4
- https://huggingface.co/datasets/crawlfeeds/Booking-Hotel-Reviews-Dataset
- https://tripadvisor.mediaroom.com/CO-terms-of-use
- https://www.booking.com/content/terms.html

El permiso declarado en la ficha derivada NO certifica por sí solo derechos sobre el contenido previo. Las condiciones públicas de Booking.com y Tripadvisor contienen restricciones de copia, extracción y redistribución. No se realizó descarga de las reseñas originales. La comprobación de derechos exige documentar base concreta de reutilización; no es un asesoramiento jurídico.

## Implementación funcional de la rama

- source_adapters.py: Argilla usa el rating numérico presente en prediction y conserva la etiqueta WEAK. HAMOS conserva cuádruplas y splits oficiales, requiere nombres de campos confirmados sobre archivo auténtico. CrawlFeeds conserva por separado campos positivos/negativos y NO asigna sentimiento global. Todos anotan task, source, label_origin, language, group_quality y texto SHA.
- source_import.py: inspección de nombres de columnas, filas y hash local sin mostrar reseñas; importación posterior exige archivo JSON privado de aprobación de derechos, fuente coincidente y privacidad revisada. Soporta Parquet opcional con pyarrow, CSV y JSONL local.
- weak_rating_benchmark.py: comparativa Dummy/LR/SVM solo para Argilla autorizado, sobre rating-proxy y test reservado con hashes de texto sin compartir. Resultado NO es Macro-F1 de sentimiento humano; tampoco prueba generalización por hotel porque el esquema no tiene hotel_id.
- Tests usan exclusivamente fixtures ficticios; no descargan muestras reales.

## Comandos Windows PowerShell

Desde la rama feat/multidataset-adapters, con Python 3.12, instalar pyarrow solamente si se inspecciona Parquet:
~~~powershell
.\.venv\Scripts\python.exe -m pip install pyarrow
.\.venv\Scripts\python.exe -m traveler_agent.source_import inspect --source argilla_tripadvisor --file data/private/originales/argilla.parquet
~~~

Primero obtener archivo LEGÍTIMAMENTE y verificar los derechos sobre el texto original. El archivo privado approval.json requiere source, reuse_rights_reviewed true, privacy_reviewed true, license_reference, original_content_rights_basis y reviewer_id. Nunca activar esos valores sin evidencia. Ejemplo de JSON de plantilla que permanece BLOQUEADO:
~~~json
{"source":"argilla_tripadvisor","reuse_rights_reviewed":false,"privacy_reviewed":false,"license_reference":"PENDIENTE","original_content_rights_basis":"PENDIENTE","reviewer_id":"seudonimo"}
~~~

Después de verificación auténtica:
~~~powershell
.\.venv\Scripts\python.exe -m traveler_agent.source_import import --source argilla_tripadvisor --file data/private/originales/argilla.parquet --approval data/private/approvals/argilla.json --output data/private/adaptados/argilla.jsonl
~~~

Para HAMOS los nombres de columnas requieren mapping JSON verificado antes de importar; no se adivinan. Para CrawlFeeds, el adaptador descarta campos como reviewer_name, hotel_address y URL en la salida, pero texto libre podría contener datos personales.

## Protocolo

Argilla: evaluación del rating-proxy únicamente; idioma inglés, particiones sin duplicados exactos, imposibilidad de asegurar independencia por hotel. HAMOS: respetar splits oficiales, conservar taxonomía y evaluar ABSA. CrawlFeeds: estudiar opinión mixta con revisión, no convertir automáticamente campos por segmento en etiqueta global. No combinar datasets hasta verificar tarea, supervisión, idioma, derechos, duplicados cruzados y test independiente.

Objetivo español Cartagena: conjunto test separado y anotado por humanos; todavía no existe. No declarar generalización en español usando únicamente resultados en inglés.

**Bloqueo de primera importación real:** faltan verificaciones independientes sobre los derechos de reutilización de textos de Tripadvisor/Booking.com. Hasta resolverlas, no ejecutar descargas ni publicar resultados de modelos reales sobre estos corpus.
