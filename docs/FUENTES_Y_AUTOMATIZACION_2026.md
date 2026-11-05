# Fuentes reales candidatas y automatización — 5 noviembre 2026

## Evaluación documental (no es autorización jurídica definitiva)

| Fuente | Datos reportados | Idiomas | Licencia declarada | Observaciones |
| --- | --- | --- | --- | --- |
| HAMOS v4 — https://data.mendeley.com/datasets/x46f4y4fpr/4 | 8.796 reseñas, 23.995 cuádruplas de opinión, 3.399 hoteles, splits por hotel | Inglés, vietnamita y otros | CC BY 4.0 en Mendeley Data | Mejor candidato para ABSA; revisar condiciones del texto Booking.com de origen y taxonomía; no asumir etiquetas globales |
| Argilla TripAdvisor — https://huggingface.co/datasets/argilla/tripadvisor-hotel-reviews | 20.491 reseñas; puntuaciones 1–5 de origen | Inglés | CC BY-NC 4.0 declarada | Candidato para línea base global con etiquetas derivadas de valoraciones (etiquetas débiles, NO anotación humana independiente de 3 clases); verificar derechos sobre reseñas TripAdvisor |
| CrawlFeeds Booking — https://huggingface.co/datasets/crawlfeeds/Booking-Hotel-Reviews-Dataset | ≈4.370 filas; positivo/negativo separados; datos de hotel y revisor | Mayormente inglés según archivo publicado | CC BY-NC 4.0 declarada | Riesgo de privacidad de nombres/reviewer; comprobar autorización de extracción desde Booking y no difundir archivos sin revisión |
| Yelp Open Dataset — https://business.yelp.com/data/resources/open-dataset/ | A confirmar por versión descargada | Predominantemente inglés | Acuerdo específico, NO licencia abierta irrestricta | Uso académico condicionado; restricciones adicionales de publicación y reutilización. No importar ni publicar sin validar el acuerdo específico. |

### Selección

**Primero examinar HAMOS v4 como candidato ABSA**, porque proporciona aspectos, polaridades y splits por hotel; usar únicamente después de revisar los archivos reales, atribución y términos relativos al contenido original. **Argilla TripAdvisor** es opción separada para entrenamiento global exploratorio con etiquetas débiles derivadas de estrellas; no equiparar ratings con anotación humana gold. Ninguna de estas fuentes debe presentarse como ya descargada o aceptada legalmente en este repositorio.

No declarar trabajo original de recolección cuando se reutilice un corpus de terceros. El corpus propio puede ser una combinación de material legítimamente reutilizado más contribuciones voluntarias; documentar origen y restricciones por separado y reservar revisión humana independiente para el test de referencia.

## Importación automatizada conservadora

El módulo `traveler_agent.corpus_automation` funciona sobre un CSV **ya obtenido legalmente**, nunca navega sitios web ni hace scraping, y requiere que `sources.csv` haya registrado y revisado la fuente.

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli init
.\.venv\Scripts\python.exe -m traveler_agent.corpus_automation import-csv --file data/private/archivo-autorizado.csv --source-id fuente_aprobada --text-column texto --group-column hotel_id --id-column review_id --language en
.\.venv\Scripts\python.exe -m traveler_agent.corpus_automation near-duplicates --threshold 0.88
.\.venv\Scripts\python.exe -m traveler_agent.corpus_automation preannotate-demo --model-kind lr
```

Los nombres `texto`, `hotel_id`, `review_id` son **marcadores de ejemplo**: cambiarlos por nombres de columnas observados en el CSV original. No generar grupos ficticios para datos reales; usar hoteles o autores seudónimos si la procedencia los proporciona, y documentar posibles relaciones no observables.

Importación: valida la autorización declarada, conserva la reseña original localmente, usa ID seudónimo derivado de fuente e ID original, omite duplicados exactos y **excluye filas sospechosas** con patrón de email, teléfono o URL. El informe solo muestra conteos, no textos. Los filtros de PII son incompletos (pueden pasar nombres, ubicaciones personales, etc.), requieren revisión adicional, y también pueden producir falsos positivos. Las reseñas descartadas deben revisarse fuera del sistema si se desea rescatarlas; nunca reinsertarlas sin control.

`near-duplicates`: genera un archivo local de pares de IDs y similitud indicativa; no elimina registros. El algoritmo actual es O(n²) y acepta como máximo 2000 reseñas por ejecución, por lo que un corpus mayor requiere bloques y revisión.

`preannotate-demo`: ejecuta el clasificador entrenado con ejemplos ficticios, produce un CSV separado con sugerencias y `needs_human_review=yes` para **todos**. **No modifica** `annotations.csv`, `adjudications.csv`, `train.csv` ni el test. No constituye anotación humana, validación científica ni modelo turístico entrenado.

### Qué falta antes de un estudio defendible

1. Confirmación documental de permiso y compatibilidad con reutilización del contenido original.
2. Perfil real de esquemas y cobertura de idioma/ratings, privacidad, casi duplicados y correlación por hotel.
3. Definir política de etiquetas derivadas de estrellas y mantenerlas separadas de etiquetas gold.
4. Etiquetar independientemente una muestra de referencia (test) y resolver desacuerdos, registrando kappa.
5. Evaluar sobre test reservado y una fuente externa independiente, sin modificar hiperparámetros según test.
6. Se recomienda asignar revisiones humanas por incertidumbre y diversidad de fuentes, no por cada registro.
