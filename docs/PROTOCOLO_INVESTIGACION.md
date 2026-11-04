# Protocolo experimental — borrador registrable v0.1

**Pregunta principal:** ¿en qué medida un clasificador textual reproducible permite distinguir sentimiento positivo, neutral y negativo en el reto Traveler Insights 2026 y cómo cambia su rendimiento al transferirse a reseñas turísticas auténticas de Cartagena?

**Pregunta secundaria (investigación futura):** ¿añadir análisis por aspectos y una política de abstención aporta recomendaciones más útiles sin degradar la clasificación global?

## Hipótesis que aún no se han contrastado

- H1: un modelo contextual multilingüe puede superar una línea base TF-IDF + regresión logística/SVM en Macro-F1 bajo idéntica evaluación.
- H2: el desplazamiento de dominio a Cartagena puede reducir el desempeño; la magnitud debe medirse, no suponerse.
- H3: trazabilidad, evidencia y revisión humana pueden aumentar utilidad/seguridad; requieren anotación y estudio con evaluadores.

## Diseño experimental mínimo

1. **Proveniencia:** registrar fecha, URL, licencia/permiso, SHA-256, tamaño, procedencia, idioma, columnas, distribución, duplicados y anotadores. No redistribuir reseñas sin verificar los derechos.
2. **Etiquetas:** conservar clasificación de tres clases y verificar el significado de 0/1/2 en los CSV oficiales. No fabricar una clase neutral si no existe.
3. **Particiones:** separar por reseña sin duplicados (y por ubicación o fecha cuando existan esos metadatos), fijar semillas. Reservar test sin tocar hasta el final; selección de hiperparámetros solo en desarrollo/validación.
4. **Línea base:** TF-IDF palabras/caracteres + regresión logística; SVM lineal como segunda línea base. Evitar escalado o vectorización fuera del pipeline.
5. **Alternativas 2026:** encoder multilingüe con ajuste fino controlado y metodología ABSA, condicionados a muestras anotadas, recursos y licencias.
6. **Métricas:** accuracy compatible con reto, Macro-F1 principal de investigación, Precision/Recall/F1 por clase, matriz de confusión, intervalos de confianza y análisis de error; no escoger por AUC solo.
7. **Calibración/abstención:** evaluar diagramas de fiabilidad y desempeño selectivo antes de usar un umbral probabilístico. El umbral actual es demostrativo.
8. **Evaluación fuera de dominio:** nuevo lote de reseñas legalmente obtenido en Cartagena; doble anotación y acuerdo interevaluador. Documentar sesgos de plataforma, idioma, turismo y cambios temporales.
9. **Ablación:** comparar (a) solo etiqueta global, (b) +aspectos, (c) +evidencia, (d) +revisión humana; medir utilidad con una rúbrica y evaluadores.
10. **Reproducibilidad:** registrar dependencias, hardware, duración, parámetros, semillas, hashes, scripts y resultados completos; publicar exclusivamente métricas ejecutadas.

## Advertencia de alcance

El código actual permite una prueba de integración y un holdout básico para CSV reales, pero **no** demuestra ninguna hipótesis. Para un artículo es necesario ampliar la implementación, recolección y protocolo antes de comunicar conclusiones.

## Esquema recomendado para una futura publicación

Introducción; pregunta y contribuciones; trabajos relacionados; datos y ética; arquitectura y métodos; protocolo de evaluación; resultados; ablaciones y errores; amenazas a la validez; reproducibilidad; conclusiones. Verificar formato y convocatoria concretos antes de redactar CEUR-WS u otro formato.
