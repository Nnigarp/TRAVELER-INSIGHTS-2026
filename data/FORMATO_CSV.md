# Formato del dataset verificado

Se espera un CSV UTF-8, de procedencia autorizada, con columnas `reseña` y `sentimiento`.

| reseña | sentimiento |
|---|---|
| (reseña original sin datos personales) | positivo / neutral / negativo |

**No crear ni publicar CSV sintéticos como dataset real.** Si el archivo de Kaggle trae etiquetas 0, 1, 2, convertir solo después de auditar el mapeo. El proceso descarta reseñas nulas, cortas y duplicadas idénticas, y rechaza contradicciones explícitas de etiquetas.
