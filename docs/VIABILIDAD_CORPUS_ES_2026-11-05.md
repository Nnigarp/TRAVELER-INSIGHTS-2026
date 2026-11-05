# Viabilidad de corpus turísticos auténticos en español — 5 de noviembre de 2026

**Estado:** evaluación documental de candidatas; ninguna importación real ni experimento científico se ha ejecutado. Esta es una auditoría de fuentes disponibles, no una autorización legal automática.

## Comparación

| Fuente | Evidencia documental | Clase/idioma | Derechos y acceso | Uso condicionado |
| --- | --- | --- | --- | --- |
| COAR — SINAI/UJA | 2.202 opiniones de restaurantes tomadas de TripAdvisor; cinco niveles (1=565, 2=246, 3=188, 4=333, 5=870) | Español; etiquetas de intensidad 1–5; NO consta etiquetado humano independiente de tres clases | Hugging Face declara CC BY-NC-SA 4.0 y acceso condicionado a aceptar términos; BOE-A-2025-12053 incluye COAR de SINAI-UJA bajo **CC BY 4.0**. La discrepancia y los derechos originales de textos TripAdvisor requieren confirmación escrita del titular | Primer candidato global **una vez confirmados** permiso, versión y esquema. Conversión 1/2 negativo, 3 neutral, 4/5 positivo = supervisión derivada de ratings/intensidad, NO gold humano |
| Trip-MAML | 500 reseñas de hoteles en español anotadas manualmente por aspecto a nivel de oración, además de italiano e inglés | Español; ABSA, no etiqueta global universal | Repositorio GitHub público, sin licencia de reutilización explícita verificada en los archivos vistos; origen TripAdvisor | Candidato a evaluación ABSA con permiso explícito |
| HOpinion — UB | 17.934 opiniones españolas principalmente hoteleras, 4.740 con lemas/POS | Español; rating y metadatos de usuarios | Publicación académica y descarga mencionadas; licencia y reuso no especificados; alias, edad, género, origen y otros datos personales presentes | Solo si los autores autorizan usos/retención y se eliminan metadatos sensibles |
| HotelReview Corpus — UNED | 1.000 reseñas de Booking.com con anotaciones MANUALES globales de 3 y 5 clases, además de rating automático original | Idioma de textos no confirmado en ficha; no asumir español | Disponible para investigación en XML, sin licencia de redistribución precisa verificada; posible PII | Interesante por gold global, sujeto a autorización y auditoría de idioma |
| Hotel Reviews ES — Karpacious | Visor muestra 787 filas de hotel y etiquetas positivas/negativas automáticas; tarjeta también dice 1.500 y muestra contradicción sobre porcentaje de muestra | Español; **automática binaria**, sin neutral | Declara CC BY-NC 4.0; derechos originales no establecidos, procedencia escasa | No sirve como gold triclase; baja prioridad |
| TSD / Granada | Artículo informa 38.921 textos etiquetados combinando TripAdvisor, tweets de Granada y TASS; clases negativas/neutrales/positivas | Español; mezcla de ratings derivados y distintas anotaciones | Publicación para investigación; licencia/reuso/partición por fuente no comprobados | No mezclar las fuentes sin volver a auditar textos y etiquetas |
| SFU Spanish Review Corpus | 400 reseñas de ocho dominios, de ellas 50 de hoteles; etiquetas positivas/negativas a partir de estrellas | Español; binario débil, pequeño | Recurso académico público; términos originales de Ciao.es por aclarar | Baseline de negaciones/transferencia, no principal turismo triclase |

## Evidencias trazables

- COAR Hugging Face: https://huggingface.co/datasets/SINAI/COAR
- BOE-A-2025-12053 (HTML, tabla de SINAI-UJA bajo titularidad/licencias): https://www.boe.es/diario_boe/txt.php?id=BOE-A-2025-12053
- Trip-MAML: https://github.com/diegma/trip-maml
- HOpinion UB: https://clic.ub.edu/corpus/es/hopinion
- HotelReview UNED: https://nlp.uned.es/~lplaza/resources/hotel-review-corpus/
- Hotel Reviews ES: https://huggingface.co/datasets/Karpacious/hotel-reviews-es
- TSD Granada (publicación): https://www.sciencedirect.com/org/science/article/pii/S1757988022000265
- SFU Spanish Review Corpus: https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html

## Decisión metodológica (sin forzar aprobación)

**Prioridad documental: COAR de SINAI/UJA**, por ser español, turismo/restauración, tamaño manejable y tres clases derivables. **Bloqueo:** obtener confirmación escrita del grupo titular sobre licencia correcta (CC BY 4.0 en BOE frente a CC BY-NC-SA 4.0 en HF), facultad de usar el texto original de TripAdvisor para investigación, acceso a la versión adecuada, atribución, redistribución permitida o solo procesamiento local y medidas de privacidad. No marcar approval.json como autorizado hasta tener evidencia real.

Segundo candidato para etiquetas humanas: **HotelReview UNED** por sus tres clases manuales; verificar idioma (no garantizado español), condiciones de uso del texto Booking.com, derechos y anonimización. Tercero: Trip-MAML para ABSA cuando se aclare licencia.

## Protocolo experimental mínimo después de autorización

1. Documentar carta/permisos, condiciones, versión/revisión, SHA-256, cita y política de retención; guardar constancias privadas sin publicar reseñas ni datos personales.
2. Inspeccionar archivo original: columnas, número real, idiomas, escala de etiquetas y unidades de grupo (autor seudónimo/hotel/fuente); descartar PII y auditar duplicados exactos/casi idénticos antes de dividir.
3. Declarar si las clases son **ratings derivados débiles**, etiquetas humanas originales o predicciones automáticas. No usar preanotaciones como gold.
4. Reservar test por grupos o fuente sin volver a consultarlo para seleccionar modelos; train/validation sobre grupos sin fugas. Verificar balance y clases, congelar semilla y hashes.
5. Comparar DummyClassifier, TF-IDF+LR y TF-IDF+LinearSVC bajo protocolo igual; seleccionar hiperparámetros por Macro-F1 de validación. Calibración SVM solo dentro de train con folds por grupos.
6. Evaluación una sola vez en test: Macro-F1, accuracy, precision, recall, F1 por clase, matrices de confusión, soporte por clase/idioma, auditoría de errores por negación/contraste y casos de opinión mixta; no interpretar probabilidades como calibradas externamente sin validación.
7. Obtener test independiente anotado por humanos en español para conclusiones sobre sentimiento de Cartagena; no extrapolar de ratings derivados o España.
8. Reportar datos reales SOLO tras cumplir derechos, privacidad, anotación y evaluación independiente.

**Estado final:** ningún corpus de reseñas se ha declarado aprobado; no se descargaron textos, no se generaron etiquetas gold ni métricas empíricas.
