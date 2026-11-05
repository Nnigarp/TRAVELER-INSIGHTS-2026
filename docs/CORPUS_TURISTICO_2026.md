# Traveler Insights Tourism Corpus 2026
## Fundación ejecutable de corpus académico, 5 de noviembre de 2026

Estado: infraestructura local implementada; SIN corpus turístico real publicado y SIN métricas científicas.

### Tablas privadas canónicas

Todas se crean por defecto en data/private/tourism-corpus-v1/, excluida por .gitignore.

| Tabla | Columnas y propósito |
| --- | --- |
| sources.csv | source_id, source_type, source_reference, reuse_basis, permission_reference, authorization_reviewed. Registro de origen, reutilización y evidencia de permiso. |
| reviews.csv | review_id, text, source_id, group_id, language. Texto original, identificador seudónimo y agrupación para evitar fugas. |
| annotations.csv | review_id, annotator_id, sentiment, aspects_json. Dictámenes individuales y no sobrescribibles. |
| adjudications.csv | review_id, sentiment, aspects_json, adjudicator_id, rationale. Etiqueta final revisada y justificada. |

Etiquetas globales: positivo, neutral, negativo, desconocido. Polaridades por aspecto iguales. Aspectos: servicio, higiene, espera, precio, seguridad, acceso, alimentacion, instalaciones.

Ejemplo ILUSTRATIVO, NO RESEÑA RECOLECTADA:
«El hotel estaba sucio, la atención excelente y esperamos demasiado»:
higiene negativo, servicio positivo, espera negativo. El sentimiento global debe anotarlo una persona, sin sumar mecánicamente aspectos.

### Reutilización, fuentes y privacidad

- Fuente voluntary con base consent: una opinión entregada voluntariamente tras informar propósito, usos, almacenamiento local, revisión y retirada; conservar el documento de consentimiento FUERA del dataset y del repositorio.
- Fuente open_corpus con base license: revisar la licencia concreta, atribución, redistribución y protección de datos ANTES de copiar.
- Fuente institutional con base written_permission: autorización escrita previa.
- authorization_reviewed=yes es declaración de revisión por el operador, NO comprobación jurídica del código.
- Evitar scraping no autorizado y no publicar textos o archivos de terceros. La licencia del modelo no implica licencia de las reseñas.
- Los IDs de personas/grupos deben ser seudónimos. El operador debe eliminar PII del texto antes de ingresarlo. NO existe anonimización automática ni validación legal automatizada.

### Guía de anotación

1. Leer el texto completo, interpretar negaciones y opinión mixta.
2. Asignar sentimiento global usando positivo, neutral, negativo o desconocido si no existe evidencia suficiente.
3. Anotar aspectos explícitos y sus polaridades: omitir aspectos no mencionados, usar desconocido ante ambigüedad y neutral únicamente si hay una mención sin valoración.
4. Registrar un dictamen por anotador y reseña, sin reemplazar el dictamen anterior. Recomendada doble anotación ciega en el conjunto crítico.
5. Adjudicar discrepancias con una justificación breve, conservando opiniones originales. El código exige al menos una anotación previa, NO garantiza aún doble revisión.
6. Spans y offsets de etiquetas humanas, Kappa entre anotadores y monitor de desacuerdos: próximos incrementos; aún no implementados en esta tabla.

### Instalación y comandos exactos (PowerShell)

Desde la raíz del proyecto y su entorno virtual:

~~~powershell
git fetch origin
git switch --track origin/feat/tourism-corpus-foundation
.\.venv\Scripts\python.exe -m pip install -e ".[ui,test]"
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli init
~~~

Editar a mano sources.csv bajo data/private/tourism-corpus-v1/ y completar source_id, source_type, source_reference, reuse_basis, permission_reference y authorization_reviewed, solamente después de revisar permiso.

Una contribución voluntaria autorizada se guarda primero en un archivo UTF-8 privado, revisado para evitar información personal, por ejemplo data/private/entrada.txt. Si la fuente voluntary2026 fue registrada:

~~~powershell
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli contribute --text-file data/private/entrada.txt --source-id voluntary2026 --group-id g1 --language es --consent-confirmed
~~~

La respuesta entrega un review_id; copiarlo. Nunca se imprimirá el texto original en la respuesta.

Ejemplo con ID ficticio r_ID:

~~~powershell
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli annotate --review-id r_ID --annotator a01 --sentiment neutral --aspects "{}"
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli adjudicate --review-id r_ID --adjudicator lead --sentiment neutral --aspects "{}" --reason "Revisado según guía de anotación"
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli validate
~~~

Para preparar particiones, tras reunir suficientes reseñas aprobadas y adjudicadas de las tres clases:

~~~powershell
.\.venv\Scripts\python.exe -m traveler_agent.corpus_cli split --seed 42
~~~

Genera train.csv, validation.csv, test.csv y split_manifest.json, todos dentro de data/processed/tourism-corpus-v1/ (ignorados por Git).

### Particiones y criterios metodológicos

- Se conectan filas por group_id compartido y por hash de texto idéntico tras normalización; un componente nunca se divide entre splits.
- División determinista con seed=42; proporción objetivo por componentes ~60/20/20, no porcentajes exactos de filas.
- Se requieren 9 componentes como mínimo y cada split debe contener las tres clases; si no se cumple se aborta.
- Solo se incluyen etiquetas finales adjudicadas, excluyendo desconocido.
- Manifiesto registra SHA-256 de entradas y salidas, semilla, clases, recuentos y limitaciones; no copia textos.
- Casi duplicados (paráfrasis o errores ortográficos), PII y solapamiento de autores no se detectan automáticamente: revisión futura necesaria.
- No utilizar el evaluador holdout antiguo como experimento definitivo del corpus agrupado. El próximo frente agregará Dummy/LR/SVM bajo el mismo split y ajustará modelos/calibración solo en train/validation; test debe quedar sellado.

### Audio y corpus

El PR #6 de audio sigue separado: ya se demostró una transcripción local real en Windows con edición y análisis LR/SVM. Falta documentar la prueba de subida de archivo y cerrar la aceptación antes de fusionarlo.

### Límites científicos

Las pruebas automatizadas usan reseñas sintéticas solamente para contratos de software. No se han recolectado ni evaluado reseñas auténticas en este incremento. No declarar ABSA entrenado, acuerdo entre anotadores, calidad de generalización, rendimiento turístico ni resultados del concurso.


## Interfaz local de anotación y controles de calidad

Se añadió corpus_annotation_app.py con Gradio local (127.0.0.1, share=False). No está integrada en la pestaña pública de análisis: se mantiene separada porque puede mostrar textos sensibles y requiere revisar fuentes antes de usarlos.

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui,test]"
.\.venv\Scripts\python.exe corpus_annotation_app.py
~~~

Se selecciona un review_id del corpus privado; se ve el texto original, se introduce ID seudónimo de anotador, sentimiento global, aspectos JSON y evidencias JSON con offsets Unicode [start,end) exactos. El botón «Verificar evidencias» muestra los fragmentos literales y rechaza posiciones incorrectas. Guardar no sustituye anotaciones anteriores. Los spans anotados se conservan en evidence_spans.csv dentro de data/private/tourism-corpus-v1/; esa tabla es adicional y no modifica los CSV del contrato v1.

Ejemplo ilustrativo para evidencias: un fragmento como «sucio» en una reseña real se selecciona por el valor exacto de start/end sobre el texto original; los ejemplos de documentación no son contribuciones reales.

El botón de control de calidad calcula Cohen kappa por pares de anotadores sobre reseñas compartidas, incluyendo aspectos por presencia/ausencia y polaridad cuando fueron anotados. Si el acuerdo es indefinido, devuelve null. Los posibles casi duplicados se sugieren mediante similitud literal difflib (umbral por defecto 0.88, máximo 2000 reseñas por ejecución); se muestran solamente IDs y una puntuación orientativa, sin publicar texto ni eliminar registros. Estas herramientas no garantizan calidad ni consentimiento y no sustituyen adjudicación humana. El cálculo O(n²) es exploratorio; se necesitarán filtros por bloques y revisión antes de escalar a corpus grandes.

No se ha implementado aún un editor de adjudicaciones dentro de Gradio; sigue disponible por CLI. No conectar esta UI privada a un dominio público ni deshabilitar share=False sin autenticación, consentimiento y revisión de seguridad.
