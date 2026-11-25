# Aceptación de la entrada unificada en Windows

## Arranque desde el clon validado

En PowerShell, con el entorno y los modelos ya instalados:

```powershell
Set-Location 'C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github'
git switch feat/unified-multilingual-input-20261125
$env:TRAVELER_BETO_ARTIFACT = (Resolve-Path 'artifacts/beto-v3-pr28-local').Path
$env:HF_HUB_OFFLINE = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
ollama list
.\.venv-research\Scripts\python.exe app.py
```

Abre la dirección local que imprima Gradio. Si Ollama está detenido, inicia la aplicación Ollama de Windows. El modelo esperado es `qwen3:1.7b`.

## Aceptación personal

1. Carga el ejemplo mixto. Comprueba el contador y analiza con LR, luego con SVM y BETO V3. En el JSON comprueba el clasificador efectivo y, para BETO, `seleccion_modelo.fallback=false`.
2. Abre **+ Audio · micrófono o archivo**, graba una reseña corta, detén la grabación y revisa el texto. Comprueba «excursión», nombres y negaciones. Envía la corrección y consulta ambos textos y los hashes en el JSON. Repite con un archivo. Pulsa **Limpiar** para comenzar una reseña independiente. **Descartar audio** permite retirar la grabación.
3. Con el informe apagado prueba «The bus never arrived after 2 hours». Comprueba origen inglés, destino español, negación y cifra. Corrige el español en **Idioma y traducción experimental** y vuelve a analizar.
4. Prueba «La camera era pulita ma il servizio era lento», «Le personnel était très aimable» y «酒店有2个房间。». Revisa que se refieran a una habitación, al trato del personal y a dos habitaciones, respectivamente. Si aparecen advertencias, corrige la propuesta y confirma su revisión.
5. Usa una reseña ambigua, por ejemplo «Regular». Ante un error puedes indicar el código de idioma, introducir una traducción propia, reintentar o limpiar. Activa el informe para verificar por separado la validación de ActionAgent.
6. Alterna ambos temas, usa Tab para recorrer los controles y abre Agentes. Lotes comprueba todas las filas antes de inferir y señala la primera cuyo español resulta incierto. En ese caso utiliza Análisis individual. Las traducciones de lote tienen presupuesto **cero**. Los informes de lote conservan el máximo de cinco solicitudes con el presupuesto temporal compartido existente.

## Contratos y límites

La aplicación conserva las palabras de entrada en `procesamiento_linguistico.texto_original`. Cuando hay ASR, `procedencia_entrada` conserva el hash del archivo recibido y el del WAV procesado, duración, motor, idioma y edición. El texto en español efectivamente entregado a los agentes está en `procesamiento_linguistico.texto_analizado`. Los fragmentos de evidencia del clasificador se extraen de ese texto. Las ediciones y trazas son datos de sesión.

El vocabulario del detector histórico cubre español, inglés e italiano con palabras compartidas como «un». Sigue disponible en el perfil descriptivo del clasificador. El nuevo filtro considera señales léxicas diferenciadas, mezcla y escritura. Consulta a Qwen ante incertidumbre o discrepancia con Whisper. La identificación y la traducción usan mensajes independientes, esquema cerrado, respuesta local acotada, hasta 5.000 caracteres y máximo de 60 segundos por solicitud. Un resultado dudoso bloquea la inferencia hasta que el usuario aporte una decisión. Los indicios de cifras, negación, contraste y falsos amigos cubren errores concretos. Un JSON válido puede contener una traducción equivocada.

La generación opcional posterior conserva `ollama-tourism-report-v2`, diez eventos del orquestador y tres eventos generativos cuando se solicita el informe. Los resultados lingüísticos se almacenan fuera de esas trazas. Las pruebas con frases sintéticas verifican el flujo y sus límites. La calidad de traducción, el reconocimiento de voz y la predicción con turistas reales requieren evaluaciones independientes.

## Sustentación y evidencia histórica

En la figura de arquitectura, añade el preprocesamiento perceptivo entre entrada editable y TravelerOrchestrator. En PEAS, el teclado y micrófono capturan observaciones del entorno turístico. La fidelidad del texto y la derivación a revisión forman parte del desempeño. Actualiza el diagrama de secuencia con Whisper, identificación y traducción, el plan de cuatro especialistas, el informe opcional y la memoria episódica por reseña. En PLN y aprendizaje supervisado, muestra sobre qué cadena operan TF-IDF y el Transformer BETO. Distingue la traducción generativa del control de revisión. Sustituye la figura de cuatro pestañas por una captura aceptada y añade un caso con corrección humana. Conserva tablas y métricas de experimentos históricos.

El 25 de noviembre, en **main c08f433**, el estudiante registró para la reseña mixta un score de 0,4592 con LR, 0,4159 con SVM y 0,9966 con BETO. Las latencias históricas del informe fueron 5877,68, 5837,08 y 6108,13 ms respectivamente. La aceptación por consola de BETO + Qwen registró 14921,71 ms. También se probó Whisper Tiny en español e italiano. Estas cifras corresponden a la versión estable.

## Comprobaciones del incremento en Lenovo

| Recorrido | Resultado observado |
|---|---|
| Español, reseña mixta | LR, SVM y BETO V3 efectivos. En los tres casos el informe Qwen real terminó en GENERADO y hubo diez eventos principales. |
| Inglés, francés y chino escrito | Con informe apagado, la interfaz recibió traducciones de Qwen local y clasificó el español con LR. El caso chino «酒店有2个房间。» produjo «El hotel tiene 2 habitaciones». |
| Italiano | Qwen confundió «camera» con «cámara». El control específico detuvo el análisis. La corrección a «habitación» y su confirmación permitieron clasificar. |
| Archivo y dictado | Chrome cargó un WAV con voz sintética. En una segunda ejecución, Chrome reprodujo el WAV mediante un micrófono virtual de prueba. En ambos recorridos Whisper Tiny real devolvió texto editable, se registró la corrección y se ejecutó LR. La identificación de idioma y las palabras de la voz sintética tuvieron errores. Queda pendiente el micrófono físico. |
| Navegador | 68 comprobaciones, ambos temas, escritorio y móvil. Se verificaron un solo envío ante doble clic, invalidación del informe, contador, carga de CSV y descargas. [Registro completo](acceptance/2026-11-25-unificada/browser-check.json). |
| Aceptación académica | Ejecuciones funcionales y HTTP con LR y SVM en Windows completadas sin violaciones. La batería académica identifica de forma explícita el idioma de sus fixtures españoles preinscritos. |

La ventana de revisión permite corregir también traducciones que pasan los controles. La selección manual de idioma ayuda cuando la voz, la ortografía o la mezcla de idiomas dificultan la identificación. Las pruebas simuladas en pytest cubren respuestas malformadas, servicio desconectado y contenido adversarial.

## Repetición y rollback

```powershell
.\.venv-research\Scripts\python.exe -m pytest -q
```

Para desactivar la asistencia lingüística de Qwen, establece `$env:TRAVELER_LANGUAGE_QWEN='0'` antes del arranque. El español identificado localmente y la traducción manual siguen disponibles. Para volver al incremento estable, detén Gradio con Ctrl+C, ejecuta `git switch main`, verifica `git rev-parse --short HEAD` y vuelve a ejecutar el comando de arranque. La base verificada para esta entrega es `c08f433`.

