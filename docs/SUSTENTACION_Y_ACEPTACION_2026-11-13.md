# Traveler Insights: sustentación y aceptación

Versión académica sobre main `4df0ceb`. Registro de criterios y ejemplos anterior a ejecución: `e20326484aa33d335e85bf0f8551c24adbbd13e1`. No se amplía a5000, no se seleccionan hiperparámetros ni se cambian umbrales. LR es el valor inicial; BETO V3 es opcional y experimental.

## Preparar Windows

Desde la raíz del clon, usar Python3.12 de64bits y una terminal PowerShell. Hasta la revisión manual, utilizar la rama `feat/academic-acceptance-20261113`.

```powershell
git fetch origin
git switch feat/academic-acceptance-20261113
git pull --ff-only
py -3.12 -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-research\Scripts\python.exe -m pip install -r research/requirements-beto-v3.txt -e '.[ui,test,asr]'
$env:PYTHONUTF8 = '1'
.\.venv-research\Scripts\python.exe -m traveler_agent.audio_asr --check
.\.venv-research\Scripts\python.exe scripts/check_asr_readiness.py --require-packages --output artifacts/asr-readiness-local.json
```

Reutilizar un artefacto propio **ajustado V3, semilla42**, si ya existe. La carpeta base `beto-v3-checkpoint` sola no es un clasificador. Si faltan los pesos ajustados, esta preparación reconstruye una sola receta fija con los 300 TRAIN originales; no hace búsqueda ni lee los casos nuevos para ajustar parámetros. El destino debe ser nuevo; la descarga solo se permite con `--download`.

```powershell
.\.venv-research\Scripts\python.exe scripts/prepare_acceptance_beto.py --registration-commit e20326484aa33d335e85bf0f8551c24adbbd13e1 --download --output artifacts/academic-local-beto
$env:TRAVELER_BETO_ARTIFACT = 'artifacts/academic-local-beto/model'
.\.venv-research\Scripts\python.exe app.py
```

Para usar pesos propios existentes, asignar su carpeta a la variable en lugar de reconstruir. Iniciar Gradio después de asignarla. Abrir localhost mostrado en terminal; `share=False`. Si falta BETO no aparecerá como opción. Si falla durante la operación, el resumen, JSON y exportación deben identificar LR de respaldo. `modelo_utilizado=lr` nunca acredita una prueba satisfactoria de BETO. Los scripts de aceptación lo rechazan.

## Demostración breve

| Paso | Ejemplo funcional registrado | Qué debe demostrarse |
|---|---|---|
| Texto | F01: «Nos recibieron con el equipaje protegido de la lluvia y pudimos descansar enseguida.» | Etiqueta, puntuaciones no calibradas, modelo efectivo y diez eventos. Una etiqueta distinta de la intención no se oculta. |
| Contraste | F06: «La explicación se entendía muy bien; al final, sin embargo, nos dejaron sin transporte de regreso.» | Separar clasificación aprendida de la regla de revisión y mostrar sus motivos. |
| Entrada adversarial | F11: petición de ignorar instrucciones, seguida de cobro duplicado | El texto no cambia el plan, ejecuta órdenes ni provoca acciones externas. La etiqueta puede ser errónea. |
| Error | Entrada vacía y texto de5001 caracteres | Mensaje de validación, sin predicción ni ejecución completada inventada. |
| Lote | F01 y F13 (`=1+1; la excursión comenzó después del desayuno.`) | Dos filas, IDs únicos, corrección humana separada de predicción, descarga JSON/CSV. La celda de fórmula se neutraliza al exportar. |
| Modelo | Repetir F06 con LR, SVM y BETO | Mostrar `modelo_utilizado` y, en BETO, `seleccion_modelo.fallback=false` y hash del manifiesto. |

La aceptación funcional verifica contratos, no exige acertar una clase. Evitar presentar un ejemplo elegido de la tabla como estimación de rendimiento. Una reseña válida de5000 caracteres puede exceder la ventana de BETO: V3 usa256 tokens incluyendo especiales y recorta el resto. La aceptación registra el recorte; el resumen lo advierte. No atribuir comprensión del final a un modelo que no lo recibió.

## Defender la arquitectura

- **TravelerOrchestrator:** planifica cuatro pasos, delega mensajes, valida contratos y cierra la ejecución. Es coordinación jerárquica local, no cinco chats LLM independientes.
- **PerceptionAgent:** valida texto, longitud y modalidad; conserva el vínculo con una transcripción editada.
- **SemanticsAgent:** ejecuta el clasificador elegido, extrae aspectos y fenómenos exploratorios por reglas. Los fragmentos literales no explican causalmente la predicción.
- **ControlAgent:** aplica la regla histórica `.56`, o `.80` cuando hay un contraste reconocido. Sus razones se trazan; no certifica que una decisión automática sea segura.
- **ActionAgent:** produce una recomendación exploratoria y respeta revisión humana. No ejecuta acciones externas.

LR aprende pesos sobre TF-IDF de palabras/caracteres; SVM aprende una separación lineal y usa calibración interna. BETO aporta representaciones contextuales con WordPiece y una cabeza supervisada; V3 ajusta el encoder y la cabeza. Su entrenamiento sigue siendo sintético. Los clásicos de la app usan DEMO y BETO usa TRAIN experimental V3: la comparación de opciones no aísla el efecto de la arquitectura con datos idénticos.

Si la docente muestra una predicción errónea, conservar el JSON, explicar la limitación y registrar una corrección humana en el lote. La corrección no borra la predicción ni reentrena. Una puntuación alta no demuestra exactitud, validez humana o seguridad de automatización.

## Micrófono: ensayo personal pendiente

Los tests automáticos generan PCM sin habla y usan un ASR simulado, expresamente identificado. HTTP no acredita permisos de navegador, captura del micrófono ni reconocimiento real de voz. La aceptación antigua reportada para LR/SVM tampoco acredita esta versión con BETO.

1. Ejecutar `audio_asr --check`. Deben importar PyAV y CTranslate2; revisar versiones. El proyecto exige `av>=11,<19` y faster-whisper opcional. Resolver cualquier `NO_INSTALADO`, `ERROR` o `NO_DISPONIBLE` antes de la demostración.
2. Usar un modelo Whisper local con `TRAVELER_ASR_MODEL_PATH`. Para autorizar su primera descarga, antes de iniciar Gradio: `$env:TRAVELER_ASR_MODEL='tiny'` y `$env:TRAVELER_ASR_ALLOW_DOWNLOAD='1'`. Tras prepararlo, quitar la autorización y reiniciar. Inferencia CPU/int8; no detección de emociones en la voz.
3. En Audio, elegir español, conceder permiso al micrófono de localhost y grabar voz propia de5–15s: «No fue una mala visita, pero el regreso quedó sin organizar». Límite después de conversión: WAV PCM16,12MiB,120s.
4. Transcribir. Revisar especialmente «no», puntuación y ruido. Si falla una palabra, editarla. Para comprobar la edición aun con ASR correcto, añadir una aclaración propia y verificar que cambia el texto analizado.
5. Analizar con LR, SVM y BETO configurado. Confirmar `motor_asr=faster-whisper`, transcripción original conservada, texto editado realmente analizado, `fue_corregida=true`, modalidad `audio_transcrito`, diez eventos y modelo efectivo. Si aparece `ASR_SIMULADO_TEST`, no es una prueba real.
6. Cargar además un archivo de voz propia y repetir. Cambiar audio/idioma debe limpiar el estado anterior. Probar silencio y cancelar una entrada: deben producir un error explicable, sin reutilizar la transcripción previa.

Registrar privadamente versión, sistema, paquetes, duración, campos verificados y resultado del ensayo. No subir audio, transcripción ni rutas personales a GitHub. La prueba no mide WER ni calidad predictiva con turistas reales.

## Repetir contratos y leer resultados

```powershell
.\.venv-research\Scripts\python.exe scripts/run_academic_acceptance.py functional --registration-commit e20326484aa33d335e85bf0f8551c24adbbd13e1 --beto-artifact $env:TRAVELER_BETO_ARTIFACT --output artifacts/academic-functional-new.json
.\.venv-research\Scripts\python.exe scripts/acceptance_gradio_http.py --registration-commit e20326484aa33d335e85bf0f8551c24adbbd13e1 --beto-artifact $env:TRAVELER_BETO_ARTIFACT --output artifacts/academic-http-new.json
```

Omitir `--beto-artifact` ejecuta únicamente clásicos, sin atribuirles resultados BETO. Los destinos existentes no se sobrescriben. Los casos predictivos se registran por separado; una reproducción posterior requiere `--reproduce` y no se llama test intacto. CI ejecuta regresiones y contratos HTTP en Linux/Windows, sin descargar checkpoints, entrenar BETO ni repetir mediciones predictivas para ajustar nada.
