# Incremento 07 — Audio y ASR local (2026-11-04)

## Funcionalidad implementada
- Pestaña Gradio Audio con micrófono y carga, conversión a WAV antes de llamar al backend. La conversión depende de los códecs disponibles localmente; no se garantiza cualquier archivo original.
- Verificación del WAV procesado: PCM 16 bits mono/estéreo, 8–48 kHz, 0,25–120 segundos y máximo 12 MiB. Se comprueba truncamiento y silencio digital exacto.
- Reconocimiento local opcional mediante faster-whisper CPU int8 (dos hilos y un trabajador) para español, inglés, italiano o detección automática. Sin descargas implícitas; se necesita el modelo local o autorizar explícitamente descarga inicial.
- Transcripción editable y análisis mediante LR/SVM existentes. Conserva original, corrección, idioma, motor ASR y SHA-256 del WAV transformado, sin copiar audio ni rutas al JSON.
- Cambio de audio o idioma limpia la transcripción y resultado anterior.
- Tests unitarios y callbacks con ASR simulado: CI no descarga pesos.

## Instalación (CMD Windows)

Desde la raíz del proyecto, con entorno virtual Python 3.11/3.12:

~~~cmd
.venv\Scripts\python.exe -m pip install -e ".[ui,test,asr]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe app.py
~~~

El complemento opcional ASR puede requerir runtimes nativos compatibles (PyAV/CTranslate2); el análisis de texto funciona sin instalarlo.

### Primera descarga con consentimiento explícito

Por defecto se buscan pesos locales y **no** se conectará a la red para buscarlos. Si decides autorizar la descarga del modelo tiny en la sesión actual (requiere Internet, espacio y condiciones aplicables), ejecuta:

~~~cmd
set TRAVELER_ASR_ALLOW_DOWNLOAD=1
set TRAVELER_ASR_MODEL=tiny
.venv\Scripts\python.exe app.py
~~~

La descarga se iniciará al solicitar una transcripción. Una vez guardado el modelo, vuelve al modo sin descarga reiniciando Gradio:

~~~cmd
set TRAVELER_ASR_ALLOW_DOWNLOAD=
.venv\Scripts\python.exe app.py
~~~

Alternativamente configura la carpeta de un modelo local:
~~~cmd
set TRAVELER_ASR_MODEL_PATH=C:\ruta\a\modelo
~~~

Gradio se abre en localhost y share=False. El audio no se envía a APIs externas de transcripción durante la inferencia; una descarga inicial autorizada sí contacta repositorios externos.

## Prueba manual

1. En Audio y transcripción local, permitir micrófono en navegador y grabar 5–10 s con una reseña corta en español; transcribir.
2. Editar la transcripción, corrigiendo una negación reconocida incorrectamente; analizar con LR. Verificar el JSON: transcripcion_original intacta, transcripcion_editada final, fue_corregida true.
3. Cambiar el selector a SVM y analizar de nuevo; verificar modelo_utilizado svm.
4. Subir audio compatible; probar inglés e italiano y validar idioma solicitado.
5. Probar silencio, error de archivo y audio largo; comprobar errores controlados.
6. Cambiar audio o idioma; confirmar que se limpian estado y resultados.

## Privacidad, riesgo y limitaciones
- No detecta emociones a partir de la voz. No afirma comprender negaciones, sarcasmo ni ironía; estas pueden perderse por error de ASR.
- El VAD filtra segmentos sin voz pero también puede perder palabras; ningún resultado está científicamente validado.
- El hash es del WAV convertido, no necesariamente del archivo originalmente subido. Gradio no informa siempre de si el audio se grabó o cargó: no se inventa esa distinción en la trazabilidad.
- Los límites se aplican después de que Gradio convierta el archivo, no necesariamente antes. Evitar subir audios grandes; Gradio puede dejar archivos temporales en caché. No compartir voz privada sin consentimiento.

## Mantenimiento pendiente de GitHub Actions (no abordado aquí)
Los logs informan de compatibilidad de Node.js para actions/checkout@v4 y actions/setup-python@v5. Actualizarlos en un PR separado a versiones compatibles (por ejemplo checkout@v5, setup-python@v6 tras verificar sus requisitos) y validar CI; considerar fijación por SHA.
