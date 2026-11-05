# Incidente ASR Windows — diagnóstico reproducible (2026-11-05)

## Incidente observado, todavía sin causa raíz identificada

Micrófono y Gradio funcionan y el WAV supera la validación; el error genérico se producía en la ejecución de transcripción de faster-whisper. **No hay aún un traceback del equipo Windows**, por lo que no se puede atribuir la incidencia a CPU, VAD, PyAV, CTranslate2, versiones, caché, ruido ni modelo descargado.

## Cambios ejecutables de este PR

El backend separa cuatro etapas técnicas y registra en la terminal la **clase de excepción**, códigos numéricos Windows/OS cuando estén presentes, nombre de función y número de línea del traceback: model_import, model_load, decode y inference (con subtipos por módulo ctranslate2 o vad_inference). Evita emitir argumentos de las excepciones, rutas, texto transcrito o contenido del audio, pues podrían ser privados. Por ello el traceback está **reducido a ubicaciones y tipos**, y no contiene el mensaje de excepción original. Las pruebas de regresión cubren fallos simulados en cada etapa sin bajar pesos.

La decodificación explícita mediante faster_whisper.audio.decode_audio produce PCM NumPy antes de la inferencia real. Esto permite identificar si el error proviene de PyAV; los mocks siguen recibiendo una ruta para conservar la CI. No se ha desactivado VAD, cambiado el modelo ni impuesto nuevas versiones de paquetes por conjeturas.

## Procedimiento en Windows CMD (desde la raíz del repositorio)

Trabajar en la rama del PR #6 y con el mismo entorno virtual usado por Gradio:

~~~cmd
git fetch origin
git switch feat/audio-local-asr
git pull --ff-only origin feat/audio-local-asr
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m traveler_agent.audio_asr --check
.venv\Scripts\python.exe -m pytest -q
set TRAVELER_ASR_ALLOW_DOWNLOAD=
set TRAVELER_ASR_MODEL=tiny
.venv\Scripts\python.exe app.py
~~~

El comando --check produce versiones instaladas de faster-whisper, ctranslate2, av, onnxruntime, tokenizers, huggingface-hub, numpy y Gradio, así como los tipos de cómputo soportados por CTranslate2 en CPU y una prueba de importación de PyAV. **No carga modelo ni audio ni hace llamadas de red**.

Grabar una frase corta y pulsar Transcribir localmente. En la **terminal original de CMD**, identificar ASR_STAGE, ERROR_TYPE, WINERROR/ERRNO y la cadena de funciones/líneas, pero no la voz ni los datos privados.

### Interpretación sin suposiciones

| Etapa | Qué ha fallado | Qué verificar después |
| --- | --- | --- |
| model_import | importación de faster-whisper, PyAV o dependencias | pip check, versiones, runtimes Windows, DLL |
| model_load | lectura o inicialización del modelo tiny | disponibilidad local, integridad/caché, arquitectura y memoria |
| decode | PyAV no decodifica el WAV | formato/códec del WAV convertido, versión de PyAV, compatibilidad |
| ctranslate2 | error cuya clase procede de CTranslate2 | soporte CPU/INT8, DLL y runtime nativo |
| vad_inference | error cuya clase procede de onnxruntime | versión/instalación ONNX Runtime y VAD |
| inference | fallo restante de inferencia o recorrido diferido | funciones de traceback, dependencias; no concluir que es ruido |

La detección por nombre del módulo de excepción es orientativa: una excepción genérica RuntimeError emitida por una biblioteca nativa puede seguir categorizada como inference. Para precisar la causa hará falta el mensaje del error original inspeccionado localmente con control de privacidad o evidencia adicional.

### Comprobación de dependencias, sin reinstalaciones por intuición

~~~cmd
.venv\Scripts\python.exe -m pip show faster-whisper ctranslate2 av onnxruntime tokenizers huggingface-hub
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -c "import ctranslate2; print(sorted(ctranslate2.get_supported_compute_types('cpu')))"
.venv\Scripts\python.exe -c "import av; print('PyAV importa correctamente')"
~~~

Requisitos declarados por faster-whisper upstream incluyen Python 3.9+, ctranslate2 4.x, av >=11, onnxruntime >=1.14, tokenizers >=0.13, huggingface-hub >=0.23. Son referencias generales: el entorno real requiere verificación por pip check y los valores efectivamente instalados. No bajar ni subir versiones ni instalar CUDA sin un error concreto; esta configuración es CPU INT8. Documentación oficial: https://github.com/SYSTRAN/faster-whisper

### Evidencia mínima para un correctivo de causa raíz

Compartir solo las líneas de ASR_STAGE, ERROR_TYPE, WINERROR, ERRNO, MODULE y funciones del traceback; y el reporte de versiones sin rutas personales. **No compartir audio, transcripciones, cachés, tokens ni rutas completas**. Con esos datos se formula la hipótesis y se realiza un cambio mínimo con regresión real.

## Criterios de cierre

- CI Python 3.11/3.12 verde con mocks y regresiones de logging.
- Diagnóstico real en Windows que identifica la etapa y causa técnica del fallo.
- Al menos una transcripción real exitosa en Windows, con edición manual y LR/SVM.
- El PR #6 sigue Draft y **no debe fusionarse** hasta entonces.
