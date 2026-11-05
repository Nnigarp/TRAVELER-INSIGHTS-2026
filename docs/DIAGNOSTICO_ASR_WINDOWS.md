# Incidente ASR Windows — causa identificada y corrección de dependencia (2026-11-05)

## Diagnóstico confirmado en Windows

Micrófono, Gradio, WAV, importación PyAV y CPU INT8 funcionan; `pip check` no identificó conflictos declarados. En PowerShell, el backend reportó:

```text
ASR_STAGE=decode ERROR_TYPE=TypeError
function=_decode_local_audio
function=decode_audio
function=av.container.core.open
```

Versiones observadas: Python 3.12.7, faster-whisper 1.2.1 y av/PyAV 19.0.2. **Causa concreta:** faster-whisper 1.2.1 invoca `av.open(..., metadata_errors="ignore")`; PyAV 19 eliminó ese argumento. El caso coincide con la incidencia upstream https://github.com/SYSTRAN/faster-whisper/issues/1589 y el registro de cambios de PyAV 19.0.2.

**Corrección aplicada:** el extra `asr` de `pyproject.toml` limita `av>=11,<19`. Así evitamos cambiar VAD, CTranslate2, modelo, calidad de voz o toda la arquitectura por intuición. La CI con mocks comprueba el límite de dependencia y reproduce el error de keyword sin descargar pesos. **Aún falta comprobar una transcripción real en el equipo Windows con PyAV compatible.**


## Aplicar el arreglo demostrado — PowerShell Windows

Cerrar Gradio (`Ctrl+C`); sincronizar la rama, reinstalar el complemento opcional respetando el límite y comprobar:

~~~powershell
git fetch origin
git switch feat/audio-local-asr
git pull --ff-only origin feat/audio-local-asr
.\.venv\Scripts\python.exe -m pip install -e ".[ui,test,asr]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m traveler_agent.audio_asr --check
.\.venv\Scripts\python.exe -m pytest -q
$env:TRAVELER_ASR_ALLOW_DOWNLOAD = "0"
$env:TRAVELER_ASR_MODEL = "tiny"
.\.venv\Scripts\python.exe app.py
~~~

Se espera `av` en versión inferior a 19 (por ejemplo 18.1.0); si el resolver no lo respeta, instalar explícitamente `"av==18.1.0"` y repetir `pip check`. Grabación de 5–10 segundos, transcribir, editar y analizar. Si aparece **otro** fallo, conservar el nuevo `ASR_STAGE`, sin reinstalar paquetes a ciegas. No copiar audios ni transcripciones a los logs o incidencias.

## Cambios ejecutables de este PR

El backend separa cuatro etapas técnicas y registra en la terminal la **clase de excepción**, códigos numéricos Windows/OS cuando estén presentes, nombre de función y número de línea del traceback: model_import, model_load, decode y inference (con subtipos por módulo ctranslate2 o vad_inference). Evita emitir argumentos de las excepciones, rutas, texto transcrito o contenido del audio, pues podrían ser privados. Por ello el traceback está **reducido a ubicaciones y tipos**, y no contiene el mensaje de excepción original. Las pruebas de regresión cubren fallos simulados en cada etapa sin bajar pesos.

La decodificación explícita mediante faster_whisper.audio.decode_audio produce PCM NumPy antes de la inferencia real. Esto permite identificar si el error proviene de PyAV; los mocks siguen recibiendo una ruta para conservar la CI. No se ha desactivado VAD ni cambiado el modelo. El límite PyAV se impuso tras obtener evidencia directa de incompatibilidad.

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
