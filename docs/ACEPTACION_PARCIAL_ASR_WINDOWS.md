# Registro de aceptación parcial del Incremento 07 — Windows (2026-11-05)

## Evidencia reportada por la persona que ejecutó el prototipo

- Gradio en localhost recibió grabación real por micrófono.
- Modelo faster-whisper tiny procesó audio real de 5,46 segundos en español tras corregir incompatibilidad de PyAV.
- Se realizó segunda grabación real de 8,82 segundos; la transcripción original se conservó y el usuario editó la reseña antes de ejecutar el análisis.
- Se verificó análisis con regresión logística y con SVM lineal del agente existente.
- La segunda salida JSON confirmó modalidad audio, WAV PCM16, idioma español, modelo tiny y campos transcripcion_original, transcripcion_editada y fue_corregida=true.
- Se detectaron aspectos y conectores mediante heurísticas; **las etiquetas resultantes no son métricas de rendimiento científico**.
- La corrección de dependencia av>=11,<19 resolvió el TypeError observado durante decodificación con faster-whisper 1.2.1 y PyAV 19.0.2.
- El usuario compartió capturas en la conversación; no se incorporan capturas, audios, transcripciones ni hashes privados al repositorio.

## Validación pendiente antes de fusionar

- [x] Audio de micrófono real transcrito.
- [x] Corrección humana de texto preservando original.
- [x] Análisis con LR y SVM.
- [x] CI con ASR simulado sobre Python 3.11 y 3.12.
- [ ] Carga de un **archivo** de audio compatible y transcripción real.
- [ ] Revisión final de seguridad y privacidad de caché temporal de Gradio.
- [ ] Aprobación final de merge, manteniendo main estable.

**Estado:** integración funcional validada parcialmente con audio real. PR #6 sigue Draft; no fusionar antes de la prueba de archivos y revisión final.
