# Validación funcional de Windows — 5 de noviembre de 2026

Versión: main SHA 290921090555f634723b82dc611a8904809c01ea.
Estado: **validación funcional básica completada**; no implica validez científica.
Origen de las observaciones: pruebas reportadas personalmente por el usuario; no se publican audios, textos privados ni logs íntegros.
CI confirmada: GitHub Actions 41993135581, Python 3.12, 140 passed, 1 xfailed, success.

## Entorno y pruebas

- Windows, Python 3.12 operativo; instalación de dependencias y pip check correctos.
- Suite local: 140 passed, 1 xfailed (reportado por operador).
- Gradio: http://127.0.0.1:7860 accesible.
- Caso LR: «El guía fue amable, pero esperamos demasiado para entrar». Negativo, puntuación máxima **no calibrada** 0.393; servicio y espera; contraste; revisión humana. Aprobado funcionalmente.
- Caso SVM: mismo texto. Negativo, puntuación máxima **no calibrada** 0.3813; servicio y espera; contraste; revisión humana. Aprobado funcionalmente.
- Micrófono real: 7.32 segundos, español, faster-whisper tiny transcribe; SVM predice positivo, detecta servicio e higiene, contraste y revisión humana. Aprobado funcionalmente.
- Archivo OGG/Opus procedente de WhatsApp: Gradio muestra «Error». **FALLÓ**; no se ha diagnosticado el motivo. No prometer compatibilidad.
- El mismo archivo convertido a WAV PCM16, 16 kHz, mono: carga y transcripción correctas, duración 8.733 segundos; LR positivo, servicio/espera, revisión humana, trazabilidad. Aprobado funcionalmente.
- Lote de tres reseñas: demo-001 positivo, demo-002 negativo, demo-003 neutral; tabla, tendencias y exportaciones JSON/CSV correctas.
- Corrección humana: demo-001 recibió etiqueta neutral únicamente para verificar la funcionalidad; no es etiqueta gold y no debe incorporarse al conjunto de entrenamiento o test. La predicción original se conserva.

## Límites y defectos abiertos

1. Fallo OGG/Opus sin diagnóstico técnico. La compatibilidad confirmada se limita a WAV transformado compatible.
2. Detección léxica incompleta: las expresiones «habitaciones limpias» y «tuvimos que esperar mucho tiempo» no fueron detectadas en casos reportados. El extractor no es ABSA entrenado.
3. Los LR/SVM de la UI son modelos de **demostración sobre ejemplos ficticios**; puntuaciones no calibradas externamente.
4. No se han calculado métricas científicas sobre opiniones turísticas auténticas, ni evaluación independiente en español.
5. Los filtros de privacidad del corpus no garantizan anonimización; permisos originales sin confirmar.
6. No existe lockfile completo y el lanzador INICIAR_WINDOWS.bat conserva py -3.11 cuando falta el entorno. Usar Python 3.12 explícitamente en PowerShell.

## Demostración interna, viernes 6 de noviembre de 2026

1. Confirmar SHA de main, versión de Python, pip check, pytest y estado de Gradio.
2. Mostrar reseñas ilustrativas con LR y SVM, aspectos, evidencias, revisión humana y recomendación.
3. Grabar voz real, corregir transcripción y comparar procedencia original/editada.
4. Cargar WAV PCM16 válido; no mostrar OGG/Opus como capacidad confirmada.
5. Procesar un lote de tres reseñas, demostrar corrección de etiqueta solo en sesión y exportaciones CSV/JSON.
6. Mostrar corpus y benchmark sin presentar pruebas sintéticas como evidencia turística.
7. Explicar siguiente fase: datos autorizados, procedencia/privacidad, test cerrado por grupos, Dummy/LR/SVM y evaluación real.

La CI del commit indicado está disponible en:
https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/41993135581
