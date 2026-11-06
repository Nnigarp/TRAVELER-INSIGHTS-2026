# Vista multiagente — Midnight Indigo

**Base exacta:** main SHA 82a125494983dd327dcbb44cd03d8f22a5ff59a3, incorpora los PR #13 y #15.

## Regresión previa y congelación

GitHub Actions run 42183447525 (Python 3.12): **171 passed, 1 xfailed** sobre ese SHA; éxito. La revisión de código confirma que el orquestador, Gradio, audio, lotes y módulos científicos coexisten. CI no equivale a comprobación física de micrófono ni de navegadores Windows.

Los benchmarks permanecen congelados: primer conjunto sintético 336 reseñas, semilla 42, Macro-F1 test LR/SVM 1,0000; desafío sintético 1.260 textos de 180 núcleos, semilla 913, LR 0,3926 y SVM 0,3882. Ninguna cifra mide turismo humano real; este PR no altera sus datos, modelos o métricas.

## Qué cambia

- Nuevo módulo de render puro: src/traveler_agent/multiagent_view.py, lectura exclusiva del resultado de inferencia.
- En app.py, analyze_with_agents envuelve el callback analyze original y devuelve exactamente sus dos salidas intactas (Markdown, JSON) más un HTML visual. Se ejecuta **una sola inferencia**; no se modifican políticas, clasificación ni trazabilidad JSON.
- La pestaña Reseña individual agrega un gr.HTML con tarjeta de TravelerOrchestrator, cuatro tarjetas especializadas, estado final, motivos de revisión, recomendación y los eventos ordenados. El acordeón JSON completo se conserva. Audio, lotes, correcciones y exportaciones no se modifican.
- El CSS usa variables del tema existente Midnight Indigo, grid responsive y sin librerías nuevas.

## Interpretación de datos y privacidad

- Cada tarjeta muestra OK o FAILED solamente si la traza registra ese estado. DELEGATED se muestra si hay delegación registrada sin respuesta del especialista. En los demás casos se marca **Sin ejecutar**.
- Los eventos no se sintetizan a partir del plan; se presentan en orden registrado por el campo sequence.
- HTML muestra solamente nombres de agentes/tareas/estados y códigos de nota incluidos en listas permitidas. No replica reseñas, transcripciones, rutas privadas ni valores internos de modelo en timeline.
- Los textos de recomendaciones se escapan con html.escape y no se inyectan como HTML.
- Un error en la API del agente puede producir excepción y JSON vacío: en ese caso **no existe una traza de fallo retornada**. La vista dice Sin traza disponible; jamás representa ejecución completada. Si un registro verificable contiene DETENIDO_SEGURO y FAILED, puede mostrarlo sin inventarlo.
- Puntuaciones DEMO no calibradas; memoria episódica por ejecución, aprendizaje supervisado offline y planificación fija, no autónoma general.

## Matriz de rúbrica UTB

| Criterio | Evidencia | Límite |
| --- | --- | --- |
| Problema/usuario | reseña turística y operador que revisa | sin validación en usuarios reales |
| Percepción | recepción, validación, modalidad texto/transcrito | ASR previo; no detecta emociones vocales |
| Planificación | PLANNED y delegaciones reales | plan fijo, no planificación general |
| Razonamiento | semántica y controles del orquestador | reglas y scores no calibrados |
| Decisión | CONTINUAR o REVISION_HUMANA | revisión guiada por umbrales demostrativos |
| Acción | sugerencia o abstención | sin acciones turísticas externas |
| Memoria | estado y traza efímeros | no memoria persistente |
| Aprendizaje | modelos LR/SVM entrenados previamente | no aprendizaje en línea |
| Caso completo | entrada, 4 delegaciones, resultados, cierre | validación humana externa pendiente |
| Limitaciones | scores DEMO, errores de generalización | métricas solo sintéticas |

## Prueba de aceptación en Windows

Desde el repositorio existente, proteger cambios locales y ejecutar:

~~~powershell
git fetch origin
git switch --track origin/feat/multiagent-visualization
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe app.py
~~~

En http://127.0.0.1:7860, analizar la reseña «El guía fue amable, pero esperamos demasiado para entrar». Comparar visualmente plan, delegaciones, estado y motivos con Trazabilidad completa · JSON. Verificar Light, Dark y System; ventana móvil; voz y lotes en sus pestañas originales. El fallo OGG/Opus de WhatsApp continúa sin diagnóstico de compatibilidad. No confundir un check verde de CI con esta comprobación manual.

**No fusionar automáticamente: integración y aceptación manual corresponden al responsable del proyecto.**
