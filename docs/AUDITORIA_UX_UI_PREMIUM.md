# Auditoría UX/UI de Traveler Insights 2026

La revisión parte de `main` en `906471b` (PR #32 integrado), del prototipo HTML y de las capturas de ejecución en Windows. La franja lateral es visible en los dos temas. Las capturas no indican el zoom efectivo y no permiten atribuir la escala exacta a Windows o al navegador.

| Hallazgo | Causa en la interfaz del PR #32 | Prioridad, esfuerzo y riesgo |
|---|---|---|
| Área útil alineada a la izquierda y franja amplia a la derecha | El límite de 1.500 px está en `.gradio-container`. Gradio expande su envoltorio exterior, mientras el CSS elimina el centrado del contenedor interior. La captura reproduce la asimetría. | Alta · bajo · bajo |
| Fondo oscuro y texto claro residual en UTB Claro | La capa `gradio-app` queda fuera de las variables sobre `.gradio-container`. El tema nativo también puede derivarse del esquema del sistema. El selector debe sincronizar las capas y pintar un fondo común. | Alta · medio · bajo |
| Lectura deficiente | Etiquetas, avisos, indicadores y trazas se fijaron en 10–11 px. La división fija 38/58 reduce el espacio legible en resoluciones intermedias. | Alta · medio · bajo |
| JSON con barra horizontal en el acordeón | CodeMirror tiene líneas sin ajuste; el acordeón recibe la presión del ancho intrínseco. Limitar el editor al ancho disponible y ajustar líneas visualmente conserva el valor para copiar y descargar. | Alta · bajo · bajo |
| Controles poco fáciles de descubrir | La selección en Análisis afecta Audio y Lotes, pero no está a la vista allí. Los botones de carga, revisión y exportación necesitan mostrar su relación con el resultado. | Media · bajo · bajo |
| Resultado y tareas del mismo peso visual | Se repite la recomendación tradicional y se mezclan ayuda, nombres de modelos y estados. La estructura de agentes en el estado vacío enumera roles, pero no muestra el orden ni el punto de intervención de Qwen. | Media · medio · bajo |

## Decisiones para este incremento

1. **Usabilidad.** Contenedor fluido, áreas de lectura alineadas y tipografía base de 15–16 px. Ambos temas mantienen la jerarquía y un foco visible.
2. **Descubrimiento.** Hacer visible la configuración compartida en Audio y Lotes, agrupar exportación y revisión, y presentar el flujo desde el orquestador hasta la salida.
3. **Simplificación.** Quitar insignias promocionales, mostrar una sola recomendación en la salida por reglas y mantener detalles y trazas bajo demanda.
4. **Cambios de bajo riesgo.** Tema claro predeterminado con preferencia local, cuadrícula adaptada al ancho real del informe, descarga JSON nativa y estados de procesamiento de la cola de Gradio.
5. **Patrones de dashboard.** Entrada acotada y resultado prioritario, resumen arriba, indicadores con etiqueta y valor, detalles progresivos y tablas con desplazamiento propio. Las cifras de lotes son conteos observados.
6. **Límites de Gradio.** Los componentes nativos incluyen controles de carga con textos dependientes de la versión e interfaz del navegador. Las pestañas pueden recurrir al menú de desbordamiento en anchos muy estrechos. El editor virtualiza JSON largo, por lo que la verificación de integridad debe usar la descarga.
7. **Futuro.** Evaluar filtros de tabla, navegación por URL, selección de reseñas con historial y pruebas de usabilidad con docentes antes de ampliar el alcance. Son cambios de interacción y estado que requieren evaluación separada.

El resultado del navegador se valida con inferencia DEMO de LR y SVM. Qwen, BETO y Whisper reales se aceptan por separado en el Lenovo del usuario.
