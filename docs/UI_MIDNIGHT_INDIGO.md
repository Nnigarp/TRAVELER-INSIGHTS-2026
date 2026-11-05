# Traveler Insights 2026 — Midnight Indigo (rediseño de presentación)

## Alcance del PR

Rediseño exclusivo de `app.py` y pruebas de conectividad visual. La identidad gráfica oficial es **Midnight Indigo**.

- Fondo claro `#F8FAFC`; tarjetas `#FFFFFF`; texto principal `#1E293B`; primario `#4F46E5`; acento `#818CF8`.
- El tema nativo de Gradio define colores oscuros independientes y no fuerza CSS fijo sobre textos, tablas, formularios ni componentes.
- CSS estructural mínimo para el encabezado, con variables del tema; se mantiene la visualización responsive que ofrece Gradio.
- Encabezado descriptivo con avisos destacados de prototipo, puntuaciones no calibradas y modelos entrenados con ejemplos sintéticos.
- Tres pestañas idénticas: reseña individual, audio y transcripción local, lotes y revisión humana.
- El resumen Markdown presenta sentimiento, modelo, puntuaciones no calibradas, aspectos exploratorios, revisión humana y recomendación; la edición del audio se indica cuando aplica.
- JSON completo disponible en acordeón cerrado por defecto, sin alterar ninguna clave.
- Tablas, tendencias, corrección humana, exportaciones, cambios de idioma/audio, callbacks y estados sin cambios de contrato.

## Compatibilidad del tema

Gradio 4/5 aplica `theme` y `css` a `Blocks`; Gradio 6 los recibe en `launch()`. Se selecciona la ubicación por versión mayor usando el mismo objeto `THEME`. Los fondos de modo claro y oscuro están definidos por separado; la opción **System** es gestionada por las preferencias del navegador y Gradio.

## Prueba manual Windows (Python 3.12)

```powershell
git fetch origin
git switch --track origin/feat/ui-midnight-indigo
.\.venv\Scripts\python.exe -m pytest -q
$env:TRAVELER_ASR_ALLOW_DOWNLOAD = "0"
.\.venv\Scripts\python.exe app.py
```

Abrir http://127.0.0.1:7860 y verificar:

1. Cambiar apariencia Light/Dark/System desde Gradio o preferencias del navegador. Verificar legibilidad de textos, botones, tablas y JSON.
2. Analizar la reseña mixta ilustrativa con LR y SVM; el resultado visible es un resumen, mientras el JSON mantiene todas las claves.
3. Grabar audio propio o subir WAV PCM16 compatible; transcribir, editar y comprobar procedencia en el acordeón.
4. Ejecutar un lote, corregir etiqueta sin alterar predicción, exportar JSON y CSV.
5. Comprobar responsive en ventana estrecha y navegación por teclado con foco visible.
6. No afirmar compatibilidad OGG/Opus de WhatsApp: existe un fallo observado cuya causa no se ha diagnosticado.

## Criterios no modificados

No se altera ningún modelo, regla, corpus, benchmark, política de derechos, archivo de datos, dependencia o workflow. Las puntuaciones siguen siendo internas y no calibradas externamente. No se integra React, chatbot, RAG, BERTopic ni otras funciones nuevas.
