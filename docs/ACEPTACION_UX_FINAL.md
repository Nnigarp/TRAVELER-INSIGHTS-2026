# Aceptación final de interfaz y percepción

## Referencia histórica

El estudiante confirmó en el Lenovo la base integrada del PR #34, `cb08717`, con 606 pruebas aprobadas, un xfail histórico y 27 advertencias en 112,44 s. La activación de BETO V3 y BETO + Qwen pasaron con cero violaciones y sin fallback. El informe local es `artifacts/postmerge-qwen-20261125-141735.json`. La ejecución anterior de 588 pruebas correspondía a `c08f433` con configuración incompleta.

En la reseña «El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión», LR, SVM y BETO clasificaron negativo. Los scores no calibrados fueron 0,4592, 0,4159 y 0,9966. Control solicitó revisión para LR y SVM. Los tres informes con Qwen real fueron validados por Acción.

| Prueba escrita del PR #34 con BETO efectivo | Score negativo | Traducción | Revisión |
|---|---:|---:|---|
| Español | 0,9966 | 0 ms | No |
| Italiano | 0,9912 | 4730,66 ms | No |
| Inglés | 0,9740 | 2041,13 ms | No |
| Francés | 0,9533 | 2029,23 ms | No |
| Chino | 0,6671 | 706,24 ms | Sí |

El francés y el chino se tradujeron con el informe apagado. En francés, «nous avons manqué l'excursion» se convirtió en «nos faltó la excursión». En chino aparecieron «vehículo de albergamiento» y «actividad de visita». En aquel caso no hubo advertencia lingüística y Control activó `CONTRASTE_CON_SCORE_LIMITADO`. En italiano e inglés hubo citas conservadoras sin incidente identificado formalmente, aunque el aspecto acceso y la clasificación negativa estaban presentes. Son cinco casos funcionales individuales.

**Micrófono físico en la base anterior.** En español, Whisper Tiny detectó es en 8,52 s de audio. Conservó «exclusión» sin corrección. BETO obtuvo 0,5157 negativo, con servicio, higiene y acceso, revisión requerida e informe Qwen validado en 4029,67 ms. En italiano, «Il servizio era eccellente, ma il trasporto non è mai arrivato e abbiamo perso l'escursione» produjo «y el servicio era excelente, maí, transporto, no me maí, arriba a todo y habíamos perso y el escorsión». Whisper registró es en 8,34 s de audio. Se omitió la traducción, BETO dio 0,58 positivo y 0,4114 negativo, sin revisión. El informe fue conservador. El error observado comenzó en percepción.

## Qué cambia

El editor reúne escritura, contador, adjunto, micrófono, clasificador e informe. Los botones abren el control nativo de Gradio en la modalidad elegida. **Micrófono → Grabar** inicia la captura en el navegador y **Detener** envía a Whisper Tiny. El panel permite reproducir, descartar o elegir otro idioma y **Retranscribir y sustituir el texto**. Esa sustitución es explícita. Los ejemplos y Limpiar permiten iniciar otra entrada.

La etapa previa a clasificación guarda la probabilidad de idioma automático cuando Whisper la devuelve, y por segmento los valores disponibles de log-verosimilitud media, compresión y probabilidad de ausencia de voz. Los umbrales de decodificación se leen de la ejecución. Una probabilidad de idioma de hasta 0,5 señala ausencia de mayoría estimada. Es una regla operativa experimental. La selección manual de idioma no proporciona confianza automática. Se contrastan el texto y el idioma informado y se reconoce el patrón mixto del fallo italiano documentado. Las señales solicitan escuchar, corregir o retranscribir y confirmar. Qwen sigue identificando texto incierto y traduciendo su contenido. El sistema carece de evidencia para reconstruir mediante Qwen qué se dijo realmente.

La traducción se puede corregir por separado. Se añadieron avisos acotados para la pérdida de la negación china «从未», el caso francés de perder la excursión y el traslado chino traducido como «albergamiento». Una traducción válida en estructura puede contener otros errores. Un score alto tampoco acredita una buena transcripción.

Agentes resume modalidad, idioma, clasificador, traducción, informe, revisión y eventos observados. El diagrama sitúa a TravelerOrchestrator sobre las etapas. La cronología y el JSON conservan eventos y contratos. Las advertencias de percepción ocurren antes de Semántica y de Control. Los diez eventos principales y los tres de informe opcional siguen perteneciendo al orquestador.

## Comprobar en PowerShell

```powershell
Set-Location 'C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github'
git switch feat/final-ux-human-review-20261125
.\scripts\iniciar_demo.ps1
```

El script imprime las versiones locales (referencia: Python 3.12.10 y Gradio 6.17.3), comprueba el manifiesto y la disponibilidad de `qwen3:1.7b` en el endpoint local. Abre la dirección que muestra Gradio. Si Ollama está detenido, inicia la aplicación Ollama de Windows y comprueba `ollama list`.

Tras detener Gradio con Ctrl+C:

```powershell
$env:TRAVELER_BETO_ARTIFACT = (Resolve-Path 'artifacts/beto-v3-pr28-local').Path
$env:HF_HUB_OFFLINE = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$marca = Get-Date -Format 'yyyyMMdd-HHmmss'
.\.venv-research\Scripts\python.exe -m pytest -q
.\.venv-research\Scripts\python.exe scripts/check_beto_activation.py --artifact $env:TRAVELER_BETO_ARTIFACT --output "artifacts/ux-beto-$marca.json"
.\.venv-research\Scripts\python.exe scripts/check_qwen_local.py --classifier beto --output "artifacts/ux-qwen-$marca.json"
```

**Aceptación personal.** Escribe y selecciona texto en ambos temas. Dicta las frases española e italiana, reproduce la toma, revisa el idioma y corrige «exclusión» a «excursión» si aparece. En JSON compara los textos, los hashes y los indicadores de edición. Prueba la retranscripción sobre la misma toma. Abre Agentes con y sin informe. En Lotes, corrige una etiqueta y verifica las dos columnas de predicción y corrección en el CSV. La calidad de voz humana en este incremento queda pendiente de aceptación física.

**Rollback.** Detén Gradio, ejecuta `git switch main` y comprueba que la referencia estable sea `cb08717`. Con las variables de entorno anteriores, ejecuta `.\.venv-research\Scripts\python.exe app.py`. El script de arranque nuevo pertenece a la rama del incremento.

## Evidencia visual y rúbrica

[Análisis anterior](acceptance/2026-11-25-final/analysis-before.png) · [Análisis actual](acceptance/2026-11-25-final/analysis-after.png) · [Editor en reposo](acceptance/2026-11-25-final/composer.png) · [Agentes anterior](acceptance/2026-11-25-final/agents-before.png) · [Agentes actual](acceptance/2026-11-25-final/agents-after.png) · [Comprobaciones de Chromium](acceptance/2026-11-25-final/browser-check.json).

Para la sustentación, presenta el caso del transporte desde la perspectiva de quien revisa la experiencia del visitante. La percepción reúne voz, texto y procedencia. El razonamiento sigue un plan jerárquico con evidencias y un clasificador supervisado. Control decide si requiere revisión; Acción valida una propuesta que el usuario puede inspeccionar. La memoria episódica de sesión conserva la edición, la decisión y los eventos. Las correcciones no provocan aprendizaje en línea. Compara finalmente el caso completo con el fallo italiano para explicar sesgos de idioma, ruido y límites de generalización.

