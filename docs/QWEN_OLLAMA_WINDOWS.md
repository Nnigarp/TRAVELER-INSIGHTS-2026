# Activar Qwen3 1.7B en Windows

Desde la raíz del repositorio, con Gradio cerrado y Ollama iniciado. Se reutilizan
`.venv-research`, el modelo `qwen3:1.7b` ya instalado y los pesos BETO V3 existentes.
No hay entrenamiento ni descarga de modelos en estos pasos.

```powershell
git fetch origin
git switch feat/qwen-ollama-tourism-report-20261117
git pull --ff-only
.\.venv-research\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path '.\src').Path
$env:TRAVELER_BETO_ARTIFACT = (Resolve-Path '.\artifacts\beto-v3-pr28-local').Path
$env:TRAVELER_OLLAMA_ENDPOINT = 'http://127.0.0.1:11434'
$env:TRAVELER_OLLAMA_MODEL = 'qwen3:1.7b'
$env:TRAVELER_OLLAMA_TIMEOUT = '120'
ollama list
Invoke-RestMethod "$env:TRAVELER_OLLAMA_ENDPOINT/api/version"
python scripts/check_qwen_local.py --classifier beto --output artifacts/qwen-windows-01.json
if ($LASTEXITCODE -ne 0) { throw 'Revisar el JSON: BETO + Qwen no han superado la comprobación.' }
python app.py
```

Si tu artefacto BETO tiene otro nombre, usa su carpeta instalada dentro de
`artifacts` o `data/private`. El comando falla si BETO recurre a LR o si Qwen no
responde con un informe validado. El comando no contiene respuestas fijas ni activa simulaciones.
En cada repetición, usa un nombre de salida nuevo para conservar la evidencia.

En Gradio (`http://127.0.0.1:7860`), selecciona **BETO supervisado** y marca
**Activar informe local Qwen3 1.7B (Ollama)**. Introduce:

> El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión.

Comprueba en JSON `modelo_utilizado: beto`, `seleccion_modelo.fallback: false`,
`generacion_qwen.estado: GENERADO`, `modelo_ejecutado: qwen3:1.7b` y
`validado_por: ActionAgent`. El informe debe identificar el transporte y proponer
comprobaciones sobre lo narrado. No se fija una etiqueta de sentimiento ni se
incluye una respuesta generativa de ejemplo. El clasificador decide la etiqueta.

Qwen desactivado conserva diez eventos; activado registra trece, incluidos la
consulta después de Control, su respuesta y la validación por Acción. Los eventos
solo contienen metadatos, no razonamientos internos. La latencia incluye la
carga inicial del modelo. Texto, transcripción corregida y lotes utilizan el mismo
adaptador; CSV/JSON conservan estado, modelo, latencia e informe. Las correcciones
humanas no regeneran ni cambian el informe original.

Si falla Ollama, se muestra el código del error y se conserva el análisis
convencional. `OLLAMA_DESCONECTADO`: inicia Ollama; `MODELO_NO_DISPONIBLE`:
comprueba `ollama list`; `TIMEOUT`: comprueba carga/memoria y vuelve a intentarlo.
El plazo se configura con `TRAVELER_OLLAMA_TIMEOUT` (máximo 600 s), sin modificar
presupuestos ni modelos científicos. JSON incompleto, modelo inesperado,
descripciones no literales y propuestas rechazadas nunca se publican como informe.
No hay reintentos ni sustitución del modelo generativo.

Para verificar las pruebas sin cargar modelos:

```powershell
python -m pytest -q tests/test_ollama_report.py tests/test_qwen_integration.py tests/test_qwen_gradio_http.py tests/test_qwen_local_check.py
```

Estas pruebas simulan la API; las de Gradio usan HTTP y sesiones reales. El
contrato BETO se prueba con una interfaz inyectada; no acredita inferencia con
pesos reales. La aceptación con Ollama/Qwen, BETO V3 y el micrófono físicos se
verifica en tu portátil con el comando anterior y Gradio. El resumen y el problema
son extractivos para limitar hechos inventados; las propuestas generadas siguen
necesitando valoración humana. No se ejecuta ninguna acción externa.

Contrato API utilizado: [chat](https://docs.ollama.com/api/chat),
[JSON estructurado](https://docs.ollama.com/capabilities/structured-outputs),
[separación de thinking](https://docs.ollama.com/capabilities/thinking).
