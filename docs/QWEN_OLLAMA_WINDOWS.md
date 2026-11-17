# Qwen3 1.7B en Windows: integración opcional y aceptación

Se reutilizan Python 3.12, `.venv-research`, Ollama, `qwen3:1.7b` y el artefacto
BETO V3 `artifacts/beto-v3-pr28-local` que el responsable ya confirmó en Windows.
No entrenar BETO ni reconstruir los pesos para esta integración.

## 1. Preparar rama y servicio local

Cerrar Gradio antes de actualizar. Desde la raíz del repositorio:

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
ollama --version
ollama list
Invoke-RestMethod "$env:TRAVELER_OLLAMA_ENDPOINT/api/version"
```

Si el servicio está desconectado, abrir Ollama desde Inicio; alternativamente,
ejecutar `ollama serve` en otra terminal. No iniciar un segundo servidor si ya
está activo. **Solo si falta el modelo en `ollama list`**, instalarlo una vez:

```powershell
ollama pull qwen3:1.7b
```

No se requiere una biblioteca Python de Ollama ni una API externa. Para medir
RAM se utiliza `psutil`, incluido en el entorno de investigación; si falta, el
comprobador lo marca pendiente. No estima RAM a partir del archivo del modelo.

## 2. Comprobar aplicación y modelos sin entrenar

```powershell
python scripts/check_qwen_local.py --classifier beto --output artifacts/qwen-windows-callback-01.json
if ($LASTEXITCODE -ne 0) { throw 'Leer el JSON: no se acreditó BETO + Qwen.' }
python app.py
```

La primera llamada puede cargar ambos modelos. El script no acepta `PASSED` si
BETO cae a LR, Qwen no responde, el informe es rechazado o no se verifica
versión/digest local de Ollama. No fuerza una clase de sentimiento. Cada
repetición exige otro nombre de salida; no sobrescribe evidencias anteriores.
El digest es el declarado por la API local, no una atestación remota.

## 3. Demostrar Gradio y guardar aceptación por HTTP

En `http://127.0.0.1:7860` seleccionar **BETO supervisado** y marcar
**Activar informe local Qwen3 1.7B (Ollama)**. Usar:

> El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión.

Comprobar en JSON:

- `modelo_utilizado = beto`, `seleccion_modelo.effective = beto`, `fallback = false`.
- `generacion_qwen.generacion_llm = true`, `estado = GENERADO`,
  `modelo_ejecutado = qwen3:1.7b`, `validado_por = ActionAgent`.
- El sentimiento, aspectos y revisión del informe coinciden con los originales.
  La cita del problema incluye el transporte; ninguna propuesta atribuye el
  transporte al hotel ni afirma una acción externa ejecutada.
- `multiagente.traza` conserva diez eventos. `traza_generacion` registra tres
  eventos independientes: solicitud tras Control, respuesta y validación de Acción.

En una **segunda terminal**, activar el mismo entorno y establecer las mismas
variables de Ollama. Con Gradio aún abierto:

```powershell
python scripts/check_qwen_local.py --classifier beto --gradio-url http://127.0.0.1:7860 --output artifacts/qwen-windows-http-01.json
if ($LASTEXITCODE -ne 0) { throw 'Aceptación HTTP pendiente; revisar violations y observed.' }
```

Este modo utiliza la cola y el callback del servidor Gradio real. La primera
prueba solo llama al callback de aplicación directamente; no confundir ambas.
El JSON registra commit, entorno, BETO efectivo, protocolo, parámetros, hash del
esquema, versión/digest de Ollama y latencia de generación y del recorrido.
Registra instantáneas de RAM disponible, RSS del comprobador y procesos Ollama
accesibles. **No son picos, memoria exclusiva por modelo ni memoria de GPU**;
las páginas compartidas pueden contabilizarse varias veces. En modo HTTP el
RSS del comprobador no es el del servidor Gradio. Complementar con Administrador
de tareas y anotar arranque frío frente a modelo cargado.

## 4. Audio, lote, corrección y exportación

Con Qwen habilitado, grabar la misma frase usando Whisper Tiny, comprobar la
transcripción original y corregirla antes de analizar con BETO. El informe usa
el texto **editado**, conserva procedencia ASR y revisión de Control. La prueba
real de esta combinación con micrófono se realiza en el portátil; el
funcionamiento anterior de Whisper fue confirmado por el responsable.

En Lotes introducir dos reseñas, analizar y exportar JSON/CSV. Se conservan estado,
modelo, informe, latencia y `generacion_llm`. Una corrección humana conserva el
informe original: interfaz y CSV indican que es anterior a la corrección; no cambia
la etiqueta del modelo ni se regenera automáticamente. JSON conserva predicción
y corrección separadas. No publicar exportaciones privadas ni notas identificables.

Para volver al comportamiento anterior, **desmarcar Qwen** y analizar de nuevo:
no se consulta Ollama ni se añade un informe generativo; LR sigue predeterminado.
Desmarcar no detiene el modelo ya cargado en Ollama. Opcionalmente,
`ollama stop qwen3:1.7b` libera su carga local al terminar la prueba.

## 5. Contrato, límites y errores explícitos

Generación restringida: Qwen compone JSON eligiendo citas completas y propuestas
de un catálogo versionado. Resumen/problema **no son paráfrasis libres**; el problema
puede quedar vacío. El catálogo no promete pagos ni identifica culpables. Una
expresión regular no puede certificar la veracidad de prosa arbitraria. El modelo
aún puede elegir una cita poco pertinente; la validación estructural no reemplaza
la revisión semántica humana.

| Control | Valor inicial / comportamiento |
|---|---|
| Activación | Checkbox desactivado; cero consultas al servicio |
| Endpoint | HTTP(S) de loopback, sin credenciales, rutas, proxies ni redirecciones |
| Modelo | `TRAVELER_OLLAMA_MODEL`, inicialmente `qwen3:1.7b` |
| Tiempo | `TRAVELER_OLLAMA_TIMEOUT`: 120 s, máximo 600; límite de pared incluso con respuesta a goteo |
| Entrada generativa | 1.200 caracteres; hasta ocho unidades de evidencia de 600 caracteres |
| Clasificación | Conserva límite de 5.000 caracteres; no recorta entrada para forzar Qwen |
| Salida | 65.536 bytes, 1.024 tokens máximos, contexto 4.096 |
| Muestreo | Temperatura 0, semilla 42; no garantiza identidad bit a bit entre runtimes |
| Razonamiento | `think=false`; thinking separado/bloques completos descartados, bloque sin cerrar rechazado |
| Lote | Hasta cinco consultas secuenciales y presupuesto compartido de 120 s iniciales |

Las demás filas del lote se clasifican y conservan recomendación por reglas.
No se realizan reintentos, sustituciones generativas, descargas ni entrenamiento.
En fallo: `generacion_llm=false`, `informe=null`, código visible y
`alternativa=RECOMENDACION_TRADICIONAL_POR_REGLAS`. No se publica una respuesta
rechazada como informe Qwen ni se cambia el clasificador por un fallo generativo.

| Código | Acción local |
|---|---|
| `OLLAMA_DESCONECTADO` | Iniciar Ollama y comprobar puerto |
| `MODELO_NO_DISPONIBLE` | Comprobar nombre con `ollama list`; un 404 indica recurso no disponible |
| `TIMEOUT` | Comprobar carga/RAM; reintentar conscientemente o ampliar tiempo dentro del límite |
| `CONFIGURACION_INVALIDA` | Corregir variables; no usar servidores externos |
| `ANALISIS_ALTERADO` | Informe rechazado por intentar cambiar sentimiento/aspectos/revisión |
| `HECHOS_NO_SUSTENTADOS` / `PROPUESTA_NO_ADMITIDA` | Informe rechazado; conservar clasificación y revisar relato |
| `ENTRADA_GENERATIVA_EXCESIVA` / `EVIDENCIA_GENERATIVA_EXCESIVA` | Clasificación conservada; generación omitida, sin truncado oculto |
| `LIMITE_LOTE_GENERATIVO` / `PRESUPUESTO_LOTE_AGOTADO` | Clasificación conservada; usar revisión individual si se necesita informe |

Fuentes primarias HTTP: [chat](https://docs.ollama.com/api/chat),
[JSON estructurado](https://docs.ollama.com/capabilities/structured-outputs),
[thinking](https://docs.ollama.com/capabilities/thinking),
[modelos locales](https://docs.ollama.com/api/tags).

## 6. Alcance comprobado

Las pruebas utilizan HTTP Ollama **simulado**, LR/SVM reales de la DEMO y una
interfaz BETO inyectada para comprobar aislamiento y ausencia de fallback. Sesiones,
cola, CSV y descargas Gradio se ejercitan por HTTP real. No acreditan Qwen, pesos
BETO ni voz física en CI. La nueva aceptación Ollama/Qwen + Gradio en Windows
sigue pendiente hasta ejecutar los pasos 2–4. Ver [guion académico](QWEN_DEMOSTRACION_2026-11-17.md).
