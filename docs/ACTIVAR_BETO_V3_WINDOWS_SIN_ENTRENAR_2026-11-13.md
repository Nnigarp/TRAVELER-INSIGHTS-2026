# Activar BETO V3 en Windows sin volver a entrenar

Se recuperó el artefacto ajustado que produjo las respuestas de aceptación del PR28. Su manifiesto SHA256 es `c66453d041090bf6d10a9af10fdfbb8ab2d53b70a18f44e4439921a28914a7b7`, igual al binding publicado antes de aquella medición. Se conserva semilla42, TRAIN300, cabeza80 épocas, ajuste completo1 época y ventana256 tokens. No se reentrena, cuantiza, calibra ni modifica la receta.

## Archivo que debes descargar

Descargar de la entrega de esta conversación **beto-v3-pr28-seed42-overlay-v1.zip**. Es un artefacto externo, no un archivo que deba añadirse a Git.

| Propiedad | Valor |
|---|---|
| Versión | `beto-v3-pr28-seed42-overlay-v1` |
| Tamaño ZIP | 407.210.123 bytes, aproximadamente388MiB |
| SHA256 ZIP | `4b676117b204a84fb23bec8706268d887bdaffe9aa298c7dcac63b8d1eb56769` |
| Encoder ajustado | SHA256 `276348048d5926a4a84a835437db1248531e082fbb8d131ac575dccc970f7ae4` |
| Cabeza supervisada | SHA256 `2d045c47d1128773c919c41a3f06d1fbd22b0187af5ab4c62663c060b864ba5f` |
| Base omitida del ZIP | `pytorch_model.bin`,439.621.341 bytes; SHA256 `e131a95091c777bbd45250fb647fec415010cb8cd1ad6e1d59babeb82a0be360` |

El ZIP contiene la adaptación completa, cabeza, manifiesto y configuración/tokenizador/aviso original. La base omitida se busca en `artifacts/`, incluidas las carpetas creadas por tu intento interrumpido. Su hash se verifica antes de reutilizarla. No basta con descomprimir el ZIP y seleccionar esa carpeta: el instalador completa la base y valida todos los archivos para el cargador ya existente.

## PowerShell

Cerrar con Ctrl+C el Gradio anterior. Abrir PowerShell **en la raíz del repositorio**. Mientras se revisa el PR, usar esta rama; no modificar main directamente:

```powershell
git fetch origin
git switch feat/beto-v3-local-artifact-20261113
git pull --ff-only

$py = if (Test-Path '.\.venv-research\Scripts\python.exe') {
    (Resolve-Path '.\.venv-research\Scripts\python.exe').Path
} elseif (Test-Path '.\.venv\Scripts\python.exe') {
    (Resolve-Path '.\.venv\Scripts\python.exe').Path
} else { (Get-Command python -ErrorAction Stop).Source }
$env:PYTHONPATH = (Resolve-Path '.\src').Path
$env:PYTHONUTF8 = '1'
& $py --version

$zip = Join-Path $env:USERPROFILE 'Downloads\beto-v3-pr28-seed42-overlay-v1.zip'
$expected = '4b676117b204a84fb23bec8706268d887bdaffe9aa298c7dcac63b8d1eb56769'
if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $expected) {
    throw 'El archivo no coincide: no instalar.'
}

& $py -m traveler_agent.beto_distribution install --archive $zip --output artifacts/beto-v3-pr28-local
if ($LASTEXITCODE -ne 0) { throw 'La instalación no terminó: revisar el mensaje antes de continuar.' }

& $py scripts/check_beto_activation.py --artifact artifacts/beto-v3-pr28-local --output artifacts/beto-activation-01.json
if ($LASTEXITCODE -ne 0) { throw 'No se confirmó BETO real; no considerar LR de respaldo como BETO.' }

$env:TRAVELER_BETO_ARTIFACT = (Resolve-Path 'artifacts/beto-v3-pr28-local').Path
& $py app.py
```

Usar el mismo entorno Python3.12 donde ya funcionan LR/SVM y se cargó el encoder. Si guardaste el ZIP en otro lugar o el navegador añadió `(1)`, ajustar `$zip`. No desactivar controles de hash.

El instalador no sobrescribe versiones existentes. Si esa carpeta ya está instalada y válida, **no reinstalar**: ejecutar directamente `check_beto_activation.py` con un nombre nuevo de informe y luego asignar la variable e iniciar Gradio.

### Si falta la caché base

`BASE_CACHE_MISS` no requiere entrenar. Si sabes dónde estaba, añadir `--base-cache 'artifacts/beto-v3-demo/checkpoint'` al comando de instalación. Puede ser otra carpeta con el archivo base correcto. Si fue eliminada, autorizar únicamente la descarga oficial fijada:

```powershell
& $py -m traveler_agent.beto_distribution install --archive $zip --output artifacts/beto-v3-pr28-local --download-base
```

Esto descarga la revisión `c4d86612f51b4f46759c8390d1798c2febe71b93`, verifica su hash y no entrena. La adaptación ajustada siempre procede del ZIP recuperado. La instalación funciona por bloques de1MiB y necesita aproximadamente1GiB libre adicional para la carpeta final y reserva, además de ZIP/caché ya guardados. No crea un optimizador ni mantiene el ZIP entero en RAM.

## Qué debe verse

- `check_beto_activation.py`: `status=PASSED`, `modelo_utilizado=beto`, `fallback=false`, `trace_events=10`, `ui_beto_available=true`, `default_model=lr` y el manifiesto `c66453…`.
- En el nuevo Gradio, LR sigue siendo el valor inicial. Seleccionar **BETO supervisado (experimental sintético)** expresamente. Abrir el JSON: modelo efectivo `beto`, `seleccion_modelo.fallback=false`, manifiesto correcto y diez eventos.
- La opción se ofrece solo si están presentes los archivos completos. Es una comprobación rápida de presencia/tamaño/cabecera; el cargador comprueba los hashes completos al cargar. Un fallo sigue produciendo LR de respaldo explícito; la comprobación de activación lo rechaza.

Si no aparece: revisar ruta dentro de `artifacts/` o `data/private/`, entorno Python correcto, carpeta completa y **reinicio del servidor**. Cambiar una variable en PowerShell no actualiza un Python que ya estaba ejecutándose. Revisar la dirección localhost del servidor nuevo y recargar esa página.

## Qué significó BudgetExceeded

En main, `replay()` limita el trabajo a `min(900, presupuesto registrado)`:900s para preparar representaciones, ajustar la cabeza y ejecutar el encoder. `encoder_batch_finished` es una comprobación de tiempo **después de un minibatch de entrenamiento**. Una llamada síncrona puede terminar después del límite; la ejecución se declara interrumpida y no publica ese ajuste parcial como clasificador válido.

La excepción confirma agotamiento de plazo; por sí sola no distingue CPU lenta, carga concurrente, presión de RAM/paginación u otras causas. El archivo local `replay-interruption.json` registra etapa y tiempo efectivo. No se ha inspeccionado remotamente el hardware ni ese archivo de tu portátil. La memoria y el tiempo de inferencia tampoco se garantizan a partir del hardware del constructor.

Conservar los directorios interrumpidos. Para un diagnóstico limitado a metadatos, sin repetir entrenamiento:

```powershell
& $py scripts/diagnose_beto_local.py --output artifacts/beto-diagnostico-01.json
```

El paquete evita ejecutar esos pasos de entrenamiento en Windows. No se cambian los900s, epochs, filas, semillas ni longitud de secuencia para forzar un éxito.

## Procedencia y límites

El descriptor versionado está en [models/beto-v3-pr28-seed42-overlay-v1.json](models/beto-v3-pr28-seed42-overlay-v1.json). El aviso del checkpoint original se conserva en `checkpoint/README.md`, incluido el caveat de licencia/corpus de origen. Los pesos grandes se entregan fuera de Git.

Esta activación es una comprobación funcional, no otra evaluación predictiva. Los resultados históricos y los36 casos del PR28 permanecen congelados. BETO conserva su carácter experimental sintético, su limitación de revisión humana y el recorte a256 tokens. LR sigue por defecto. No se amplía a5.000 ni se afirma validación humana.
