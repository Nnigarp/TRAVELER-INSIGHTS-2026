[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python = '.\.venv-research\Scripts\python.exe'
    if (!(Test-Path -LiteralPath $python)) { throw 'Falta el intérprete del entorno .venv-research.' }
    & $python -c "import sys, gradio; print('Python', sys.version.split()[0], '/ Gradio', gradio.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'Gradio no está disponible en .venv-research.' }
    $artifact = (Resolve-Path 'artifacts/synthetic-tourism-v4.2/beto-v4-partial256').Path
    $expected = '24be5eb9ecf2ac20c1085b5a93aba09c53feb2ab57c5e290f56534faef817a63'
    if ((Get-FileHash (Join-Path $artifact 'model_manifest.json') -Algorithm SHA256).Hash -ne $expected) {
        throw 'El manifiesto del artefacto BETO difiere del validado para la sustentación.'
    }
    $env:TRAVELER_BETO_ARTIFACT = $artifact
    $env:GRADIO_ANALYTICS_ENABLED = 'False'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRAVELER_ASR_ALLOW_DOWNLOAD = '0'
    try {
        $models = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3
        if ('qwen3:1.7b' -notin $models.models.name) { Write-Warning 'Qwen3 1.7B no figura en Ollama local.' }
    } catch { Write-Warning 'Ollama local está desconectado. El español reconocido sigue disponible.' }
    & $python app.py
    if ($LASTEXITCODE -ne 0) { throw 'El proceso de Gradio terminó con error.' }
} finally { Pop-Location }

