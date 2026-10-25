# Instala el modelo neuronal de embeddings LOCAL para NYX Brain v1.
# Uso: .\scripts\enable_brain.ps1
# No publica claves, no toca archivos .env.security y no elimina volúmenes.
param(
    [ValidatePattern('^[A-Za-z0-9_./:-]+$')]
    [string]$Model = "embeddinggemma:300m"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path $Root ".env.brain"
Push-Location $Root
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker no está instalado o no figura en PATH."
    }
    docker compose version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose no está disponible."
    }

    Write-Host "Descargando el modelo de embeddings local: $Model"
    docker compose exec -T ollama ollama pull $Model
    if ($LASTEXITCODE -ne 0) {
        throw "La descarga falló. No se modificó la configuración cognitiva."
    }

    $Existing = @()
    if (Test-Path -LiteralPath $EnvPath) {
        $Existing = @(Get-Content -LiteralPath $EnvPath -Encoding UTF8)
    }
    $Remaining = @($Existing | Where-Object {
        $_ -notmatch '^\s*NYX_BRAIN_EMBED_MODEL\s*='
    })
    $Remaining += "NYX_BRAIN_EMBED_MODEL=$Model"
    $Content = ($Remaining -join [Environment]::NewLine) + [Environment]::NewLine
    [System.IO.File]::WriteAllText($EnvPath, $Content,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "Configuración privada guardada en .env.brain."

    docker compose up -d --force-recreate api
    if ($LASTEXITCODE -ne 0) {
        throw "El contenedor API no pudo recrearse. Comprueba docker compose logs api."
    }
    Write-Host "Modelo configurado. NYX puede indexar registros desde MEMORIA > CEREBRO > RECUPERAR."
}
finally {
    Pop-Location
}
