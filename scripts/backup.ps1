# Copia consistente de modelos y memoria de NYX. Requiere Docker Desktop.
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backupDir = Join-Path $project ('backups\nyx-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Push-Location $project
$stopped = $false
try {
    docker compose stop
    if ($LASTEXITCODE -ne 0) { throw 'No se pudieron detener los servicios.' }
    $stopped = $true
    foreach ($volume in @('nyx_models', 'nyx_memory')) {
        Write-Host "Respaldando $volume..."
        docker run --rm --mount "type=volume,source=$volume,target=/source,readonly" --mount "type=bind,source=$backupDir,target=/backup" alpine:3.21 sh -c "cd /source && tar -czf /backup/$volume.tar.gz ."
        if ($LASTEXITCODE -ne 0) { throw "Fallo al respaldar $volume" }
    }
    Write-Host "Respaldo creado: $backupDir"
    Write-Host 'Copia esta carpeta en almacenamiento seguro externo; NO en GitHub publico.'
} finally {
    if ($stopped) { docker compose start }
    Pop-Location
}
