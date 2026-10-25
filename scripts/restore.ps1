# ATENCION: restaura sobrescribiendo datos existentes de los volumenes Docker.
param(
    [Parameter(Mandatory=$true)][string]$BackupFolder,
    [switch]$Overwrite
)
$ErrorActionPreference = 'Stop'
if (-not $Overwrite) { throw 'Debes indicar -Overwrite para reemplazar los datos.' }
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backupDir = (Resolve-Path $BackupFolder).Path
foreach ($volume in @('nyx_models', 'nyx_memory')) {
    if (-not (Test-Path (Join-Path $backupDir "$volume.tar.gz"))) { throw "Falta $volume.tar.gz" }
}
Push-Location $project
try {
    docker compose down
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo detener NYX.' }
    foreach ($volume in @('nyx_models', 'nyx_memory')) {
        docker volume create $volume | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "No se pudo crear $volume" }
        docker run --rm --mount "type=volume,source=$volume,target=/target" --mount "type=bind,source=$backupDir,target=/backup,readonly" alpine:3.21 sh -c "find /target -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + && cd /target && tar -xzf /backup/$volume.tar.gz"
        if ($LASTEXITCODE -ne 0) { throw "Fallo al restaurar $volume" }
    }
    Write-Host 'Restauracion lista. Ejecuta docker compose up -d --build.'
} finally { Pop-Location }
