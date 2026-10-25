# ATENCION: sustituye datos existentes de volumenes Docker; comprueba tus copias.
param(
    [Parameter(Mandatory=$true)][string]$BackupFolder,
    [switch]$Overwrite
)
$ErrorActionPreference = 'Stop'
if (-not $Overwrite) { throw 'Debes indicar -Overwrite para reemplazar datos.' }
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backupDir = (Resolve-Path $BackupFolder).Path
foreach ($volume in @('nyx_models','nyx_memory')) {
    if (-not (Test-Path (Join-Path $backupDir "$volume.tar.gz"))) {
        throw "Falta $volume.tar.gz (el respaldo no está completo)"
    }
}
Push-Location $project
try {
    docker compose down
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo detener NYX.' }
    foreach ($volume in @('nyx_models','nyx_memory','nyx_voice')) {
        $archive = Join-Path $backupDir "$volume.tar.gz"
        if (-not (Test-Path $archive)) {
            if ($volume -eq 'nyx_voice') {
                Write-Warning 'Respaldo antiguo sin voz; tendras que instalarla de nuevo.'
                continue
            }
            throw "Falta $archive"
        }
        docker volume create $volume | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "No se pudo crear $volume" }
        docker run --rm --mount "type=volume,source=$volume,target=/target" --mount "type=bind,source=$backupDir,target=/backup,readonly" alpine:3.21 sh -c "find /target -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + && cd /target && tar -xzf /backup/$volume.tar.gz"
        if ($LASTEXITCODE -ne 0) { throw "Fallo restaurando $volume" }
    }
    Write-Host 'Restauracion lista. Ejecuta docker compose up -d --build.'
} finally { Pop-Location }
