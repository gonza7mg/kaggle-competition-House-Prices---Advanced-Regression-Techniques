$ErrorActionPreference = "Stop"

$Competition = "house-prices-advanced-regression-techniques"
$DataDir = Join-Path $PSScriptRoot "..\data\raw"
$ZipPath = Join-Path $DataDir "$Competition.zip"

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

Write-Host "Descargando datos de Kaggle..."
kaggle competitions download -c $Competition -p $DataDir

Write-Host "Extrayendo archivos..."
Expand-Archive -Path $ZipPath -DestinationPath $DataDir -Force

Write-Host "Listo. Archivos en $DataDir"
