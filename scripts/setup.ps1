$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectDir

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
Write-Output "环境准备完成。"

