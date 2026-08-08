$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectDir

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    throw "未找到 .venv，请先运行 scripts\setup.ps1"
}

& ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

