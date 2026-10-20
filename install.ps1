# Windows PowerShell: installs Pal in its own virtual environment, then checks the computer.
#   .\install.ps1          install pal-code from PyPI
#   .\install.ps1 -Dev     install this source folder in editable mode, with the test tools
param([switch]$Dev)
$ErrorActionPreference = "Stop"
if (-not (Test-Path venv)) { python -m venv venv }
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
if ($Dev) { python -m pip install -e ".[dev,full]" } else { python -m pip install pal-code }
pal doctor
Write-Host ""
Write-Host "Run, in the folder you want to build in:  pal cli    (or  pal web  for the browser)"
