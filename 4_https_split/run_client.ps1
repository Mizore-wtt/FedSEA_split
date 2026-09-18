$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Project Python environment is missing." }
& $Python -X utf8 (Join-Path $PSScriptRoot "client.py") @args
exit $LASTEXITCODE
