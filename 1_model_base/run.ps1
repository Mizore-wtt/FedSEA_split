$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment is missing. See the root README.md."
}
& $Python -X utf8 (Join-Path $PSScriptRoot "baseline.py") @args
exit $LASTEXITCODE
