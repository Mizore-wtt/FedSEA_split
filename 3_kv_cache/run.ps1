$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment is missing. See docs/setup.md."
}
& $Python -X utf8 (Join-Path $PSScriptRoot "run_cached.py") @args
exit $LASTEXITCODE
