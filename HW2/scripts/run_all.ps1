param([switch]$SkipDownload, [string]$Config = 'config/experiment.json')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $python = Join-Path $projectRoot '.venv/Scripts/python.exe'
    $settings = Get-Content -Raw -Encoding UTF8 -LiteralPath $Config | ConvertFrom-Json
    & $python scripts/fetch_official_inputs.py
    if ($LASTEXITCODE -ne 0) { throw 'Official input verification failed.' }
    if (-not $SkipDownload) {
        & $python scripts/download.py --start $settings.raw_start_utc --end $settings.raw_end_utc_exclusive --out data/raw --workers 6 --max-gb 60
        if ($LASTEXITCODE -ne 0) { throw 'Source download failed.' }
    }
    & $python scripts/preprocess.py --raw data/raw --out data/processed
    if ($LASTEXITCODE -ne 0) { throw 'Official preprocessing failed.' }
    & $python run.py --config $Config
    if ($LASTEXITCODE -ne 0) { throw 'Training or evaluation failed.' }
    & $python run.py --config $Config --predict-only
    if ($LASTEXITCODE -ne 0) { throw 'Prediction reproduction failed.' }
    & $python scripts/evaluate.py --pred "output_$($settings.student_id).csv" --input submission/test_input.csv
    if ($LASTEXITCODE -ne 0) { throw 'Submission format check failed.' }
} finally {
    Pop-Location
}
