param(
  [Parameter(Mandatory=$true)]
  [string]$Tag
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (Get-Location).Path
$expRoot = Join-Path $root "experiments"
$outRoot = Join-Path $expRoot $Tag
$scorePath = Join-Path $outRoot "experiment_score.json"
if (-not (Test-Path $scorePath)) { throw "Missing $scorePath. Run score_experiment.ps1 first." }

$ledgerPath = Join-Path $expRoot "experiment_ledger.csv"
$score = Get-Content -Raw $scorePath | ConvertFrom-Json

$row = [pscustomobject]@{
  tag = $score.tag
  timestamp = $score.timestamp
  lambda = $score.lambda
  median_CV = $score.median_CV
  median_training = $score.median_training
  median_gap = $score.median_gap
  score_eff = $score.score_eff
  n_series_used = $score.n_series_used
}

if (Test-Path $ledgerPath) {
  $existing = Import-Csv $ledgerPath
  $filtered = $existing | Where-Object { $_.tag -ne $Tag }
  $all = @($filtered) + @($row)
  $all | Sort-Object timestamp | Export-Csv -NoTypeInformation $ledgerPath
} else {
  @($row) | Export-Csv -NoTypeInformation $ledgerPath
}

Write-Host ("Updated ledger: {0}" -f $ledgerPath)
