param(
  [double]$MinSeriesUsed = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (Get-Location).Path
$ledgerPath = Join-Path (Join-Path $root "experiments") "experiment_ledger.csv"
if (-not (Test-Path $ledgerPath)) { throw "Missing ledger: $ledgerPath" }

$rows = Import-Csv $ledgerPath |
  Where-Object { [double]$_.n_series_used -ge $MinSeriesUsed }

if (-not $rows) { throw "No eligible rows in ledger." }

$best = $rows |
  Sort-Object @{Expression={[double]$_.score_eff}; Descending=$true},
              @{Expression={$_.timestamp}; Descending=$true} |
  Select-Object -First 1

# Output just the tag so scripts can capture it
$best.tag
