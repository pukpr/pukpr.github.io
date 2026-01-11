param(
  [Parameter(Mandatory=$true)]
  [string]$Tag,

  [double]$Lambda = 0.3,

  [string]$ExcludeSeriesRegex = "^QBO30$"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Median([double[]]$xs) {
  if (-not $xs -or $xs.Count -eq 0) { return $null }
  $s = $xs | Sort-Object
  $n = $s.Count
  if ($n % 2 -eq 1) { return [double]$s[($n-1)/2] }
  return ([double]$s[$n/2 - 1] + [double]$s[$n/2]) / 2.0
}

$root = (Get-Location).Path
$outRoot = Join-Path (Join-Path $root "experiments") $Tag
$metricsCsv = Join-Path $outRoot "metrics_tally.csv"
if (-not (Test-Path $metricsCsv)) { throw "Missing $metricsCsv. Run the experiment first." }

$rows = Import-Csv $metricsCsv |
  Where-Object { $_.series -notmatch $ExcludeSeriesRegex } |
  Where-Object { $_.exitcode -eq "0" }

# Pull numeric vectors; skip blanks
$cv = @()
$tr = @()
$gap = @()

foreach ($r in $rows) {
  if ($r.CV -eq "" -or $r.training -eq "") { continue }
  $c = [double]$r.CV
  $t = [double]$r.training
  $cv += $c
  $tr += $t
  $gap += [math]::Max(0.0, ($t - $c))
}

$medCV = Median $cv
$medTr = Median $tr
$medGap = Median $gap

if ($null -eq $medCV) { throw "No usable CV values found in $metricsCsv (after filtering)." }
if ($null -eq $medGap) { $medGap = 0.0 }

$scoreEff = $medCV - ($Lambda * $medGap)

# Write per-experiment score file too (handy)
$scorePath = Join-Path $outRoot "experiment_score.json"
$scoreObj = [pscustomobject]@{
  tag = $Tag
  lambda = $Lambda
  median_CV = $medCV
  median_training = $medTr
  median_gap = $medGap
  score_eff = $scoreEff
  n_series_used = $cv.Count
  timestamp = (Get-Date).ToString("o")
}
($scoreObj | ConvertTo-Json -Depth 5) | Out-File -Encoding utf8 $scorePath

Write-Host "Score for $Tag"
Write-Host ("  median(CV)      = {0:N6}" -f $medCV)
Write-Host ("  median(training)= {0:N6}" -f $medTr)
Write-Host ("  median(gap)     = {0:N6}" -f $medGap)
Write-Host ("  score_eff       = {0:N6}  (lambda={1})" -f $scoreEff, $Lambda)
Write-Host ("  n series used   = {0}" -f $cv.Count)
Write-Host ("Wrote: {0}" -f $scorePath)
