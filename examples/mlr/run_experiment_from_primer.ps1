param(
  # Primer tag to seed from; if omitted, auto-select from ledger
  [string]$PrimerTag = "",

  # Number of child experiments to spawn
  [int]$Children = 4,

  # Jitter magnitude (relative for amplitudes, additive for phases). Keep small.
  [double]$JitterAmpRel = 0.03,
  [double]$JitterPhaseAbs = 0.03, # radians-ish if your phases are rad

  # Probability a given parameter gets perturbed (sparsifies changes)
  [double]$PerturbProb = 0.35,

  # CV interval bounds
  [int]$Low = 1940,
  [int]$High = 1970,

  # Optimizer speed knob
  [int]$MaxIters = 50,

  # Penalty weight for score_eff
  [double]$Lambda = 0.3,

  # Exclusions
  [string]$ExcludeSeriesRegex = "^QBO30$",

  # Python + script path (relative from experiments/<tag>/<series>)
  [string]$Python = "python3",
  [string]$LteScriptRel = "..\..\..\lte_mlr.py",

  # Turn on plot (default off); also sets MPLBACKEND=Agg
  [switch]$Plot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-Tag() {
  return (Get-Date -Format "yyyyMMdd-HHmmss")
}

function Parse-MetricsBlock([string[]]$lines) {
  $cv = $null; $tr = $null; $mse = $null; $pr = $null
  $inMetrics = $false
  foreach ($ln in $lines) {
    if ($ln -match "^\s*Metrics:\s*$") { $cv=$null; $tr=$null; $mse=$null; $pr=$null; $inMetrics=$true; continue }
    if (-not $inMetrics) { continue }
    if ($ln -match "^\s*CV\s*=\s*([-+0-9\.eE]+)\s*$") { $cv = [double]$Matches[1]; continue }
    if ($ln -match "^\s*training\s*=\s*([-+0-9\.eE]+)\s*$") { $tr = [double]$Matches[1]; continue }
    if ($ln -match "^\s*MSE\s*=\s*([-+0-9\.eE]+)\s*$") { $mse = [double]$Matches[1]; continue }
    if ($ln -match "^\s*Pearson\s+r\s*=\s*([-+0-9\.eE]+)\s*$") { $pr = [double]$Matches[1]; continue }
  }
  return [pscustomobject]@{ CV=$cv; training=$tr; MSE=$mse; pearson_r=$pr }
}

function Jitter-ParamsJsonFile([string]$path, [double]$ampRel, [double]$phaseAbs, [double]$p) {
  if (-not (Test-Path $path)) { return }

  $obj = Get-Content -Raw $path | ConvertFrom-Json

  # Heuristic: parameters might be at top-level, or under .params
  $paramNode = $obj
  if ($obj.PSObject.Properties.Name -contains "params") {
    $paramNode = $obj.params
  }

  # Targets by name patterns (customize as you learn the schema)
  $ampLike   = "Amp|Amplitude|Imp_|Hold|Damp"
  $phaseLike = "Phase"

  foreach ($prop in $paramNode.PSObject.Properties) {
    $name = $prop.Name
    $val = $prop.Value

    # Only jitter numeric scalars (skip arrays/objects)
    if ($val -isnot [double] -and $val -isnot [float] -and $val -isnot [int]) { continue }
    if ((Get-Random) -gt $p) { continue }

    $x = [double]$val

    if ($name -match $phaseLike) {
      $delta = (Get-Random -Minimum (-$phaseAbs) -Maximum $phaseAbs)
      $paramNode.$name = $x + $delta
    } elseif ($name -match $ampLike) {
      # multiplicative noise; keep sign
      $r = (Get-Random -Minimum (-$ampRel) -Maximum $ampRel)
      $paramNode.$name = $x * (1.0 + $r)
    } else {
      # small generic nudge
      $r = (Get-Random -Minimum (-$ampRel) -Maximum $ampRel)
      $paramNode.$name = $x * (1.0 + 0.5*$r)
    }
  }

  ($obj | ConvertTo-Json -Depth 50) | Out-File -Encoding utf8 $path
}

# --- Locate paths
$root = (Get-Location).Path
$expRoot = Join-Path $root "experiments"
New-Item -ItemType Directory -Force $expRoot | Out-Null

# Choose primer
if ($PrimerTag -eq "") {
  $PrimerTag = & .\select_primer.ps1
  Write-Host ("Auto-selected primer: {0}" -f $PrimerTag)
}

$primerRoot = Join-Path $expRoot $PrimerTag
if (-not (Test-Path $primerRoot)) { throw "Primer experiment directory not found: $primerRoot" }

# Discover baseline series dirs (source of ts.dat)
$baselineSeries = Get-ChildItem -Directory -Path $root |
  Where-Object { $_.Name -ne "experiments" } |
  Where-Object { Test-Path (Join-Path $_.FullName "ts.dat") } |
  Where-Object { $_.Name -notmatch $ExcludeSeriesRegex }

if (-not $baselineSeries) { throw "No baseline series dirs found. Run from examples/mlr/." }

# Env knobs
$env:max_iters = "$MaxIters"
$env:MPLBACKEND = "Agg"

Write-Host ""
Write-Host "Primer:        $PrimerTag"
Write-Host "Children:      $Children"
Write-Host "max_iters:     $env:max_iters"
Write-Host "Jitter:        AmpRel=$JitterAmpRel PhaseAbs=$JitterPhaseAbs Prob=$PerturbProb"
Write-Host "CV window:     low=$Low high=$High"
Write-Host "Exclude:       $ExcludeSeriesRegex"
Write-Host ""

# Run children
for ($c = 1; $c -le $Children; $c++) {
  $tag = New-Tag
  $childRoot = Join-Path $expRoot $tag
  New-Item -ItemType Directory -Force $childRoot | Out-Null

  Write-Host "=== Child $c/$Children : $tag ==="

  $metricRows = @()

  foreach ($sd in $baselineSeries) {
    $series = $sd.Name
    $srcDir = $sd.FullName
    $dstDir = Join-Path $childRoot $series
    New-Item -ItemType Directory -Force $dstDir | Out-Null

    # Copy baseline data
    Copy-Item -Force (Join-Path $srcDir "ts.dat") (Join-Path $dstDir "ts.dat")

    # Seed params from primer if present, else baseline
    $pPrimer = Join-Path (Join-Path $primerRoot $series) "ts.dat.p"
    $pBase   = Join-Path $srcDir "ts.dat.p"
    if (Test-Path $pPrimer) {
      Copy-Item -Force $pPrimer (Join-Path $dstDir "ts.dat.p")
    } elseif (Test-Path $pBase) {
      Copy-Item -Force $pBase (Join-Path $dstDir "ts.dat.p")
    }

    # Jitter seed params (to diversify children)
    $pOut = Join-Path $dstDir "ts.dat.p"
    if (Test-Path $pOut) {
      Jitter-ParamsJsonFile -path $pOut -ampRel $JitterAmpRel -phaseAbs $JitterPhaseAbs -p $PerturbProb
    }

    # Run lte_mlr.py in dstDir
    Push-Location $dstDir
    try {
      $args = @("ts.dat", "--cc", "--random")
      if ($Plot) { $args += "--plot" }
      $args += @("--low", "$Low", "--high", "$High")

      $logPath = Join-Path $dstDir "run.log"
      $ts = Get-Date
      # $output = & $Python $LteScriptRel @args 2>&1 | Tee-Object -FilePath $logPath
      $output = & $Python $LteScriptRel @args |& Tee-Object -FilePath $logPath
      $exit = $LASTEXITCODE

      $m = Parse-MetricsBlock -lines $output

      $metricRows += [pscustomobject]@{
        series     = $series
        CV         = $m.CV
        training   = $m.training
        MSE        = $m.MSE
        pearson_r  = $m.pearson_r
        exitcode   = $exit
        timestamp  = $ts.ToString("o")
        log        = "experiments/$tag/$series/run.log"
      }

      if ($exit -ne 0) {
        Write-Warning "Series $series failed (exit=$exit). See $logPath"
      }
    } finally {
      Pop-Location
    }
  }

  # Write metrics tally
  $metricsCsv = Join-Path $childRoot "metrics_tally.csv"
  $metricRows | Sort-Object series | Export-Csv -NoTypeInformation $metricsCsv
  Write-Host ("Wrote: {0}" -f $metricsCsv)

  # Score + update ledger
  & .\score_experiment.ps1 -Tag $tag -Lambda $Lambda -ExcludeSeriesRegex $ExcludeSeriesRegex | Out-Host
  & .\update_ledger.ps1 -Tag $tag | Out-Host

  Start-Sleep -Seconds 1  # ensures unique timestamps for tags even on fast machines
}

Write-Host ""
Write-Host "Done generating children. Primer selection for next round:"
$best = & .\select_primer.ps1
Write-Host ("  Best current primer = {0}" -f $best)
