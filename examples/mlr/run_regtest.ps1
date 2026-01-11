<#
run_regtest.ps1
================
Runs a fast regression test for all time-series subdirectories under examples/mlr/,
writing ALL outputs into examples/mlr/experiments/<Tag>/<series>/...

Key features:
- Leaves baseline directories untouched
- Copies ts.dat (+ ts.dat.p if present) into the output directory and runs there
- Captures stdout/stderr into <series>\run.log
- Parses the final "Metrics:" block from output and writes experiments\<Tag>\metrics_tally.csv
- Excludes QBO30 by default
- Avoids interactive plot windows by default (no --plot unless -Plot is passed)
- Sets MPLBACKEND=Agg to discourage GUI windows even if plotting is enabled

Usage (from examples/mlr/):
  $env:max_iters=3
  .\run_regtest.ps1 -Tag regtest-fast -Low 1940 -High 1970

Optional:
  .\run_regtest.ps1 -Tag regtest-plot -Plot
  .\run_regtest.ps1 -ExcludeSeriesRegex "^(QBO30|SOMEOTHER)$"
#>

param(
  # Tag name under experiments/, e.g. regtest-20260109 or regtest-fast
  [string]$Tag = ("regtest-" + (Get-Date -Format "yyyyMMdd-HHmmss")),

  # CV interval bounds (matches your CLI)
  [int]$Low = 1940,
  [int]$High = 1970,

  # If python is on PATH, leave default. Otherwise pass full path.
  [string]$Python = "python",

  # lte_mlr.py relative to the per-series directory.
  # From examples/mlr/experiments/<tag>/<series>, ..\..\..\lte_mlr.py is correct.
  [string]$LteScriptRel = "..\..\..\lte_mlr.py",

  # Max iterations for regression speed; also set via $env:max_iters externally if you prefer.
  [int]$MaxIters = 3,

  # Enable plots (off by default to avoid GUI popups)
  [switch]$Plot,

  # Exclude series by directory name (regex). Default excludes QBO30.
  [string]$ExcludeSeriesRegex = "^QBO30$"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-MetricsBlock([string[]]$lines) {
  # We want the LAST metrics block in case multiple are printed.
  # Expected format:
  # Metrics:
  #   CV = 0.416629
  #   training = 0.672273
  #   MSE = 0.628933
  #   Pearson r = 0.6105317137598516

  $cv = $null; $tr = $null; $mse = $null; $pr = $null
  $inMetrics = $false

  for ($i = 0; $i -lt $lines.Count; $i++) {
    $ln = $lines[$i]

    if ($ln -match "^\s*Metrics:\s*$") {
      # Reset for a new block; keep parsing so the LAST block wins.
      $cv = $null; $tr = $null; $mse = $null; $pr = $null
      $inMetrics = $true
      continue
    }

    if (-not $inMetrics) { continue }

    # If we hit an empty line after metrics started, we can keep going;
    # but don't force stop—some outputs have blank lines mid-block.
    if ($ln -match "^\s*CV\s*=\s*([-+0-9\.eE]+)\s*$") { $cv = [double]$Matches[1]; continue }
    if ($ln -match "^\s*training\s*=\s*([-+0-9\.eE]+)\s*$") { $tr = [double]$Matches[1]; continue }
    if ($ln -match "^\s*MSE\s*=\s*([-+0-9\.eE]+)\s*$") { $mse = [double]$Matches[1]; continue }
    if ($ln -match "^\s*Pearson\s+r\s*=\s*([-+0-9\.eE]+)\s*$") { $pr = [double]$Matches[1]; continue }
  }

  return [pscustomobject]@{
    CV        = $cv
    training  = $tr
    MSE       = $mse
    pearson_r = $pr
  }
}

# Ensure we’re in examples/mlr/
$root = (Get-Location).Path
$expRoot = Join-Path $root "experiments"
$outRoot = Join-Path $expRoot $Tag

New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

# Fast regression knob
$env:max_iters = "$MaxIters"

# Avoid GUI plot windows for matplotlib if used
$env:MPLBACKEND = "Agg"

Write-Host "Regression tag:      $Tag"
Write-Host "Output root:         $outRoot"
Write-Host "max_iters (env):     $env:max_iters"
Write-Host "ExcludeSeriesRegex:  $ExcludeSeriesRegex"
Write-Host ("Args:                --cc --random {0} --low {1} --high {2}" -f ($(if ($Plot) { "--plot" } else { "" }), $Low, $High))
Write-Host ""

# Find series directories: any directory that contains a ts.dat file at its top level,
# excluding experiments/ itself and excluded series.
$seriesDirs = Get-ChildItem -Directory -Path $root |
  Where-Object { $_.Name -ne "experiments" } |
  Where-Object { Test-Path (Join-Path $_.FullName "ts.dat") } |
  Where-Object { $_.Name -notmatch $ExcludeSeriesRegex }

if (-not $seriesDirs) {
  throw "No series directories found (expected subdirs containing ts.dat). Are you running this from examples/mlr/?"
}

# Running tally rows
$metricRows = @()

foreach ($sd in $seriesDirs) {
  $seriesName = $sd.Name
  $srcDir = $sd.FullName
  $dstDir = Join-Path $outRoot $seriesName

  New-Item -ItemType Directory -Force -Path $dstDir | Out-Null

  # Copy input data
  Copy-Item -Force (Join-Path $srcDir "ts.dat") (Join-Path $dstDir "ts.dat")

  # Copy baseline params if present (used as initialization)
  $srcP = Join-Path $srcDir "ts.dat.p"
  if (Test-Path $srcP) {
    Copy-Item -Force $srcP (Join-Path $dstDir "ts.dat.p")
  }

  Push-Location $dstDir
  try {
    Write-Host "=== Running series: $seriesName ==="

    $args = @("ts.dat", "--cc", "--random")
    if ($Plot) { $args += "--plot" }
    $args += @("--low", "$Low", "--high", "$High")

    $logPath = Join-Path $dstDir "run.log"
    $ts = Get-Date

    # Capture stdout+stderr and tee to run.log
    $output = & $Python $LteScriptRel @args 2>&1 | Tee-Object -FilePath $logPath
    $exit = $LASTEXITCODE

    # Parse metrics regardless of exit code (sometimes partial output is useful)
    $m = Parse-MetricsBlock -lines $output

    $metricRows += [pscustomobject]@{
      series     = $seriesName
      CV         = $m.CV
      training   = $m.training
      MSE        = $m.MSE
      pearson_r  = $m.pearson_r
      exitcode   = $exit
      timestamp  = $ts.ToString("o")
      log        = "experiments/$Tag/$seriesName/run.log"
    }

    if ($exit -ne 0) {
      throw "lte_mlr.py exited with code $exit for series $seriesName (see $logPath)"
    }

    # Basic sanity: warn if metrics missing
    if ($null -eq $m.CV -or $null -eq $m.training) {
      Write-Warning "Metrics parsing did not find CV/training for series $seriesName. Check $logPath"
    }

  } finally {
    Pop-Location
  }
}

# Write the tally CSV
$metricsCsv = Join-Path $outRoot "metrics_tally.csv"
$metricRows | Sort-Object series | Export-Csv -NoTypeInformation -Path $metricsCsv

Write-Host ""
Write-Host "Done."
Write-Host "Outputs in:         $outRoot"
Write-Host "Metrics tally CSV:  $metricsCsv"
Write-Host "Next: run your compare_regtest.ps1 against -Tag $Tag"
