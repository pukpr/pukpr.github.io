param(
  # Tag name under experiments/, e.g. regtest-20260109 or regtest-fast
  [string]$Tag = ("regtest-" + (Get-Date -Format "yyyyMMdd")),

  # CLI args you normally use
  [int]$Low = 1940,
  [int]$High = 1970,

  # If python is on PATH, leave default. Otherwise pass full path.
  [string]$Python = "python3",

  # lte_mlr.py relative to the per-series directory. From examples/mlr/<series>, ..\lte_mlr.py matches your usage.
  [string]$LteScriptRel = "..\..\..\lte_mlr.py",

  # Max iters for fast regression
  [int]$MaxIters = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure we’re in examples/mlr/
$root = (Get-Location).Path
$expRoot = Join-Path $root "experiments"
$outRoot = Join-Path $expRoot $Tag

New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

# Fast regression knob
$env:max_iters = "$MaxIters"

Write-Host "Regression tag: $Tag"
Write-Host "Output root:    $outRoot"
Write-Host "max_iters:      $env:max_iters"
Write-Host "Args:           --cc --random --low $Low --high $High"
Write-Host ""

# Find series directories: any directory that contains a ts.dat file at its top level,
# excluding experiments/ itself.
$seriesDirs = Get-ChildItem -Directory -Path $root |
  Where-Object { $_.Name -ne "experiments" } |
  Where-Object { Test-Path (Join-Path $_.FullName "ts.dat") }

if (-not $seriesDirs) {
  throw "No series directories found (expected subdirs containing ts.dat). Are you running this from examples/mlr/?"
}

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
    & $Python $LteScriptRel "ts.dat" "--cc" "--random" "--low" "$Low" "--high" "$High"
    if ($LASTEXITCODE -ne 0) {
      throw "lte_mlr.py exited with code $LASTEXITCODE for series $seriesName"
    }
  } finally {
    Pop-Location
  }
}

Write-Host ""
Write-Host "Done. Outputs in: $outRoot"
Write-Host "Next: run .\compare_regtest.ps1 -Tag $Tag"
