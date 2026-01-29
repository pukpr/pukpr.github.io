#!/usr/bin/env python3
"""
Fit station-specific g() using a shared/common forcing computed analytically.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from lte_mlr import build_mask, model_step_algorithm, normalize_rms_y, read_two_column_csv
from sinusoidal_regression import fit_sinusoidal_regression


NUMERIC_DIR = re.compile(r"^\d+$")


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_list(values: object) -> List[float]:
    if not isinstance(values, list):
        return []
    floats: List[float] = []
    for entry in values:
        val = safe_float(entry)
        if val is not None:
            floats.append(val)
    return floats


def numeric_station_dirs(root: Path, include_nonnumeric: bool) -> List[Path]:
    if (root / "ts.dat").exists() and (root / "ts.dat.p").exists():
        return [root]
    dirs: List[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not include_nonnumeric and not NUMERIC_DIR.match(entry.name):
            continue
        if (entry / "ts.dat").exists() and (entry / "ts.dat.p").exists():
            dirs.append(entry)
    if include_nonnumeric:
        return sorted(dirs, key=lambda p: p.name)
    return sorted(dirs, key=lambda p: int(p.name))


def load_params(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def build_common_params(param_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_keys = [
        "Initial",
        "Year",
        "Imp_Stride",
        "Imp_Amp",
        "Hold",
        "Delta_Y",
        "Imp_Amp2",
        "Damp",
        "DeltaTime",
        "StartTime",
        "DC",
        "Scale",
        "LTE_Freq",
    ]
    list_keys = [
        "Periods",
        "PeriodsAmp",
        "PeriodsPhase",
        "Aliased",
        "AliasedAmp",
        "AliasedPhase",
    ]
    common: Dict[str, Any] = {}
    for key in scalar_keys:
        values = [safe_float(params.get(key)) for params in param_list]
        values = [val for val in values if val is not None]
        if not values:
            continue
        if key == "Imp_Stride":
            common[key] = int(round(mean(values)))
        else:
            common[key] = mean(values)

    for key in list_keys:
        arrays = [to_list(params.get(key)) for params in param_list]
        max_len = max((len(arr) for arr in arrays), default=0)
        if max_len == 0:
            continue
        merged: List[float] = []
        for idx in range(max_len):
            vals = [arr[idx] for arr in arrays if len(arr) > idx]
            merged.append(mean(vals))
        common[key] = merged
    return common


def apply_common_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "DeltaTime": 0.0,
        "Initial": 0.04294,
        "Year": 365.242,
        "Imp_Stride": 12,
        "Imp_Amp": 1.0,
        "Periods": [1.0],
        "PeriodsAmp": [1.0],
        "PeriodsPhase": [0.0],
        "Aliased": [1.0],
        "AliasedAmp": [1.0],
        "AliasedPhase": [0.0],
        "Hold": 0.5,
        "Delta_Y": 0.0,
        "Imp_Amp2": 0.0,
        "Damp": 0.0,
        "StartTime": 1800.0,
    }
    merged = dict(defaults)
    merged.update(params)
    return merged


def compute_common_forcing(time: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = apply_common_defaults(params)
    delta_time = float(params.get("DeltaTime", 0.0))
    initial = float(params.get("Initial", 0.04294))
    year = float(params.get("Year", 365.242))
    imp_stride = int(params.get("Imp_Stride", 12))
    imp_amp = float(params.get("Imp_Amp", 1.0))
    periods = list(params.get("Periods", [1.0]))
    periods_amp = list(params.get("PeriodsAmp", [1.0] * len(periods)))
    periods_phase = list(params.get("PeriodsPhase", [0.0] * len(periods)))
    aliased = list(params.get("Aliased", [1.0]))
    aliased_amp = list(params.get("AliasedAmp", [1.0] * len(aliased)))
    aliased_phase = list(params.get("AliasedPhase", [0.0] * len(aliased)))
    hold = float(params.get("Hold", 0.5))
    delta_y = float(params.get("Delta_Y", 0.0))
    imp_amp2 = float(params.get("Imp_Amp2", 0.0))
    damp = float(params.get("Damp", 0.0))
    start_time = float(params.get("StartTime", 1800.0))

    n = time.size
    forcing = np.zeros_like(time, dtype=float)
    alias = np.zeros_like(time, dtype=float)
    state: Dict[str, Any] = {}
    v = 0.0
    sup = 0.0
    last_time = start_time + delta_time
    for i in range(n):
        t_i = float(time[i] + delta_time)
        n_steps = int(round((t_i - last_time) * 12))
        j = 1
        while j <= n_steps:
            time_i = float(t_i - (n_steps - j) / 12.0)
            v, state, sup = model_step_algorithm(
                i,
                time_i,
                float(time[i]),
                state,
                initial,
                year,
                imp_stride,
                imp_amp,
                periods,
                periods_amp,
                periods_phase,
                aliased,
                aliased_amp,
                aliased_phase,
                hold,
                delta_y,
                imp_amp2,
                damp,
            )
            j += 1
        forcing[i] = float(v)
        alias[i] = float(sup)
        last_time = t_i
    total = forcing + alias
    return total, forcing, alias


def parse_shift_candidates(value: Optional[str]) -> List[float]:
    if not value:
        return [0.0]
    parts = re.split(r"[\s,]+", value.strip())
    shifts = [float(part) for part in parts if part]
    return shifts or [0.0]


def shift_forcing(time: np.ndarray, forcing: np.ndarray, shift: float) -> np.ndarray:
    if shift == 0.0:
        return forcing
    shifted = np.interp(time - shift, time, forcing, left=np.nan, right=np.nan)
    finite = np.isfinite(shifted)
    if finite.any():
        shifted[~finite] = 0.0
    return shifted


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):
        value = np.corrcoef(x, y)[0, 1]
    return float(value)


def dtw_distance(series_a: np.ndarray, series_b: np.ndarray, window: Optional[int]) -> float:
    if series_a.size == 0 or series_b.size == 0:
        return float("nan")
    n = series_a.size
    m = series_b.size
    if window is None and n * m > 5_000_000:
        print(f"Warning: DTW cost matrix size {n}x{m} may be large; consider --dtw-window.")
    w = window if window is not None and window > 0 else max(n, m)
    w = max(w, abs(n - m))
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = abs(series_a[i - 1] - series_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / (n + m))


def fit_station(
    time: np.ndarray,
    observed: np.ndarray,
    forcing: np.ndarray,
    params: Dict[str, Any],
    low: Optional[float],
    high: Optional[float],
    dtw_weight: float,
    dtw_window: Optional[int],
    shift_candidates: List[float],
) -> Dict[str, Any]:
    mask = build_mask(time, low, high, False)
    harmonics_value = params.get("Harmonics")
    harmonics = harmonics_value if harmonics_value is not None else [1]
    harmonics = [int(h) for h in harmonics]
    lte_freq = float(params.get("LTE_Freq", 1.0))

    best: Dict[str, Any] = {"score": -math.inf}
    for shift in shift_candidates:
        shifted = shift_forcing(time, forcing, shift)
        valid = mask & np.isfinite(shifted) & np.isfinite(observed)
        if np.count_nonzero(valid) < 2:
            continue
        x_fit = shifted[valid]
        y_fit = observed[valid]
        fit = fit_sinusoidal_regression(x_fit, y_fit, N_list=harmonics, k=lte_freq, intercept=True, add_linear_x=True)
        model_full = fit["predict"](shifted)
        residuals = model_full - observed
        pearson = pearson_r(model_full[valid], observed[valid])
        dtw = 0.0
        if dtw_weight > 0.0:
            dtw = dtw_distance(model_full[valid], observed[valid], dtw_window)
        if math.isnan(pearson) or math.isnan(dtw):
            score = -math.inf
        else:
            score = pearson - dtw_weight * dtw
        if score > best["score"]:
            best = {
                "shift": shift,
                "forcing": shifted,
                "fit": fit,
                "model": model_full,
                "residuals": residuals,
                "pearson_r": pearson,
                "dtw_distance": dtw,
                "score": score,
                "mask": valid,
            }
    return best


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit station-specific g() using a shared common-mode forcing series.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("examples/mlr"),
        help="Root directory containing station subdirectories.",
    )
    parser.add_argument(
        "--common-params",
        type=Path,
        default=Path("examples/mlr/ts.dat.pp"),
        help="JSON file with common forcing parameters (fallback to station mean if missing).",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=None,
        help="Lower time bound for fitting (inclusive).",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=None,
        help="Upper time bound for fitting (inclusive).",
    )
    parser.add_argument(
        "--dtw-weight",
        type=float,
        default=0.0,
        help="Weight for DTW distance when selecting phase shift (start around 0.2; typical 0.1-0.5 for RMS-normalized data).",
    )
    parser.add_argument(
        "--dtw-window",
        type=int,
        default=None,
        help="Window size for DTW (in samples).",
    )
    parser.add_argument(
        "--shift-candidates",
        type=str,
        default="0.0",
        help="Comma/space-separated forcing phase shifts to test.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/common_mode_forcing_fit"),
        help="Output directory for per-station fits.",
    )
    parser.add_argument(
        "--include-nonnumeric",
        action="store_true",
        help="Include non-numeric station directories (e.g., nino34).",
    )
    args = parser.parse_args()

    stations = numeric_station_dirs(args.data_root, args.include_nonnumeric)
    if not stations:
        raise SystemExit(f"No station directories found under {args.data_root}.")

    params_list: List[Dict[str, Any]] = []
    station_params: Dict[str, Dict[str, Any]] = {}
    for station_dir in stations:
        params_path = station_dir / "ts.dat.p"
        if not params_path.exists():
            continue
        params = load_params(params_path)
        station_params[station_dir.name] = params
        params_list.append(params)

    if not params_list:
        raise SystemExit("No station parameter files found to build common forcing.")

    common_params_source = "station_mean"
    if args.common_params.exists():
        common_params = load_params(args.common_params)
        common_params_source = str(args.common_params)
    else:
        common_params = build_common_params(params_list)

    shift_candidates = parse_shift_candidates(args.shift_candidates)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for station_dir in stations:
        station_id = station_dir.name
        ts_path = station_dir / "ts.dat"
        params = station_params.get(station_id)
        if params is None or not ts_path.exists():
            continue
        time, observed = read_two_column_csv(str(ts_path))
        if time.size == 0:
            continue
        observed = normalize_rms_y(observed)
        forcing_total, _, _ = compute_common_forcing(time, common_params)
        result = fit_station(
            time,
            observed,
            forcing_total,
            params,
            args.low,
            args.high,
            args.dtw_weight,
            args.dtw_window,
            shift_candidates,
        )
        if not result or "fit" not in result:
            continue

        model = result["model"]
        residuals = result["residuals"]
        forcing = result["forcing"]

        out_csv = out_dir / f"{station_id}_common_forcing_fit.csv"
        data = np.column_stack([time, observed, model, forcing, residuals])
        header = "time,observed,model,forcing,residual"
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="")

        fit = result["fit"]
        summary = {
            "station_id": station_id,
            "shift": result.get("shift"),
            "metrics": {
                "pearson_r": result.get("pearson_r"),
                "dtw_distance": result.get("dtw_distance"),
                "score": result.get("score"),
                "mse": float(np.mean(residuals[result["mask"]] ** 2)) if np.any(result["mask"]) else float("nan"),
                "r2": fit.get("R2"),
            },
            "fit": {
                "intercept": fit.get("intercept"),
                "coef_x": fit.get("coef_x"),
                "coefs_by_N": fit.get("coefs_by_N"),
                "harmonics": params.get("Harmonics"),
                "lte_freq": params.get("LTE_Freq"),
            },
            "common_params_source": common_params_source,
            "dtw": {
                "weight": args.dtw_weight,
                "window": args.dtw_window,
            },
        }
        out_json = out_dir / f"{station_id}_common_forcing_fit.json"
        with out_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=to_serializable)

        summary_rows.append(
            {
                "station_id": station_id,
                "shift": result.get("shift"),
                "pearson_r": result.get("pearson_r"),
                "dtw_distance": result.get("dtw_distance"),
                "score": result.get("score"),
                "mse": summary["metrics"]["mse"],
                "r2": summary["metrics"]["r2"],
                "output_csv": str(out_csv),
                "output_json": str(out_json),
            }
        )

    summary_csv = out_dir / "common_mode_forcing_fit_summary.csv"
    if summary_rows:
        fieldnames = [
            "station_id",
            "shift",
            "pearson_r",
            "dtw_distance",
            "score",
            "mse",
            "r2",
            "output_csv",
            "output_json",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
