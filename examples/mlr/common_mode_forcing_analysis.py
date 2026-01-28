#!/usr/bin/env python3
"""
Generate a common-mode forcing time-series graphic from fitted_out.csv files,
grouping stations by similarities in their ts.dat.p parameters.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NUMERIC_DIR = re.compile(r"^\d+$")
BASE_FEATURE_NAMES = [
    "Imp_Stride",
    "Hold",
    "DeltaTime",
    "LTE_Freq",
    "Damp",
    "Imp_Amp",
    "Imp_Amp2",
]
DERIVED_FEATURE_NAMES = [
    "Periods_mean",
    "Periods_std",
    "PeriodsAmp_mean",
    "PeriodsAmp_std",
]
FEATURE_NAMES = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES


def numeric_station_dirs(root: Path, include_nonnumeric: bool) -> List[Path]:
    dirs = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not include_nonnumeric and not NUMERIC_DIR.match(entry.name):
            continue
        if (entry / "fitted_out.csv").exists() and (entry / "ts.dat.p").exists():
            dirs.append(entry)
    if include_nonnumeric:
        return sorted(dirs, key=lambda p: p.name)
    return sorted(dirs, key=lambda p: int(p.name))


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


def load_params(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_forcing_series(path: Path, low: Optional[float], high: Optional[float]) -> pd.Series:
    df = pd.read_csv(path)
    if "forcing" not in df.columns or "time" not in df.columns:
        raise ValueError(f"Expected time and forcing columns in {path}; found {list(df.columns)}")
    series = pd.Series(
        pd.to_numeric(df["forcing"], errors="coerce").to_numpy(),
        index=pd.to_numeric(df["time"], errors="coerce").to_numpy(),
    )
    series = series.dropna()
    if low is not None:
        series = series[series.index >= low]
    if high is not None:
        series = series[series.index <= high]
    mean = float(series.mean()) if not series.empty else 0.0
    std = float(series.std()) if not series.empty else 0.0
    if std > 0.0:
        series = (series - mean) / std
    else:
        series = series - mean
    return series


def extract_features(params: Dict[str, object]) -> np.ndarray:
    features: List[float] = []
    for key in BASE_FEATURE_NAMES:
        val = safe_float(params.get(key))
        features.append(val if val is not None else float("nan"))
    periods = to_list(params.get("Periods"))
    amps = to_list(params.get("PeriodsAmp"))
    features.extend(
        [
            float(np.nanmean(periods)) if periods else float("nan"),
            float(np.nanstd(periods)) if periods else float("nan"),
            float(np.nanmean(amps)) if amps else float("nan"),
            float(np.nanstd(amps)) if amps else float("nan"),
        ]
    )
    return np.array(features, dtype=float)


def standardize_features(features: np.ndarray) -> np.ndarray:
    filled = features.copy()
    means = np.nanmean(filled, axis=0)
    nan_mask = np.isnan(filled)
    if nan_mask.any():
        filled[nan_mask] = np.take(means, np.where(nan_mask)[1])
    means = np.mean(filled, axis=0)
    stds = np.std(filled, axis=0)
    stds[stds == 0.0] = 1.0
    scaled = (filled - means) / stds
    return scaled


def kmeans_grouping(features: np.ndarray, groups: int, seed: int, max_iters: int) -> np.ndarray:
    if features.shape[0] == 0:
        return np.array([], dtype=int)
    groups = min(groups, features.shape[0])
    rng = np.random.default_rng(seed)
    centers = features[rng.choice(features.shape[0], size=groups, replace=False)]
    labels = np.full(features.shape[0], -1, dtype=int)
    for _ in range(max_iters):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        had_empty = False
        for idx in range(groups):
            members = features[new_labels == idx]
            if len(members) > 0:
                new_centers[idx] = members.mean(axis=0)
            else:
                had_empty = True
                if features.shape[0] == 1:
                    new_centers[idx] = features[0]
                else:
                    new_centers[idx] = features[rng.integers(0, features.shape[0])]
        if np.array_equal(new_labels, labels) and not had_empty:
            break
        labels = new_labels
        centers = new_centers
    return labels


def compute_group_means(
    series_by_station: Dict[str, pd.Series],
    labels: np.ndarray,
    station_ids: List[str],
) -> Dict[int, pd.Series]:
    grouped: Dict[int, List[pd.Series]] = {}
    for station_id, label in zip(station_ids, labels):
        grouped.setdefault(int(label), []).append(series_by_station[station_id])
    means: Dict[int, pd.Series] = {}
    for label, series_list in grouped.items():
        df = pd.concat(series_list, axis=1)
        means[label] = df.mean(axis=1, skipna=True)
    return means


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot common-mode forcing time series grouped by parameter similarity.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("examples/mlr"),
        help="Root directory containing station subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/common_mode_forcing"),
        help="Output directory for plots and CSV summaries.",
    )
    parser.add_argument("--groups", type=int, default=4, help="Number of parameter groups.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for grouping.")
    parser.add_argument("--max-iters", type=int, default=50, help="Max iterations for grouping.")
    parser.add_argument("--low", type=float, default=None, help="Lower time bound.")
    parser.add_argument("--high", type=float, default=None, help="Upper time bound.")
    parser.add_argument(
        "--include-nonnumeric",
        action="store_true",
        help="Include non-numeric station directories (e.g., nino34).",
    )
    args = parser.parse_args()

    stations = numeric_station_dirs(args.data_root, args.include_nonnumeric)
    if not stations:
        raise SystemExit(f"No station directories found under {args.data_root}.")

    features: List[np.ndarray] = []
    station_ids: List[str] = []
    series_by_station: Dict[str, pd.Series] = {}
    skipped: List[str] = []
    for station_dir in stations:
        params_path = station_dir / "ts.dat.p"
        fitted_path = station_dir / "fitted_out.csv"
        params = load_params(params_path)
        try:
            series = load_forcing_series(fitted_path, args.low, args.high)
        except ValueError as exc:
            skipped.append(f"{station_dir.name}: {exc}")
            continue
        if series.empty:
            skipped.append(f"{station_dir.name}: forcing series empty after time filtering")
            continue
        station_ids.append(station_dir.name)
        features.append(extract_features(params))
        series_by_station[station_dir.name] = series

    if not station_ids:
        raise SystemExit("No stations had usable forcing series.")
    if skipped:
        print("Skipped stations:")
        for entry in skipped:
            print(f"  {entry}")

    feature_matrix = np.vstack(features)
    scaled_features = standardize_features(feature_matrix)
    labels = kmeans_grouping(scaled_features, args.groups, args.seed, args.max_iters)

    group_means = compute_group_means(series_by_station, labels, station_ids)
    overall_mean = pd.concat(series_by_station.values(), axis=1).mean(axis=1, skipna=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment_rows = []
    for station_id, label, feature_row in zip(station_ids, labels, feature_matrix):
        row = {"station_id": station_id, "group": int(label)}
        row.update({name: value for name, value in zip(FEATURE_NAMES, feature_row)})
        assignment_rows.append(row)
    pd.DataFrame(assignment_rows).to_csv(out_dir / "forcing_group_assignments.csv", index=False)

    combined = pd.DataFrame({"overall_mean": overall_mean})
    for label, series in sorted(group_means.items()):
        combined[f"group_{label}_mean"] = series
    combined.sort_index().to_csv(out_dir / "common_mode_forcing_timeseries.csv", index_label="time")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(overall_mean.index, overall_mean.values, color="black", linewidth=2.0, label="Overall mean")
    palette = plt.cm.tab10.colors
    unique_labels, counts = np.unique(labels, return_counts=True)
    group_counts = dict(zip(unique_labels, counts))
    for idx, (label, series) in enumerate(sorted(group_means.items())):
        color = palette[idx % len(palette)]
        count = group_counts.get(label, 0)
        ax.plot(
            series.index,
            series.values,
            color=color,
            linewidth=1.2,
            label=f"Group {label} (n={count})",
            alpha=0.9,
        )
    ax.set_title("Common-mode forcing grouped by parameter similarity")
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized forcing (z-score)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "common_mode_forcing.png", dpi=150)

    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
