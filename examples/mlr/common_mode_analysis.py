#!/usr/bin/env python3
"""
Summarize common-mode MLR model parameters and station excursions.

This script focuses on the numeric station subdirectories under examples/mlr
and aggregates the model parameter files (ts.dat.p) to identify:
  - common-mode parameters (mean/median/std across stations)
  - station excursions relative to the common-mode baseline
  - top excursions to highlight the most divergent stations
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


NUMERIC_DIR = re.compile(r"^\d+$")
ID_LINE_PATTERN = re.compile(
    r"^\s*(?P<id>-?\d+)\s+(?P<code>\d+)\s+(?P<name>.+?)\s+"
    r"(?P<lat>-?\d+\.\d+)\s+(?P<ns>[NS])\s+(?P<lon>-?\d+\.\d+)\s+"
    r"(?P<ew>[EW])\s+(?P<start>\d+)\s+(?P<stop>\d+)\s+"
    r"(?P<quality>\d+\.\d+)\s+(?P<I>\d+)\s+(?P<country>.+?)\s*$"
)


@dataclass(frozen=True)
class StationInfo:
    station_id: str
    name: str
    country: str
    latitude: float
    longitude: float


def parse_station_metadata(path: Path) -> Dict[str, StationInfo]:
    if not path.exists():
        raise FileNotFoundError(f"Station metadata file not found: {path}")
    stations: Dict[str, StationInfo] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("ID  Code"):
                continue
            match = ID_LINE_PATTERN.match(line)
            if not match:
                continue
            sid = match.group("id")
            name = match.group("name").strip()
            country = match.group("country").strip()
            lat = float(match.group("lat"))
            if match.group("ns") == "S":
                lat = -lat
            lon = float(match.group("lon"))
            if match.group("ew") == "W":
                lon = -lon
            stations[sid] = StationInfo(
                station_id=sid,
                name=name,
                country=country,
                latitude=lat,
                longitude=lon,
            )
    return stations


def numeric_station_dirs(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and NUMERIC_DIR.match(p.name)],
        key=lambda p: int(p.name),
    )


def load_params(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def stdev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return statistics.median(values)


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


def build_common_mode(params: List[Dict[str, object]]) -> Dict[str, object]:
    scalar_keys = [
        "DC",
        "Damp",
        "DeltaTime",
        "Hold",
        "Imp_Amp",
        "Imp_Amp2",
        "Imp_Stride",
        "Initial",
        "LTE_Amp",
        "LTE_Freq",
        "LTE_Phase",
        "LTE_Zero",
        "StartTime",
        "Year",
    ]
    vector_keys = [
        "Aliased",
        "AliasedAmp",
        "AliasedPhase",
        "Harmonics",
        "Periods",
        "PeriodsAmp",
        "PeriodsPhase",
    ]

    common: Dict[str, object] = {}
    for key in scalar_keys:
        vals = [safe_float(p.get(key)) for p in params]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        common[key] = {
            "mean": mean(vals),
            "median": median(vals),
            "std": stdev(vals),
            "min": min(vals),
            "max": max(vals),
        }

    for key in vector_keys:
        arrays = [to_list(p.get(key)) for p in params]
        if not arrays:
            continue
        length = max((len(arr) for arr in arrays), default=0)
        series: List[Dict[str, float]] = []
        for idx in range(length):
            vals = [arr[idx] for arr in arrays if len(arr) > idx]
            series.append(
                {
                    "index": idx,
                    "mean": mean(vals),
                    "median": median(vals),
                    "std": stdev(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
            )
        common[key] = series
    return common


def extract_station_excursions(
    station_id: str,
    params: Dict[str, object],
    common: Dict[str, object],
    station_info: Optional[StationInfo],
) -> List[Dict[str, object]]:
    excursions: List[Dict[str, object]] = []

    def add_excursion(param: str, index: Optional[int], value: float, baseline: float, spread: float) -> None:
        if math.isnan(value) or math.isnan(baseline):
            return
        delta = value - baseline
        zscore = delta / spread if spread and spread > 0 else float("nan")
        excursions.append(
            {
                "station_id": station_id,
                "name": station_info.name if station_info else "",
                "country": station_info.country if station_info else "",
                "parameter": param,
                "index": index if index is not None else "",
                "value": value,
                "baseline": baseline,
                "delta": delta,
                "zscore": zscore,
            }
        )

    for key, stats in common.items():
        if isinstance(stats, list):
            values = to_list(params.get(key))
            for entry in stats:
                idx = entry["index"]
                if idx >= len(values):
                    continue
                add_excursion(key, idx, values[idx], entry["mean"], entry["std"])
        elif isinstance(stats, dict):
            value = safe_float(params.get(key))
            if value is None:
                continue
            add_excursion(key, None, value, stats["mean"], stats["std"])

    return excursions


def write_common_mode_periods(common: Dict[str, object], out_path: Path) -> None:
    periods = common.get("Periods")
    if not isinstance(periods, list):
        return
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "period", "mean_amp", "std_amp", "mean_phase", "std_phase"])
        amp_stats = common.get("PeriodsAmp")
        phase_stats = common.get("PeriodsPhase")
        for row in periods:
            idx = row["index"]
            period = row["mean"]
            amp_mean = amp_stats[idx]["mean"] if isinstance(amp_stats, list) and idx < len(amp_stats) else ""
            amp_std = amp_stats[idx]["std"] if isinstance(amp_stats, list) and idx < len(amp_stats) else ""
            phase_mean = phase_stats[idx]["mean"] if isinstance(phase_stats, list) and idx < len(phase_stats) else ""
            phase_std = phase_stats[idx]["std"] if isinstance(phase_stats, list) and idx < len(phase_stats) else ""
            writer.writerow([idx, period, amp_mean, amp_std, phase_mean, phase_std])


def write_common_mode_params(common: Dict[str, object], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "index", "mean", "median", "std", "min", "max"])
        for key, stats in sorted(common.items()):
            if isinstance(stats, list):
                for row in stats:
                    writer.writerow(
                        [
                            key,
                            row["index"],
                            row["mean"],
                            row["median"],
                            row["std"],
                            row["min"],
                            row["max"],
                        ]
                    )
            elif isinstance(stats, dict):
                writer.writerow(
                    [
                        key,
                        "",
                        stats["mean"],
                        stats["median"],
                        stats["std"],
                        stats["min"],
                        stats["max"],
                    ]
                )


def write_station_excursions(
    excursions: List[Dict[str, object]],
    out_path: Path,
) -> None:
    fieldnames = [
        "station_id",
        "name",
        "country",
        "parameter",
        "index",
        "value",
        "baseline",
        "delta",
        "zscore",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in excursions:
            writer.writerow(row)


def write_top_excursions(
    excursions: List[Dict[str, object]],
    out_path: Path,
    top_n: int,
) -> None:
    def sort_key(row: Dict[str, object]) -> float:
        zscore = row.get("zscore")
        if isinstance(zscore, (int, float)):
            if math.isnan(zscore):
                return 0.0
            return abs(zscore)
        return 0.0

    excursions_sorted = sorted(excursions, key=sort_key, reverse=True)
    top_rows = excursions_sorted[:top_n]
    fieldnames = [
        "station_id",
        "name",
        "country",
        "parameter",
        "index",
        "value",
        "baseline",
        "delta",
        "zscore",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in top_rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize common-mode MLR parameters and station excursions.",
    )
    parser.add_argument(
        "--params-root",
        type=Path,
        default=Path("examples/mlr"),
        help="Root directory containing numeric station subdirectories.",
    )
    parser.add_argument(
        "--station-metadata",
        type=Path,
        default=Path("examples/IDCodeName.txt"),
        help="Station metadata file (IDCodeName.txt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/common_mode_mlr"),
        help="Output directory for summary CSV files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top excursions to report.",
    )
    args = parser.parse_args()

    params_root = args.params_root
    station_metadata = parse_station_metadata(args.station_metadata)
    stations = numeric_station_dirs(params_root)
    if not stations:
        raise SystemExit(f"No numeric station directories found under {params_root}")

    params_list: List[Dict[str, object]] = []
    station_params: Dict[str, Dict[str, object]] = {}
    for station_dir in stations:
        param_path = station_dir / "ts.dat.p"
        if not param_path.exists():
            continue
        params = load_params(param_path)
        params_list.append(params)
        station_params[station_dir.name] = params

    if not params_list:
        raise SystemExit("No parameter files found.")

    common = build_common_mode(params_list)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_common_mode_periods(common, out_dir / "common_mode_periods.csv")
    write_common_mode_params(common, out_dir / "common_mode_params.csv")

    excursions: List[Dict[str, object]] = []
    for station_id, params in station_params.items():
        excursions.extend(
            extract_station_excursions(
                station_id,
                params,
                common,
                station_metadata.get(station_id),
            )
        )

    write_station_excursions(excursions, out_dir / "station_excursions.csv")
    write_top_excursions(excursions, out_dir / "top_excursions.csv", args.top_n)

    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
