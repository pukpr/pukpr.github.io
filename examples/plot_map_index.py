#!/usr/bin/env python3
"""
Parse fixed-width station file (IDCodeName.txt) and create a 2-D world map (map.html)
with pins for each station using Plotly.

Requirements:
    pip install pandas plotly

Usage:
    python plot_map.py
Outputs:
    map.html (interactive world map - opens automatically if supported)
"""
import re
import pandas as pd
import plotly.express as px
import webbrowser
from pathlib import Path

DATAFILE = Path("ID.txt")
OUTHTML = Path("map_index.html")

if not DATAFILE.exists():
    raise SystemExit(f"Data file {DATAFILE} not found. Place the provided IDCodeName.txt in the same folder.")

rows = []
pattern = re.compile(r"^\s*(-?\d+)\s+(\d+)\s+(.+?)\s+(-?\d+\.\d+)\s+([NS])\s+(-?\d+\.\d+)\s+([EW])")

with DATAFILE.open("r", encoding="utf-8") as fh:
    for ln in fh:
        if ln.strip() == "" or ln.lstrip().startswith("ID  Code"):
            continue
        m = pattern.match(ln)
        if not m:
            # skip lines that don't match the expected pattern
            continue
        sid = int(m.group(1))
        code = int(m.group(2))
        name = m.group(3).strip()
        label = m.group(3).strip()[:6]
        lat_val = float(m.group(4))
        ns = m.group(5)
        lon_val = float(m.group(6))
        ew = m.group(7)
        lat = lat_val if ns == "N" else -lat_val
        lon = lon_val if ew == "E" else -lon_val
        rows.append({"id": sid, "code": code, "name": name, "latitude": lat, "longitude": lon, "label": label})

if not rows:
    raise SystemExit("No station rows parsed. Check the input file format.")

df = pd.DataFrame(rows)

fig = px.scatter_geo(
    df,
    lat="latitude",
    lon="longitude",
    hover_name="name",
    text="label",
    hover_data=["id", "code"],
    projection="natural earth",
    title="MSL or Climate Index Stations (from ID.txt)"
)

fig.update_traces(
    textposition="top center",  # Options: 'top center', 'bottom right', etc.
    textfont=dict(size=9, color="green", family="arial narrow")
)

fig.update_traces(marker=dict(size=6, color="crimson", opacity=0.8))
fig.update_layout(legend=dict(x=0.85, y=0.95))

fig.write_html(OUTHTML, auto_open=False)
print(f"Wrote {OUTHTML.resolve()}")

# try to open in the default browser
try:
    webbrowser.open(OUTHTML.resolve().as_uri())
except Exception:
    pass