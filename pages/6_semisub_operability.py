# -*- coding: utf-8 -*-
# 06_Semisub_HeaveOperability.py — North Sea: Heave response & operability from "RMS response per meter Hs" vs Tp
# Input CSV (robust):
#   Row with TP headers: TP1, TP2, ..., TPn   (n≈21)
#   First column = Hull alternative   (A, B, ...)
#   Data = RMS response per meter Hs (m heave per m Hs) for each TP
#   Optional: a row with actual "Tp [s]" centers to enable precise interpolation
#
# Outputs (North Sea zoom only):
#   • Expected heave map [m]
#   • Operability (%) — Wave-only (Hs/Tp limit)
#   • Operability (%) — Heave-only (Hs*f(Tp) <= heave_limit)
#   • Operability (%) — Wave ∩ Heave
#
# Notes:
#  - UTF-8 header included; ASCII-only code (no curly quotes) to avoid copy/paste parse errors
#  - Same plotting style and North Sea extent as your Metocean page
#  - Icons kept minimal to avoid encoding issues

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Semisub: Heave & Operability (North Sea)", layout="wide", page_icon="⚓")
st.title("Semisub — Heave Response & Operability (North Sea)")

# -----------------------------
# Constants
# -----------------------------
ZOOM_EXTENT = [-13, 35, 52, 76]
FEATURE_SCALE = "10m"
BASE_CMAP = "turbo"
CLIP_PCT = 99.6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NS_CANDIDATES = [
    os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc"),
    os.path.join(BASE_DIR, "metocean_scatter_050deg_NS_monthclim.nc"),
]

# -----------------------------
# Helpers (I/O, grids, plotting)
# -----------------------------
@st.cache_resource
def load_nc(paths):
    for p in paths:
        try:
            return xr.open_dataset(p), p
        except Exception:
            pass
    raise FileNotFoundError("Could not find metocean_scatter_050deg_NS_monthclim.nc")

def bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

def unwrap_lon_centers_from_edges(lon_edges):
    lon_c = bin_centers(lon_edges)
    if np.nanmax(lon_edges) > 180:
        lon_c = ((lon_c + 180) % 360) - 180
    return lon_c

def to_sorted_lon_lat(field2d, lat_c, lon_edges):
    if lat_c[0] > lat_c[-1]:
        field2d = field2d[::-1, :]
        lat_c = lat_c[::-1]
    lon_uns = unwrap_lon_centers_from_edges(lon_edges)
    j_sort = np.argsort(lon_uns)
    return field2d[:, j_sort], lat_c, lon_uns[j_sort]

def normalize_pdf(prob):
    tot = prob.sum(dim=("hs_bin","tp_bin"))
    return xr.where(tot > 0, prob/tot, 0)

def pct_ticks(): return np.arange(0, 101, 10)
def pct_levels(): return np.linspace(0, 100, 61)

def hs_levels_zoom(arr):
    vmin = float(np.nanmin(arr)); vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    filled   = np.arange(np.floor(vmin/0.1)*0.1, np.ceil(vmax/0.1)*0.1 + 1e-12, 0.1)
    contours = np.arange(np.floor(vmin/0.2)*0.2, np.ceil(vmax/0.2)*0.2 + 1e-12, 0.2)
    return filled, contours, contours

def plot_zoom(lon, lat, data, title, filled, contours, ticks, cmap=BASE_CMAP, show_grid=True):
    fig = plt.figure(figsize=(14, 6), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = ax.contourf(lon, lat, data, levels=filled, cmap=cmap, extend="both",
                     transform=ccrs.PlateCarree(), zorder=1)
    try:
        cs = ax.contour(lon, lat, data, levels=contours, colors="black", linewidths=0.45,
                        transform=ccrs.PlateCarree(), zorder=2)
        ax.figure.canvas.draw()
        ax.clabel(cs, fontsize=6, inline=True, inline_spacing=1, fmt="%g", rightside_up=True)
    except Exception:
        pass
    ax.add_feature(cfeature.LAND.with_scale(FEATURE_SCALE), facecolor="lightgray", edgecolor="none", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale(FEATURE_SCALE), linewidth=0.7, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale(FEATURE_SCALE), linewidth=0.3, zorder=12)
    ax.set_extent(ZOOM_EXTENT, crs=ccrs.PlateCarree())
    if show_grid:
        Lon2D, Lat2D = np.meshgrid(lon, lat)
        ax.scatter(Lon2D.ravel(), Lat2D.ravel(), s=6, color="gray", alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=3)
    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

# -----------------------------
# Load North Sea dataset
# -----------------------------
try:
    ds, used_path = load_nc(NS_CANDIDATES)
except Exception as e:
    st.error(f"Failed to load NS dataset: {e}")
    st.stop()

for k in ["prob","hs_edges","tp_edges","lat3_edges","lon3_edges"]:
    if k not in ds:
        st.error(f"Dataset missing {k}")
        st.stop()

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

# Hs units
units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c  = bin_centers(hs_edges)    # (hs_bin)
tp_c  = bin_centers(tp_edges)    # (tp_bin)
lat_c = bin_centers(lat_edges)
lon_c = unwrap_lon_centers_from_edges(lon_edges)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Aggregation")
    mode = st.radio("Use", ["By month","Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    month_label = st.selectbox("Month", months, index=0)
    month_idx = dict(zip(months, month_vals))[month_label]

    st.subheader("Wave acceptance (Hs/Tp limit)")
    limit_mode = st.radio("Limit mode", ["Single Hcrit", "Hs limit per Tp (CSV)"], index=1)
    Hcrit = st.number_input("Hs threshold (m)", min_value=0.1, max_value=15.0, value=3.5, step=0.1)
    limit_csv = st.file_uploader("Upload Hs/Tp limit curve CSV", type=["csv"], key="hs_tp_limit_csv")

    st.subheader("Heave per meter Hs CSV")
    heave_csv = st.file_uploader("Upload RMS response per meter Hs (by TP)", type=["csv"], key="heave_per_hs_csv")

    st.subheader("Heave acceptance")
    heave_limit = st.number_input("Heave limit (m)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    st.subheader("Display")
    show_grid = st.checkbox("Show grid points", True)

# -----------------------------
# Select probability field
# -----------------------------
if mode == "By month":
    prob = ds["prob"].sel(month=month_idx)
    title_suffix = f" — {month_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"
prob = normalize_pdf(prob)  # dims: hs_bin, tp_bin, lat3_bin, lon3_bin

HS = xr.DataArray(hs_c, dims=["hs_bin"])

# -----------------------------
# Build Hs_limit(Tp) from CSV or constant
# -----------------------------
def build_hs_limit(tp_centers, mode, Hcrit, csv_file):
    if mode == "Single Hcrit" or (csv_file is None):
        return np.full_like(tp_centers, Hcrit, dtype=float)
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin-1")
    cols = [c.lower().strip() for c in df.columns]
    def col_like(options):
        for o in options:
            if o in cols:
                return df.columns[cols.index(o)]
        return None
    tp_col = col_like(["tp (s)", "tp", "tp_s"])
    hs_col = col_like(["hs_limit (m)", "hs limit (m)", "hs_limit", "hs (m)", "hs"])
    if tp_col is None or hs_col is None:
        st.error("Limit CSV must contain 'Tp (s)' and 'Hs_limit (m)'. Falling back to Hcrit.")
        return np.full_like(tp_centers, Hcrit, dtype=float)
    tp_in = pd.to_numeric(df[tp_col], errors="coerce").astype(float).values
    hs_in = pd.to_numeric(df[hs_col], errors="coerce").astype(float).values
    m = np.isfinite(tp_in) & np.isfinite(hs_in)
    tp_in, hs_in = tp_in[m], hs_in[m]
    if tp_in.size < 2:
        st.error("Limit CSV must provide at least two valid Tp rows. Falling back to Hcrit.")
        return np.full_like(tp_centers, Hcrit, dtype=float)
    order = np.argsort(tp_in)
    tp_in, hs_in = tp_in[order], hs_in[order]
    hs_interp = np.interp(tp_centers, tp_in, hs_in, left=hs_in[0], right=hs_in[-1])
    return hs_interp

Hs_limit_tp = build_hs_limit(tp_c, limit_mode, Hcrit, limit_csv)  # (tp_bin,)
Hs_limit_1D = xr.DataArray(Hs_limit_tp, dims=["tp_bin"])

# -----------------------------
# Parse "RMS response per meter Hs" CSV
# -----------------------------
def parse_heave_per_hs(uploaded_csv):
    """
    Expected layout (robust):
      - One row contains many TPk labels (TP1, TP2, ..., TPn) -> this is the header row.
      - First column under that header contains hull names.
      - Data cells are RMS response per meter Hs for each TP.
      - Optional: a row where first text contains 'Tp' and the following cells are numeric Tp centers [s].
    Returns:
      configs: list[str]
      R_csv:   2D array [config, n_csv]  (heave_per_Hs vs TPk)
      tp_csv:  1D array [n_csv] of Tp centers in seconds if found, else None
    """
    df_raw = pd.read_csv(uploaded_csv, header=None)
    # Find the header row with many TPk tokens
    header_row_idx = None
    header_labels = None
    for i in range(min(6, len(df_raw))):  # search first few rows
        row = df_raw.iloc[i].astype(str).tolist()
        tps = [c.strip() for c in row if re.match(r"^TP\d+$", c.strip(), flags=re.IGNORECASE)]
        if len(tps) >= 3:  # likely header
            header_row_idx = i
            header_labels = tps
            break
    if header_row_idx is None:
        # Fallback: assume second row holds TP1..TPn
        header_row_idx = 1
        row = df_raw.iloc[header_row_idx].astype(str).tolist()
        header_labels = [c.strip() for c in row if re.match(r"^TP\d+$", c.strip(), flags=re.IGNORECASE)]
    if not header_labels:
        st.error("Could not locate 'TP1..TPn' header row in CSV.")
        st.stop()

    # Identify the TP columns by index in that row
    hdr_series = df_raw.iloc[header_row_idx].astype(str)
    tp_col_indices = [j for j, v in enumerate(hdr_series) if re.match(r"^TP\d+$", v.strip(), flags=re.IGNORECASE)]
    n_csv = len(tp_col_indices)
    if n_csv < 3:
        st.error("Found fewer than 3 TP columns; please check the CSV.")
        st.stop()

    # Read data rows below header until a blank line or non-numeric block
    data_rows = []
    for i in range(header_row_idx + 1, len(df_raw)):
        row = df_raw.iloc[i]
        # first column: hull name (string), unless row holds Tp centers
        first_cell = str(row.iloc[0]).strip()
        if first_cell == "" or first_cell.lower().startswith("tp"):
            continue
        vals = []
        ok = True
        for j in tp_col_indices:
            try:
                v = float(row.iloc[j])
            except Exception:
                ok = False
                break
            vals.append(v)
        if ok:
            data_rows.append((first_cell, vals))
    if not data_rows:
        st.error("No hull rows found under TP header in CSV.")
        st.stop()

    configs = [name for name, _ in data_rows]
    R_csv = np.vstack([np.array(vals, dtype=float) for _, vals in data_rows])  # [config, n_csv]

    # Optional: try to find a row with Tp centers [s]
    tp_centers_csv = None
    for i in range(header_row_idx + 1, len(df_raw)):
        first_cell = str(df_raw.iloc[i, 0]).strip().lower()
        if first_cell.startswith("tp"):
            # collect numeric values in the TP columns
            vals = []
            for j in tp_col_indices:
                try:
                    vals.append(float(df_raw.iloc[i, j]))
                except Exception:
                    vals.append(np.nan)
            arr = np.array(vals, dtype=float)
            if np.isfinite(arr).sum() >= 3:
                tp_centers_csv = arr
            break

    return configs, R_csv, tp_centers_csv

if heave_csv is None:
    st.info("Upload the 'RMS response per meter Hs' CSV to proceed.")
    st.stop()

cfg_names, R_heave_per_Hs_csv, tp_centers_csv = parse_heave_per_hs(heave_csv)
n_csv = R_heave_per_Hs_csv.shape[1]

# -----------------------------
# Map CSV TP bins to dataset TP bins
# -----------------------------
def interp_rows(M, x_from, x_to):
    return np.vstack([np.interp(x_to, x_from, r) for r in M])

if len(tp_c) == n_csv:
    R_use = R_heave_per_Hs_csv  # 1:1 mapping
    mapping_note = f"TP mapping: 1:1 ({n_csv} bins)."
elif tp_centers_csv is not None and np.isfinite(tp_centers_csv).sum() >= 3:
    R_use = interp_rows(R_heave_per_Hs_csv, tp_centers_csv, tp_c)
    mapping_note = "TP mapping: interpolated from CSV Tp centers."
else:
    # Ordinal mapping across dataset Tp range
    k = np.arange(1, n_csv+1)
    tp_min, tp_max = float(tp_c.min()), float(tp_c.max())
    x_from = tp_min + ((k - 0.5)/n_csv) * (tp_max - tp_min)
    R_use = interp_rows(R_heave_per_Hs_csv, x_from, tp_c)
    mapping_note = "TP mapping: ordinal across dataset Tp range."

# Pick configuration
cfg = st.selectbox("Hull alternative", cfg_names, index=0)
i_cfg = cfg_names.index(cfg)
fTp = xr.DataArray(R_use[i_cfg], dims=["tp_bin"])  # heave per meter Hs vs Tp (m/m)

# -----------------------------
# Build acceptance masks and expected heave
# -----------------------------
HS2D = xr.DataArray(hs_c, dims=["hs_bin"]).broadcast_like(prob)
TPmask = fTp.broadcast_like(prob)  # (hs,tp,y,x) along tp
M_heave = HS2D * TPmask            # heave per sea state

# Wave acceptance: Hs <= Hs_limit(Tp)
HsLim2D = xr.DataArray(Hs_limit_tp, dims=["tp_bin"]).broadcast_like(prob)
I_wave = (HS2D <= HsLim2D).astype(float)

# Heave acceptance: Hs*f(Tp) <= heave_limit
I_heave = (M_heave <= heave_limit).astype(float)

# Expected heave [m]
E_heave = (prob * M_heave).sum(dim=("hs_bin","tp_bin"))  # (y,x)

# Operabilities [%]
P_wave   = (prob * I_wave).sum(dim=("hs_bin","tp_bin")) * 100.0
P_heave  = (prob * I_heave).sum(dim=("hs_bin","tp_bin")) * 100.0
P_both   = (prob * I_wave * I_heave).sum(dim=("hs_bin","tp_bin")) * 100.0

# ---------------------------------------------
# Metric selector (same UX as MotionOperability)
# ---------------------------------------------
st.sidebar.subheader("Metric to show")
metric = st.sidebar.selectbox(
    "Metric",
    [
        "Expected heave (m)",
        "Operability: heave ≤ limit (%)",
        "Operability: wave ≤ Hs/Tp limit (%)",
        "Operability: ALL limits (%)",
    ],
)

# ---------------------------------------------
# Prepare 2D arrays
# ---------------------------------------------
def prep(field):
    arr = field.transpose("lat3_bin","lon3_bin").values
    return to_sorted_lon_lat(arr, lat_c, lon_edges)

# Compute 2D fields
heave2, latp, lonp = prep(E_heave)
heave_hi = np.nanpercentile(heave2, CLIP_PCT)
heave2 = np.clip(heave2, None, heave_hi)

p_heave2, latp, lonp = prep(P_heave)
p_wave2,  latp, lonp = prep(P_wave)
p_both2,  latp, lonp = prep(P_both)

# ---------------------------------------------
# Render map depending on selected metric
# ---------------------------------------------
if metric == "Expected heave (m)":
    filled, contours, ticks = hs_levels_zoom(heave2)
    plot_zoom(
        lonp, latp, heave2,
        f"Expected heave (m) — {cfg}{title_suffix}",
        filled, contours, ticks,
        cmap=BASE_CMAP, show_grid=show_grid
    )

elif metric == "Operability: heave ≤ limit (%)":
    plot_zoom(
        lonp, latp, p_heave2,
        f"Operability (%) — Heave ≤ {heave_limit:.2f} m {title_suffix}",
        pct_levels(), pct_ticks(), pct_ticks(),
        cmap=BASE_CMAP, show_grid=show_grid
    )

elif metric == "Operability: wave ≤ Hs/Tp limit (%)":
    plot_zoom(
        lonp, latp, p_wave2,
        f"Operability (%) — Wave (Hs/Tp) {title_suffix}",
        pct_levels(), pct_ticks(), pct_ticks(),
        cmap=BASE_CMAP, show_grid=show_grid
    )

elif metric == "Operability: ALL limits (%)":
    plot_zoom(
        lonp, latp, p_both2,
        f"Operability (%) — Wave ∩ Heave — {cfg}{title_suffix}",
        pct_levels(), pct_ticks(), pct_ticks(),
        cmap=BASE_CMAP, show_grid=show_grid
    )


# -----------------------------
# Prepare arrays for plotting
# -----------------------------
def prep_2d(field):
    arr = field.transpose("lat3_bin","lon3_bin").values
    return to_sorted_lon_lat(arr, lat_c, lon_edges)

# Expected heave
heave2, latp, lonp = prep_2d(E_heave)
hi = np.nanpercentile(heave2, CLIP_PCT)
heave2 = np.clip(heave2, None, hi)
filled_h, cont_h, ticks_h = hs_levels_zoom(heave2)
plot_zoom(lonp, latp, heave2, f"Expected heave (m) — {cfg}{title_suffix}", filled_h, cont_h, ticks_h, cmap=BASE_CMAP, show_grid=show_grid)

# Heave-only operability
phev2, latp, lonp = prep_2d(P_heave)
plot_zoom(lonp, latp, phev2, f"Operability (%) — Heave-only (heave <= {heave_limit:.2f} m){title_suffix}",
          pct_levels(), pct_ticks(), pct_ticks(), cmap=BASE_CMAP, show_grid=show_grid)

# Wave-only operability
pwav2, latp, lonp = prep_2d(P_wave)
plot_zoom(lonp, latp, pwav2, f"Operability (%) — Wave-only (Hs/Tp limit){title_suffix}",
          pct_levels(), pct_ticks(), pct_ticks(), cmap=BASE_CMAP, show_grid=show_grid)

# Combined
pboth2, latp, lonp = prep_2d(P_both)
plot_zoom(lonp, latp, pboth2, f"Operability (%) — Wave ∩ Heave — {cfg}{title_suffix}",
          pct_levels(), pct_ticks(), pct_ticks(), cmap=BASE_CMAP, show_grid=show_grid)

# Mapping note
st.caption(mapping_note)
