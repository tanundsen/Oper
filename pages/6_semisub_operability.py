# 06_Semisub_HeaveOperability.py — North Sea: Heave Response & Operability for Semisub Drilling
# ----------------------------------------------------------------------------------------------
# Supports two CSV inputs:
#   A) Per-location significant heave map:
#         lat, lon, config, heave_sig_m
#   B) Heave RAO(Tp) per configuration:
#         First row: any headers
#         Second row: "TP1, TP2, ..., TPn" (or similar labels — order will be sorted by trailing number)
#         Subsequent rows: <ConfigName>, heave, <values for TP1..TPn>
#
# Hs/Tp limit curve:
#   - "Single Hcrit" (constant Hs threshold), or
#   - "Hs limit per Tp (CSV)" with columns: 'Tp (s)', 'Hs_limit (m)'
#
# Outputs (North Sea zoom only):
#   • Heave response map:
#       - If RAO(Tp) CSV uploaded: expected heave = sum_{hs,tp} PDF * HS * RAO_heave(tp)
#       - Else: uploaded per-location heave map (interpolated to grid by nearest cell)
#   • Operability (%):
#       - Wave acceptance: 1[ Hs <= Hs_limit(Tp) ]
#       - Heave acceptance: 1[ HS*RAO_heave(tp) <= heave_limit ]  (only when RAO is provided)
#       - Intersection (wave ∩ heave) when RAO is provided
#
# Styling matches your app:
#   - PlateCarree projection, North Sea extent
#   - Filled every 0.1 m, contour lines every 0.2 m, inline_spacing=1
#   - Colorbar adapts to zoom region; percent metrics fixed 0–100 %
#   - Optional grid points overlay; optional POIs
#
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import patheffects

# ---------------------------------
# Page setup (icon)
# ---------------------------------
st.set_page_config(page_title="🛢️ Semisub Heave & Operability (NS)", layout="wide", page_icon="🛢️")
st.title("🛢️ Semisub — Heave Response & Operability (North Sea)")

# ---------------------------------
# Fixed (North Sea)
# ---------------------------------
ZOOM_EXTENT = [-13, 35, 52, 76]
FEATURE_SCALE = "10m"         # detailed coastlines for zoom
BASE_CMAP = "turbo"
LEVELS_GENERIC = 50
CLIP_PCT = 99.6               # robust capping for non-% fields

# ---------------------------------
# Data paths (regional NS 0.5°)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc"),
    os.path.join(BASE_DIR, "metocean_scatter_050deg_NS_monthclim.nc"),
]

# ---------------------------------
# Helpers
# ---------------------------------
@st.cache_resource
def load_nc(paths):
    for p in paths:
        try:
            return xr.open_dataset(p), p
        except Exception:
            continue
    raise FileNotFoundError("Could not find 'metocean_scatter_050deg_NS_monthclim.nc' in expected locations.")

def bin_centers(edges):
    return 0.5*(edges[:-1] + edges[1:])

def unwrap_lon_centers_from_edges(lon_edges):
    lon_c = bin_centers(lon_edges)
    if np.nanmax(lon_edges) > 180:
        lon_c = ((lon_c + 180) % 360) - 180
    return lon_c

def to_sorted_lon_lat(field2d, lat_c, lon_edges):
    # flip latitude if descending
    if lat_c[0] > lat_c[-1]:
        field2d = field2d[::-1, :]
        lat_c = lat_c[::-1]
    lon_unsorted = unwrap_lon_centers_from_edges(lon_edges)
    j_sort = np.argsort(lon_unsorted)
    lon_sorted = lon_unsorted[j_sort]
    field2d_sorted = field2d[:, j_sort]
    return field2d_sorted, lat_c, lon_sorted

def normalize_pdf(prob):
    tot = prob.sum(dim=("hs_bin","tp_bin"))
    return xr.where(tot > 0, prob/tot, 0)

def pct_ticks(): return np.arange(0, 101, 10)
def pct_levels(): return np.linspace(0, 100, 61)

def hs_levels_zoom(arr):
    # filled every 0.1 m; contour every 0.2 m
    vmin = float(np.nanmin(arr)); vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    filled   = np.arange(np.floor(vmin/0.1)*0.1, np.ceil(vmax/0.1)*0.1 + 1e-12, 0.1)
    contours = np.arange(np.floor(vmin/0.2)*0.2, np.ceil(vmax/0.2)*0.2 + 1e-12, 0.2)
    return filled, contours, contours

def plot_zoom(lon, lat, data, title, filled, contours, ticks, cmap=BASE_CMAP, show_grid=True, pois=None):
    fig = plt.figure(figsize=(14, 6), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())

    cf = ax.contourf(
        lon, lat, data,
        levels=filled,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1
    )
    try:
        cs = ax.contour(
            lon, lat, data,
            levels=contours,
            colors="black",
            linewidths=0.45,
            transform=ccrs.PlateCarree(),
            zorder=2
        )
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

    if pois:
        halo = [patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
        for p in pois:
            ax.scatter([p["lon"]], [p["lat"]], s=28, c="black", marker="o", transform=ccrs.PlateCarree(), zorder=20)
            ax.text(p["lon"]+0.12, p["lat"]+0.10, f'{p["nr"]} {p["name"]}',
                    transform=ccrs.PlateCarree(), fontsize=7, color="black", zorder=21, path_effects=halo)

    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

# ---------------------------------
# Load metocean dataset (regional)
# ---------------------------------
try:
    ds, used_path = load_nc(CANDIDATES)
except Exception as e:
    st.error(f"Failed to load NS dataset: {e}")
    st.stop()

for k in ["prob","hs_edges","tp_edges","lat3_edges","lon3_edges"]:
    if k not in ds:
        st.error(f"Dataset missing '{k}'")
        st.stop()

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

# Unit fix (Hs in m)
units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c  = bin_centers(hs_edges)     # (hs_bin)
tp_c  = bin_centers(tp_edges)     # (tp_bin)
lat_c = bin_centers(lat_edges)
lon_c = unwrap_lon_centers_from_edges(lon_edges)

# ---------------------------------
# Sidebar controls
# ---------------------------------
with st.sidebar:
    st.subheader("Aggregation")
    mode = st.radio("Use", ["By month", "Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1, 13)
    month_label = st.selectbox("Month", months, index=0)
    month_idx = dict(zip(months, month_vals))[month_label]

    st.subheader("Wave acceptance (Hs/Tp limit)")
    limit_mode = st.radio("Limit mode", ["Single Hcrit", "Hs limit per Tp (CSV)"], index=1)
    Hcrit = st.number_input("Hs threshold (m)", min_value=0.1, max_value=15.0, value=3.5, step=0.1)
    limit_csv = None
    if limit_mode == "Hs limit per Tp (CSV)":
        limit_csv = st.file_uploader("Upload Hs/Tp limit curve CSV", type=["csv"], key="hs_tp_limit_csv")

    st.subheader("Heave input (choose one)")
    csv_type = st.selectbox("CSV contains", ["Per-location heave map", "Heave RAO(Tp) per configuration"], index=0)
    heave_csv = st.file_uploader("Upload heave CSV", type=["csv"], key="heave_csv")

    st.subheader("Heave acceptance")
    heave_limit = st.number_input("Heave limit (m)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    st.subheader("Display")
    show_grid = st.checkbox("Show metocean grid points", True)
    show_pois = st.checkbox("Show POIs", False)

# NS fields (optional)
POIS = [
    {"name": "Ekofisk", "nr": 1, "lat": 56.5333, "lon": 3.2000},
    {"name": "Ula", "nr": 2, "lat": 57.1000, "lon": 2.8333},
    {"name": "Sleipner", "nr": 3, "lat": 58.3667, "lon": 1.9000},
    {"name": "Alvheim", "nr": 4, "lat": 59.5667, "lon": 1.9667},
    {"name": "Oseberg", "nr": 5, "lat": 60.5000, "lon": 2.8333},
    {"name": "Knarr", "nr": 6, "lat": 61.8833, "lon": 3.8333},
    {"name": "Ormen Lange", "nr": 7, "lat": 63.2500, "lon": 5.0000},
    {"name": "Skarv", "nr": 8, "lat": 65.7500, "lon": 7.6667},
    {"name": "Aasta Hansteen", "nr": 9, "lat": 67.0000, "lon": 8.0000},
    {"name": "Johan Castberg", "nr": 10, "lat": 72.0000, "lon": 22.5000},
] if show_pois else None

# ---------------------------------
# Select & normalize PDF
# ---------------------------------
if mode == "By month":
    prob = ds["prob"].sel(month=month_idx)
    title_suffix = f"— {month_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = "— Annual"
prob = normalize_pdf(prob)  # dims: hs_bin, tp_bin, lat3_bin, lon3_bin

# ---------------------------------
# Hs/Tp limit curve (build a function Hs_limit(Tp))
# ---------------------------------
def build_hs_limit(tp_centers, mode, Hcrit, csv_file):
    if mode == "Single Hcrit" or (csv_file is None):
        return np.full_like(tp_centers, fill_value=Hcrit, dtype=float)
    # CSV with columns 'Tp (s)' and 'Hs_limit (m)' (robust name detection)
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
        return np.full_like(tp_centers, fill_value=Hcrit, dtype=float)
    tp_in = pd.to_numeric(df[tp_col], errors="coerce").astype(float).values
    hs_in = pd.to_numeric(df[hs_col], errors="coerce").astype(float).values
    m = np.isfinite(tp_in) & np.isfinite(hs_in)
    tp_in, hs_in = tp_in[m], hs_in[m]
    if tp_in.size < 2:
        st.error("Limit CSV must provide at least two valid Tp rows. Falling back to Hcrit.")
        return np.full_like(tp_centers, fill_value=Hcrit, dtype=float)
    order = np.argsort(tp_in)
    tp_in, hs_in = tp_in[order], hs_in[order]
    hs_interp = np.interp(tp_centers, tp_in, hs_in, left=hs_in[0], right=hs_in[-1])
    return hs_interp

Hs_limit_tp = build_hs_limit(tp_c, limit_mode, Hcrit, limit_csv)  # (tp_bin,)

# ---------------------------------
# Heave input: parse either (A) map or (B) RAO(Tp)
# ---------------------------------
def parse_heave_map(uploaded_csv):
    """
    Expected columns (case-insensitive): 'lat', 'lon', 'config', 'heave_sig_m'
    Returns:
      configs: list[str]
      heave_maps: dict[config] -> 2D array [lat,lon] populated at nearest cells (NaN for missing)
    """
    try:
        df = pd.read_csv(uploaded_csv)
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_csv, encoding="latin-1")
    cols = {c.lower().strip(): c for c in df.columns}
    needed = ["lat","lon","config","heave_sig_m"]
    for n in needed:
        if n not in cols:
            st.error(f"Heave map CSV must contain columns: {needed}")
            st.stop()

    la = pd.to_numeric(df[cols["lat"]], errors="coerce").astype(float).values
    lo = pd.to_numeric(df[cols["lon"]], errors="coerce").astype(float).values
    cfg = df[cols["config"]].astype(str).str.strip().values
    hv  = pd.to_numeric(df[cols["heave_sig_m"]], errors="coerce").astype(float).values
    ok = np.isfinite(la) & np.isfinite(lo) & np.isfinite(hv)
    la, lo, cfg, hv = la[ok], lo[ok], cfg[ok], hv[ok]

    # nearest grid indices
    # (vectorized by broadcasting: find nearest lat and lon centers)
    i_lat = np.abs(la[:, None] - lat_c[None, :]).argmin(axis=1)
    i_lon = np.abs(lo[:, None] - lon_c[None, :]).argmin(axis=1)

    # collect by config → 2D map
    configs = sorted(np.unique(cfg))
    heave_maps = {}
    ny, nx = len(lat_c), len(lon_c)
    for c in configs:
        arr = np.full((ny, nx), np.nan, dtype=float)
        sel = (cfg == c)
        # If duplicates land in same cell, average
        # Build accumulator
        counts = {}
        sums = {}
        for ii, jj, v in zip(i_lat[sel], i_lon[sel], hv[sel]):
            key = (ii, jj)
            sums[key] = sums.get(key, 0.0) + v
            counts[key] = counts.get(key, 0) + 1
        for (ii, jj), s in sums.items():
            arr[ii, jj] = s / counts[(ii, jj)]
        heave_maps[c] = arr
    return configs, heave_maps

def parse_heave_rao(uploaded_csv):
    """
    CSV with TP1..TPn columns for heave RAO per config.
    Returns:
      configs: list[str], RAO_heave: 2D array [config, tp_bin] aligned to dataset tp_c
    """
    df_raw = pd.read_csv(uploaded_csv, header=None)
    header_tp = df_raw.iloc[1, 2:].tolist()

    def tpkey(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else 0

    tp_labels = sorted(header_tp, key=tpkey)

    # Normalize to long → wide
    records = []
    for i in range(2, len(df_raw)):
        row = df_raw.iloc[i]
        alt = row.iloc[0]
        var = row.iloc[1]
        vals = row.iloc[2:2+len(tp_labels)].tolist()
        if pd.isna(alt) and pd.isna(var):
            continue
        rec = {"Config": str(alt) if not pd.isna(alt) else None,
               "Var":    str(var) if not pd.isna(var) else None}
        for k, v in zip(tp_labels, vals):
            rec[k] = v
        records.append(rec)

    df = pd.DataFrame(records)
    df["Config"] = df["Config"].ffill().astype(str).str.strip()
    df["Var"]    = df["Var"].astype(str).str.strip()

    heave_df = df[df["Var"].str.contains("heave", case=False)]
    if heave_df.empty:
        st.error("RAO CSV: could not find a row with Var containing 'heave'.")
        st.stop()

    configs = heave_df["Config"].drop_duplicates().tolist()

    def mat_from(df_alt):
        mats = []
        for cfg in configs:
            row = df_alt[df_alt["Config"] == cfg][tp_labels]
            mats.append(row.iloc[0].astype(float).to_numpy())
        return np.vstack(mats)  # [config, n_csv]

    R_heave_csv = mat_from(heave_df)         # [config, n_csv]
    n_csv = R_heave_csv.shape[1]

    # Align CSV TP to dataset tp_c (either 1:1, ordinal mapping, or provided Tp centers row)
    tp_centers_csv = None
    tp_df = df[df["Var"].str.contains("tp", case=False)]
    if not tp_df.empty:
        try:
            arr = tp_df.iloc[0][tp_labels].astype(float).to_numpy()
            tp_centers_csv = arr
        except Exception:
            tp_centers_csv = None

    def interp_rows(M, x_from, x_to):
        return np.vstack([np.interp(x_to, x_from, r) for r in M])

    if len(tp_c) == n_csv:
        R_heave = R_heave_csv
        mapping_note = f"TP mapping: 1:1 ({n_csv} bins)."
    elif tp_centers_csv is not None:
        R_heave = interp_rows(R_heave_csv, tp_centers_csv, tp_c)
        mapping_note = "TP mapping: interpolated from CSV Tp centers."
    else:
        # ordinal mapping across dataset Tp range
        k = np.arange(1, n_csv+1)
        tp_min, tp_max = float(tp_c.min()), float(tp_c.max())
        x_from = tp_min + ((k - 0.5)/n_csv) * (tp_max - tp_min)
        R_heave = interp_rows(R_heave_csv, x_from, tp_c)
        mapping_note = "TP mapping: ordinal."
    return configs, R_heave, mapping_note  # [config, tp_bin]

# ---------------------------------
# Build heave field(s) and operability
# ---------------------------------
# Probability field on (hs, tp, y, x)
prob_hstp = prob  # rename for clarity
HS = xr.DataArray(hs_c, dims=["hs_bin"])
TP = xr.DataArray(tp_c, dims=["tp_bin"])

# Wave-limit mask: 1 if Hs <= Hs_limit(Tp)
Hs_limit_1D = xr.DataArray(Hs_limit_tp, dims=["tp_bin"])
HS2D = HS.broadcast_like(prob_hstp)
HsLim2D = Hs_limit_1D.broadcast_like(prob_hstp)
I_wave = (HS2D <= HsLim2D).astype(float)

# Heave data route A or B
heave_map_by_cfg = None
heave_RAO_by_cfg = None
cfg_names = None
mapping_note = ""

if heave_csv is None:
    st.info("Upload a heave CSV to proceed.")
    st.stop()

if csv_type == "Per-location heave map":
    cfg_names, heave_map_by_cfg = parse_heave_map(heave_csv)
else:
    cfg_names, heave_RAO_by_cfg, mapping_note = parse_heave_rao(heave_csv)

if not cfg_names:
    st.error("No configurations detected in the uploaded CSV.")
    st.stop()

cfg = st.selectbox("Configuration", cfg_names, index=0)

# ---------------------------------
# Compute maps
# ---------------------------------
# lon/lat sorted arrays for plotting
ny, nx = len(lat_c), len(lon_c)

if csv_type == "Heave RAO(Tp) per configuration":
    # Expected heave (map): E[HS * RAO_heave(Tp)]
    i_cfg = cfg_names.index(cfg)
    RAO_heave = xr.DataArray(heave_RAO_by_cfg[i_cfg], dims=["tp_bin"])  # [tp_bin]
    M_heave = HS * RAO_heave  # [hs_bin, tp_bin]

    E_heave = (prob_hstp * M_heave).sum(dim=("hs_bin","tp_bin"))  # [lat3_bin, lon3_bin]
    heave_map = E_heave.transpose("lat3_bin", "lon3_bin").values

    # Two-condition operability: wave ∩ heave
    I_heave = (M_heave <= heave_limit).astype(float)
    P_two = (prob_hstp * I_wave * I_heave).sum(dim=("hs_bin","tp_bin")) * 100.0
    oper_two = P_two.transpose("lat3_bin", "lon3_bin").values

    # Wave-only operability (for comparison)
    P_wave = (prob_hstp * I_wave).sum(dim=("hs_bin","tp_bin")) * 100.0
    oper_wave = P_wave.transpose("lat3_bin","lon3_bin").values

    heave_title = f"Expected heave (m) — {cfg} {title_suffix}"
    oper_two_title = f"Operability (%) — Wave ∩ Heave — {cfg} {title_suffix}"
    oper_wave_title = f"Operability (%) — Wave-only (Hs/Tp limit) {title_suffix}"

    # Plot: heave map
    heave2, latp, lonp = to_sorted_lon_lat(heave_map, lat_c, lon_edges)
    # Robust cap for shading
    hi = np.nanpercentile(heave2, CLIP_PCT)
    heave2 = np.clip(heave2, None, hi)
    filled, contours, ticks = hs_levels_zoom(heave2)
    plot_zoom(lonp, latp, heave2, heave_title, filled, contours, ticks, cmap=BASE_CMAP, show_grid=show_grid, pois=POIS)

    # Plot: Wave ∩ Heave operability
    op2, latp, lonp = to_sorted_lon_lat(oper_two, lat_c, lon_edges)
    plot_zoom(lonp, latp, op2, oper_two_title, pct_levels(), pct_ticks(), pct_ticks(), cmap=BASE_CMAP, show_grid=show_grid, pois=POIS)

    # Plot: Wave-only operability
    opw, latp, lonp = to_sorted_lon_lat(oper_wave, lat_c, lon_edges)
    plot_zoom(lonp, latp, opw, oper_wave_title, pct_levels(), pct_ticks(), pct_ticks(), cmap=BASE_CMAP, show_grid=show_grid, pois=POIS)

    if mapping_note:
        st.caption(mapping_note)

else:
    # Per-location significant heave map (already a map per location)
    heave_map = heave_map_by_cfg[cfg]  # [lat, lon]
    # Plot heave map
    heave2, latp, lonp = to_sorted_lon_lat(heave_map, lat_c, lon_edges)
    hi = np.nanpercentile(heave2, CLIP_PCT)
    heave2 = np.clip(heave2, None, hi)
    filled, contours, ticks = hs_levels_zoom(heave2)
    heave_title = f"Significant heave (m) — {cfg}"
    plot_zoom(lonp, latp, heave2, heave_title, filled, contours, ticks, cmap=BASE_CMAP, show_grid=show_grid, pois=POIS)

    # Wave-only operability (no RAO to evaluate per-sea-state heave)
    P_wave = (prob_hstp * I_wave).sum(dim=("hs_bin","tp_bin")) * 100.0
    oper_wave = P_wave.transpose("lat3_bin","lon3_bin").values
    opw, latp, lonp = to_sorted_lon_lat(oper_wave, lat_c, lon_edges)
    oper_wave_title = f"Operability (%) — Wave-only (Hs/Tp limit) {title_suffix}"
    plot_zoom(lonp, latp, opw, oper_wave_title, pct_levels(), pct_ticks(), pct_ticks(), cmap=BASE_CMAP, show_grid=show_grid, pois=POIS)

    st.info("To compute **two‑condition operability (wave ∩ heave)** per grid cell, upload a **Heave RAO(Tp)** CSV.")

# ---------------------------------
# CSV format tips (expanders)
# ---------------------------------
with st.expander("CSV format — Per-location heave map", expanded=False):
    st.markdown(
        """
**Columns (case-insensitive):**