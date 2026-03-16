# -*- coding: utf-8 -*-
# 06_Semisub_HeaveOperability.py — North Sea: Heave response & operability from
# "RMS response per meter Hs" vs Tp, with per‑configuration Hs/Tp limits support.

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
BASE_CMAP_CONT = "turbo"    # continuous fields (e.g., expected heave)
CMAP_OPERABILITY = "jet_r"  # operability maps (reversed jet)
CLIP_PCT = 99.6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NS_CANDIDATES = [
    os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc"),
    os.path.join(BASE_DIR, "metocean_scatter_050deg_NS_monthclim.nc"),
]

# -----------------------------
# Helpers
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

# Percent helpers (generic)
def pct_ticks(): 
    return np.arange(0, 101, 10)

def pct_levels(): 
    return np.linspace(0, 100, 61)

# Dynamic percent helpers (respect lower bound)
def pct_levels_from(lo):
    lo = float(lo)
    # 1%-step levels from lo to 100
    n = int(max(1, np.floor(100 - lo))) + 1
    return np.linspace(lo, 100, n)

def pct_contours_from(lo):
    # contour lines every 10% starting from the next 10% step >= lo
    start = int(np.ceil(lo / 10.0) * 10)
    start = min(start, 100)
    return np.arange(start, 101, 10)

def pct_ticks_from(lo):
    # ticks at lower bound (even if not on a 10% step) + decade ticks to 100
    base = np.arange(0, 101, 10)
    ticks = np.unique(np.concatenate(([float(lo)], base)))
    return ticks[ticks >= lo]

def hs_levels_zoom(arr):
    vmin = float(np.nanmin(arr)); vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    filled   = np.arange(np.floor(vmin/0.1)*0.1, np.ceil(vmax/0.1)*0.1 + 1e-12, 0.1)
    contours = np.arange(np.floor(vmin/0.2)*0.2, np.ceil(vmax/0.2)*0.2 + 1e-12, 0.2)
    return filled, contours, contours

def plot_zoom(lon, lat, data, title, filled, contours, ticks, cmap=BASE_CMAP_CONT, show_grid=True):
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

def interp_rows(M, x_from, x_to):
    return np.vstack([np.interp(x_to, x_from, r) for r in M])

def plot_hs_tp_curve(tp_vals, hs_limits, system_name, note_text=None):
    """Small line plot of the selected system’s Hs/Tp limit curve."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=200)
    ax.plot(tp_vals, hs_limits, "-o", linewidth=1.8, markersize=3)
    ax.set_xlabel("Tp [s]")
    ax.set_ylabel("Hs limit [m]")
    ax.grid(True, alpha=0.35)
    ax.set_title(f"{system_name} — Hs/Tp limit")
    if note_text:
        ax.text(0.01, 0.02, note_text, transform=ax.transAxes, fontsize=7, va="bottom", ha="left", alpha=0.8)
    st.pyplot(fig)

# -----------------------------
# Load dataset (regional)
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

units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c  = bin_centers(hs_edges)    # (hs_bin)
tp_c  = bin_centers(tp_edges)    # (tp_bin)
lat_c = bin_centers(lat_edges)
lon_c = unwrap_lon_centers_from_edges(lon_edges)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Aggregation")
    mode = st.radio("Use", ["By month","Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    month_label = st.selectbox("Month", months, index=0)
    month_idx = dict(zip(months, month_vals))[month_label]

    st.subheader("Wave acceptance (Hs/Tp limit)")
    # Three modes: single constant; single curve CSV; per-config curves CSV
    limit_mode = st.radio(
        "Limit mode",
        ["Single Hcrit", "Single Hs/Tp curve (CSV)", "Per-configuration Hs/Tp limits (CSV)"],
        index=2
    )
    Hcrit = st.number_input("Hs threshold (m)", min_value=0.1, max_value=15.0, value=3.5, step=0.1)
    limit_csv_single = st.file_uploader("Upload single Hs/Tp curve CSV (2 columns: Tp, Hs_limit)", type=["csv"], key="hs_tp_limit_single")
    limit_csv_multi  = st.file_uploader("Upload per-configuration Hs/Tp limits CSV (Tp + one column per config)", type=["csv"], key="hs_tp_limit_multi")

    st.subheader("Heave per meter Hs CSV")
    heave_csv = st.file_uploader("Upload RMS response per meter Hs (by TP)", type=["csv"], key="heave_per_hs_csv")

    st.subheader("Heave acceptance")
    heave_limit = st.number_input("Heave limit (m)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    st.subheader("Display")
    show_grid = st.checkbox("Show grid points", True)

    st.subheader("Colorbar")
    # Keep at most 95% to avoid degenerate range (needs min<max)
    cbar_lower = st.slider("Operability colorbar lower limit (%)", min_value=0, max_value=95, value=50, step=1,
                           help="Sets the lower bound of the operability colorbar. Data are NOT clipped; colors span [lower, 100].")

# -----------------------------
# Probability field
# -----------------------------
if mode == "By month":
    prob = ds["prob"].sel(month=month_idx)
    title_suffix = f" — {month_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"
prob = normalize_pdf(prob)  # (hs,tp,lat,lon)

HS = xr.DataArray(hs_c, dims=["hs_bin"])

# -----------------------------
# Parse heave-per-Hs CSV (rows = configurations; columns = TP1..TPn)
# -----------------------------
def parse_heave_per_hs(uploaded_csv):
    df_raw = pd.read_csv(uploaded_csv, header=None)

    # Find header row that contains many TPk labels
    header_row_idx = None
    for i in range(min(6, len(df_raw))):
        row = df_raw.iloc[i].astype(str).tolist()
        tps = [c.strip() for c in row if re.match(r"^TP\d+$", c.strip(), flags=re.IGNORECASE)]
        if len(tps) >= 3:
            header_row_idx = i
            break
    if header_row_idx is None:
        header_row_idx = 1  # fallback

    hdr = df_raw.iloc[header_row_idx].astype(str)
    tp_cols = [j for j, v in enumerate(hdr) if re.match(r"^TP\d+$", v.strip(), flags=re.IGNORECASE)]
    if len(tp_cols) < 3:
        st.error("Heave CSV: could not locate TP1..TPn header.")
        st.stop()

    # Data rows: first cell = config name; TP columns = numeric values
    records = []
    for i in range(header_row_idx + 1, len(df_raw)):
        first_cell = str(df_raw.iloc[i, 0]).strip()
        if first_cell == "" or first_cell.lower().startswith("tp"):
            continue
        vals = []
        ok = True
        for j in tp_cols:
            try:
                vals.append(float(df_raw.iloc[i, j]))
            except Exception:
                ok = False; break
        if ok:
            records.append((first_cell, vals))
    if not records:
        st.error("Heave CSV: no configuration rows found under TP header.")
        st.stop()

    cfg_names = [name for name, _ in records]
    R_csv = np.vstack([np.array(vals, dtype=float) for _, vals in records])  # [cfg, n_csv]

    # Optional row with Tp centers [s]
    tp_centers_csv = None
    for i in range(header_row_idx + 1, len(df_raw)):
        first_cell = str(df_raw.iloc[i, 0]).strip().lower()
        if first_cell.startswith("tp"):
            tmp = []
            for j in tp_cols:
                try: tmp.append(float(df_raw.iloc[i, j]))
                except Exception: tmp.append(np.nan)
            arr = np.array(tmp, dtype=float)
            if np.isfinite(arr).sum() >= 3:
                tp_centers_csv = arr
            break
    return cfg_names, R_csv, tp_centers_csv

if heave_csv is None:
    st.info("Upload the 'RMS response per meter Hs' CSV to proceed.")
    st.stop()

cfg_names, R_heave_per_Hs_csv, tp_centers_csv = parse_heave_per_hs(heave_csv)
n_csv = R_heave_per_Hs_csv.shape[1]

# Map CSV TP bins to dataset TP bins
if len(tp_c) == n_csv:
    R_use = R_heave_per_Hs_csv
    mapping_note = f"TP mapping: 1:1 ({n_csv} bins)."
elif (tp_centers_csv is not None) and (np.isfinite(tp_centers_csv).sum() >= 3):
    R_use = interp_rows(R_heave_per_Hs_csv, tp_centers_csv, tp_c)
    mapping_note = "TP mapping: interpolated from CSV Tp centers."
else:
    k = np.arange(1, n_csv+1)
    tp_min, tp_max = float(tp_c.min()), float(tp_c.max())
    x_from = tp_min + ((k - 0.5)/n_csv) * (tp_max - tp_min)
    R_use = interp_rows(R_heave_per_Hs_csv, x_from, tp_c)
    mapping_note = "TP mapping: ordinal across dataset Tp range."

# -----------------------------
# Build Hs_limit(Tp) per configuration
# -----------------------------
def build_single_curve(tp_centers, csv_file, fallback_val):
    if csv_file is None:
        return np.full_like(tp_centers, fallback_val, dtype=float)
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin-1")
    cols_low = [c.lower().strip() for c in df.columns]
    def col_like(options):
        for o in options:
            if o in cols_low:
                return df.columns[cols_low.index(o)]
        return None
    tp_col = col_like(["tp (s)", "tp", "tp_s", "period"])
    hs_col = col_like(["hs_limit (m)", "hs limit (m)", "hs_limit", "hs (m)", "hs"])
    if tp_col is None or hs_col is None:
        st.error("Single curve CSV must contain 'Tp (s)' and 'Hs_limit (m)'. Using Hcrit.")
        return np.full_like(tp_centers, fallback_val, dtype=float)
    tp_in = pd.to_numeric(df[tp_col], errors="coerce").astype(float).values
    hs_in = pd.to_numeric(df[hs_col], errors="coerce").astype(float).values
    m = np.isfinite(tp_in) & np.isfinite(hs_in)
    tp_in, hs_in = tp_in[m], hs_in[m]
    if tp_in.size < 2:
        st.error("Single curve CSV must have ≥2 rows. Using Hcrit.")
        return np.full_like(tp_centers, fallback_val, dtype=float)
    order = np.argsort(tp_in)
    return np.interp(tp_centers, tp_in[order], hs_in[order], left=hs_in[order][0], right=hs_in[order][-1])

def parse_limits_per_config(csv_file, cfg_names, tp_centers, fallback_val):
    """
    CSV format:
      Tp (s), A, B, C, ...
      6,     4.0, 4.5, 5.0, ...
    Returns:
      dict: cfg -> 1D array (len(tp_centers))
      missing: list of cfg names not found in CSV
    """
    if csv_file is None:
        # no CSV: same constant limits for all configs
        return {c: np.full_like(tp_centers, fallback_val, dtype=float) for c in cfg_names}, []

    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin-1")

    # Detect Tp column
    low = [c.lower().strip() for c in df.columns]
    tp_candidates = ["tp (s)", "tp", "tp_s", "period"]
    tp_col = None
    for cand in tp_candidates:
        if cand in low:
            tp_col = df.columns[low.index(cand)]
            break
    if tp_col is None:
        st.error("Per-configuration limits CSV must contain a 'Tp (s)' column.")
        return {c: np.full_like(tp_centers, fallback_val, dtype=float) for c in cfg_names}, cfg_names

    # Numeric Tp and per-config series
    tp_in = pd.to_numeric(df[tp_col], errors="coerce").astype(float).values
    ok_tp = np.isfinite(tp_in)
    tp_in = tp_in[ok_tp]

    limits_by_cfg = {}
    missing = []
    # Column name matching: case-insensitive, strip spaces
    col_map = {c.strip().lower(): c for c in df.columns if c != tp_col}
    for cfg in cfg_names:
        key = cfg.strip().lower()
        if key not in col_map:
            missing.append(cfg)
            continue
        vals = pd.to_numeric(df[col_map[key]], errors="coerce").astype(float).values
        vals = vals[ok_tp]
        if np.isfinite(vals).sum() < 2:
            missing.append(cfg)
            continue
        order = np.argsort(tp_in)
        hs_curve = np.interp(tp_centers, tp_in[order], vals[order], left=vals[order][0], right=vals[order][-1])
        limits_by_cfg[cfg] = hs_curve

    # Fill missing with fallback constant
    for cfg in missing:
        limits_by_cfg[cfg] = np.full_like(tp_centers, fallback_val, dtype=float)

    return limits_by_cfg, missing

# Build dictionary of Hs_limit(Tp) per configuration
if limit_mode == "Single Hcrit":
    Hs_limit_by_cfg = {c: np.full_like(tp_c, Hcrit, dtype=float) for c in cfg_names}
    limit_note = "Wave limit: single Hcrit for all configurations."
elif limit_mode == "Single Hs/Tp curve (CSV)":
    single_curve = build_single_curve(tp_c, limit_csv_single, Hcrit)
    Hs_limit_by_cfg = {c: single_curve for c in cfg_names}
    limit_note = "Wave limit: single Hs/Tp curve applied to all configurations."
else:
    Hs_limit_by_cfg, missing_cfgs = parse_limits_per_config(limit_csv_multi, cfg_names, tp_c, Hcrit)
    if missing_cfgs:
        st.warning("Per-configuration limit CSV missing columns for: " + ", ".join(missing_cfgs) + ". Using Hcrit fallback for those.")
    limit_note = "Wave limit: per-configuration Hs/Tp curves from CSV (fallback to Hcrit if missing)."

# -----------------------------
# Compute fields for a SELECTED configuration (MotionOperability UX)
# -----------------------------
cfg = st.selectbox("Hull alternative", cfg_names, index=0)
i_cfg = cfg_names.index(cfg)

# --- Small line plot of the selected system’s Hs/Tp limit curve ---
Hs_limit_tp_sel = Hs_limit_by_cfg[cfg]
curve_col, spacer = st.columns([1.2, 3.8])
with curve_col:
    plot_hs_tp_curve(tp_c, Hs_limit_tp_sel, cfg, note_text=limit_note)

# Heave per meter Hs vs Tp (for selected cfg)
fTp = xr.DataArray(R_use[i_cfg], dims=["tp_bin"])       # m/m

# Sea-state heave magnitude
HS2D = xr.DataArray(hs_c, dims=["hs_bin"]).broadcast_like(prob)
TPmask = fTp.broadcast_like(prob)
M_heave = HS2D * TPmask                                  # m

# Wave acceptance for selected cfg
HsLim2D_sel = xr.DataArray(Hs_limit_tp_sel, dims=["tp_bin"]).broadcast_like(prob)
I_wave_sel = (HS2D <= HsLim2D_sel).astype(float)

# Heave acceptance
I_heave = (M_heave <= heave_limit).astype(float)

# Expected heave
E_heave = (prob * M_heave).sum(dim=("hs_bin","tp_bin"))  # (lat,lon)

# Operabilities
P_heave = (prob * I_heave   ).sum(dim=("hs_bin","tp_bin")) * 100.0
P_wave  = (prob * I_wave_sel).sum(dim=("hs_bin","tp_bin")) * 100.0
P_both  = (prob * I_heave * I_wave_sel).sum(dim=("hs_bin","tp_bin")) * 100.0

# -----------------------------
# Metric selector (as in MotionOperability)
# -----------------------------
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

# Prep 2D arrays
def prep(field):
    arr = field.transpose("lat3_bin","lon3_bin").values
    return to_sorted_lon_lat(arr, lat_c, lon_edges)

heave2, latp, lonp = prep(E_heave)
heave_hi = np.nanpercentile(heave2, CLIP_PCT)
heave2 = np.clip(heave2, None, heave_hi)

p_heave2, latp, lonp = prep(P_heave)
p_wave2,  latp, lonp = prep(P_wave)
p_both2,  latp, lonp = prep(P_both)

# -----------------------------
# Render map depending on metric
# -----------------------------
if metric == "Expected heave (m)":
    filled, contours, ticks = hs_levels_zoom(heave2)
    plot_zoom(
        lonp, latp, heave2,
        f"Expected heave (m) — {cfg}{title_suffix}",
        filled, contours, ticks,
        cmap=BASE_CMAP_CONT, show_grid=show_grid
    )

elif metric == "Operability: heave ≤ limit (%)":
    filled = pct_levels_from(cbar_lower)
    contours = pct_contours_from(cbar_lower)
    ticks = pct_ticks_from(cbar_lower)
    plot_zoom(
        lonp, latp, p_heave2,
        f"Operability (%) — Heave ≤ {heave_limit:.2f} m{title_suffix}",
        filled, contours, ticks,
        cmap=CMAP_OPERABILITY, show_grid=show_grid
    )

elif metric == "Operability: wave ≤ Hs/Tp limit (%)":
    filled = pct_levels_from(cbar_lower)
    contours = pct_contours_from(cbar_lower)
    ticks = pct_ticks_from(cbar_lower)
    plot_zoom(
        lonp, latp, p_wave2,
        "Operability (%) — Wave (Hs/Tp limit)" + title_suffix,
        filled, contours, ticks,
        cmap=CMAP_OPERABILITY, show_grid=show_grid
    )

else:  # ALL limits
    filled = pct_levels_from(cbar_lower)
    contours = pct_contours_from(cbar_lower)
    ticks = pct_ticks_from(cbar_lower)
    plot_zoom(
        lonp, latp, p_both2,
        f"Operability (%) — Wave ∩ Heave — {cfg}{title_suffix}",
        filled, contours, ticks,
        cmap=CMAP_OPERABILITY, show_grid=show_grid
    )

st.caption(mapping_note + "  |  " + limit_note + f"  |  Colorbar lower bound: {cbar_lower:.0f}%")

# -----------------------------
# Multi‑configuration operability comparison (uses each cfg's own Hs/Tp curve)
# -----------------------------
st.markdown("## Operability comparison between all configurations")

results = []
for j, cfg_j in enumerate(cfg_names):
    fTp_j = xr.DataArray(R_use[j], dims=["tp_bin"])
    M_heave_j = HS2D * fTp_j.broadcast_like(prob)

    # per‑cfg wave limit
    Hs_limit_tp_j = Hs_limit_by_cfg[cfg_j]
    I_wave_j  = (HS2D <= xr.DataArray(Hs_limit_tp_j, dims=["tp_bin"]).broadcast_like(prob)).astype(float)
    I_heave_j = (M_heave_j <= heave_limit).astype(float)

    # Maps (lat,lon) of operability %
    P_heave_map = (prob * I_heave_j).sum(dim=("hs_bin","tp_bin")) * 100.0
    P_wave_map  = (prob * I_wave_j ).sum(dim=("hs_bin","tp_bin")) * 100.0
    P_both_map  = (prob * I_heave_j * I_wave_j).sum(dim=("hs_bin","tp_bin")) * 100.0

    # Reduce to scalars via spatial mean (unweighted); could switch to area-weighted if desired
    P_heave_j = float(P_heave_map.mean(dim=("lat3_bin","lon3_bin"), skipna=True))
    P_wave_j  = float(P_wave_map.mean(dim=("lat3_bin","lon3_bin"),  skipna=True))
    P_both_j  = float(P_both_map.mean(dim=("lat3_bin","lon3_bin"),  skipna=True))

    results.append({
        "Configuration": cfg_j,
        "Heave-only (%)": P_heave_j,
        "Wave-only (%)": P_wave_j,
        "Combined (%)": P_both_j
    })

df_ops = pd.DataFrame(results).set_index("Configuration")
st.dataframe(df_ops.style.format("{:.1f}"))

st.markdown("### Operability comparison chart")
fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(len(cfg_names)); w = 0.25
ax.bar(x - w, df_ops["Heave-only (%)"], width=w, label="Heave-only")
ax.bar(x,     df_ops["Wave-only (%)"],  width=w, label="Wave-only")
ax.bar(x + w, df_ops["Combined (%)"],   width=w, label="Combined")
ax.set_xticks(x); ax.set_xticklabels(cfg_names)
ax.set_ylabel("Operability (%)"); ax.set_ylim(0, 100)
ax.legend(); ax.grid(True, alpha=0.3)
st.pyplot(fig)