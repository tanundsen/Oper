# 01_Metocean.py — Metocean Explorer (hybrid: 3° global + 0.5° zoom)
# -------------------------------------------------------------------
# • Global map: PlateCarree (fixed)
# • Zoomed map: PlateCarree by default; selector for Mercator / Lambert
# • Hybrid data loading:
#       - Global view → metocean_monthclim.nc (3° grid)
#       - Zoomed view → metocean_scatter_050deg_NS_monthclim.nc (0.5°)
# • Zoomed Hs styling: filled every 0.1 m, contour lines every 0.2 m
# • Zoomed Tp styling: contour lines every 0.5 s
# • % metrics fixed 0–100 %
# • Coasts: 110m globally (speed), 10m zoomed (detail)
# • POIs (1–10) with names on zoomed map only
# • NEW: Optional per‑Tp Hs limit curve applied to BOTH Operability and P(Hs>…)
# -------------------------------------------------------------------

import math
import os
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import patheffects
from io import StringIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data sources
GLOBAL_DATA_PATH   = os.path.join(BASE_DIR, "..", "metocean_monthclim.nc")                      # 3°
REGIONAL_DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc")    # 0.5°

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(layout="wide")
st.header("🌍 Global wave statistics")

# -----------------------------
# Helpers & caching
# -----------------------------
@st.cache_resource
def load_metocean(path: str) -> xr.Dataset:
    return xr.open_dataset(path)

def bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

def unwrap_lon_centers_from_edges(lon_edges):
    lon_c = bin_centers(lon_edges)
    if np.nanmax(lon_edges) > 180:
        lon_c = ((lon_c + 180) % 360) - 180
    return lon_c

def to_sorted_lon_lat(field2d, lat_c, lon_edges):
    flip_lat = False
    if lat_c[0] > lat_c[-1]:
        field2d = field2d[::-1, :]
        lat_c = lat_c[::-1]
        flip_lat = True
    lon_unsorted = unwrap_lon_centers_from_edges(lon_edges)
    lon_sort_idx = np.argsort(lon_unsorted)
    lon_sorted = lon_unsorted[lon_sort_idx]
    field2d_sorted = field2d[:, lon_sort_idx]
    lon_inv = np.argsort(lon_sort_idx)
    return field2d_sorted, lat_c, lon_sorted, flip_lat, lon_sort_idx, lon_inv

def is_hs_quantity(label: str) -> bool:
    if "%" in label:
        return False
    return "hs" in label.lower()

def hs_ticks(step=0.5, vmin=0, vmax=10):
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + step/2, step)

def hs_shading(field, n=60):
    vmin = np.nanmin(field); vmax = np.nanmax(field)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0, 1, n)
    return np.linspace(vmin, vmax, n)

def tp_ticks(step=1.0, vmin=None, vmax=None):
    if vmin is None: vmin = 0
    if vmax is None: vmax = 20
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + step*0.5, step)

def tp_shading(field, n=80):
    vmin = np.nanmin(field); vmax = np.nanmax(field)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0, 20, n)
    return np.linspace(vmin, vmax, n)

def pct_ticks(): return np.arange(0, 101, 10)
def pct_shading(n=61): return np.linspace(0, 100, n)

def auto_levels(arr, n=50):
    vmin = np.nanmin(arr); vmax = np.nanmax(arr)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0, 1, n)
    return np.linspace(vmin, vmax, n)

def normalize_pdf(prob):
    tot = prob.sum(dim=("hs_bin","tp_bin"))
    return xr.where(tot > 0, prob/tot, 0)

def percentile_from_cdf(cdf, centers, q):
    idx_hi = (cdf >= q).argmax(dim="hs_bin")
    idx_lo = xr.where(idx_hi > 0, idx_hi - 1, 0)
    c_lo = cdf.isel(hs_bin=idx_lo); c_hi = cdf.isel(hs_bin=idx_hi)
    cen = xr.DataArray(centers, dims=["hs_bin"])
    h_lo = cen.isel(hs_bin=idx_lo); h_hi = cen.isel(hs_bin=idx_hi)
    denom = xr.where((c_hi - c_lo) > 0, c_hi - c_lo, 1)
    w = (q - c_lo)/denom
    return h_lo + w*(h_hi - h_lo)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Data")
    st.caption("Global: metocean_monthclim.nc (3°)  •  Zoom: metocean_scatter_050deg_NS_monthclim.nc (0.5°)")

    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month","Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    chosen_label = st.selectbox("Month", months, index=4)
    label_to_idx = dict(zip(months, month_vals))

    st.subheader("Statistic")
    Hcrit = st.number_input("Hs threshold (m)", 0.1, 15.0, 2.5, step=0.1)

    # NEW: threshold mode — single Hcrit vs per-Tp curve
    threshold_mode = st.radio(
        "Threshold mode",
        ["Single Hcrit", "Hs limit per Tp (table)"],
        index=0,
        help="Use one constant Hcrit or define a custom Hs limit for each Tp bin."
    )

    stat = st.selectbox(
        "Statistic:",
        [
            "Mean Hs (m)",
            "Mean Tp (s)",
            "Hs P50 (m)",
            "Hs P90 (m)",
            "Hs P95 (m)",
            f"P(Hs > Hcrit) (%)",
            f"Operability (% time Hs ≤ Hcrit)"
        ]
    )

    st.subheader("View")
    zoom_ns = st.checkbox("North Sea zoom", value=False)

    # Projection selector for zoomed view (default PlateCarree)
    zoom_proj_name = st.selectbox(
        "Zoom projection",
        ["PlateCarree (default)", "Mercator", "Lambert Conformal"],
        index=0,
        help="Applies only when 'North Sea zoom' is enabled. Global view is always PlateCarree."
    )

    st.subheader("Debug")
    show_debug = st.checkbox("Show debug", False)

# -----------------------------
# Fixed settings
# -----------------------------
# lon_min, lon_max, lat_min, lat_max
ZOOM_EXTENT = [-13, 35, 52, 76]
base_cmap = "turbo"
levels_generic = 50
clip_pct_robust = 99.6  # robust cap for shading

# -----------------------------
# Points of Interest (North Sea fields) – decimal degrees
# -----------------------------
POIS = [
    {"name": "Ekofisk",         "nr": 1,  "lat": 56.5333, "lon":  3.2000},
    {"name": "Ula",             "nr": 2,  "lat": 57.1000, "lon":  2.8333},
    {"name": "Sleipner",        "nr": 3,  "lat": 58.3667, "lon":  1.9000},
    {"name": "Alvheim",         "nr": 4,  "lat": 59.5667, "lon":  1.9667},
    {"name": "Oseberg",         "nr": 5,  "lat": 60.5000, "lon":  2.8333},
    {"name": "Knarr",           "nr": 6,  "lat": 61.8833, "lon":  3.8333},
    {"name": "Ormen Lange",     "nr": 7,  "lat": 63.2500, "lon":  5.0000},
    {"name": "Skarv",           "nr": 8,  "lat": 65.7500, "lon":  7.6667},
    {"name": "Aasta Hansteen",  "nr": 9,  "lat": 67.0000, "lon":  8.0000},
    {"name": "Johan Castberg",  "nr": 10, "lat": 72.0000, "lon": 22.5000},
]

# -----------------------------
# Load dataset (hybrid)
# -----------------------------
ds = load_metocean(REGIONAL_DATA_PATH if zoom_ns else GLOBAL_DATA_PATH)

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
    hs_edges = hs_edges/100.0

hs_c = bin_centers(hs_edges)
tp_c = bin_centers(tp_edges)
lat_c_unsorted = bin_centers(lat_edges)

# -----------------------------
# Select probability field
# -----------------------------
if agg == "By month":
    prob = ds["prob"].sel(month=label_to_idx[chosen_label])
    title_suffix = f" — {chosen_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"

_tot_before = prob.sum(dim=("hs_bin","tp_bin"))
prob = normalize_pdf(prob)  # dims: hs_bin, tp_bin, lat3_bin, lon3_bin

# ---- Hs-limit per Tp: editable table + CSV import/export (old workflow with file support) ----
def init_per_tp_limits(default_val: float, tp_centers: np.ndarray):
    key = "hs_per_tp_limits"
    if (key not in st.session_state) or (len(st.session_state[key]) != len(tp_centers)):
        st.session_state[key] = [default_val] * len(tp_centers)
    return key

if threshold_mode == "Hs limit per Tp (table)":
    limits_key = init_per_tp_limits(Hcrit, tp_c)

    st.subheader("Hs limit per Tp")

    # ------------- CSV import -------------
    up = st.file_uploader("Import CSV (columns: 'Tp (s)', 'Hs_limit (m)')", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except UnicodeDecodeError:
            df_in = pd.read_csv(up, encoding="latin-1")

        # Flexible column detection
        def find_col(options, cols):
            opts = [o.lower().strip() for o in options]
            for c in cols:
                if c.lower().strip() in opts:
                    return c
            return None

        tp_col = find_col(["tp (s)", "tp", "tp_s"], df_in.columns)
        hs_col = find_col(["hs_limit (m)", "hs limit (m)", "hs_limit", "hs (m)", "hs"], df_in.columns)

        if tp_col is None or hs_col is None:
            st.error("CSV must contain columns 'Tp (s)' and 'Hs_limit (m)'.")
        else:
            tp_in = pd.to_numeric(df_in[tp_col], errors="coerce").astype(float).values
            hs_in = pd.to_numeric(df_in[hs_col], errors="coerce").astype(float).values

            mask_valid = np.isfinite(tp_in) & np.isfinite(hs_in)
            tp_in, hs_in = tp_in[mask_valid], hs_in[mask_valid]
            if tp_in.size < 2:
                st.error("CSV must provide at least two valid Tp rows.")
            else:
                # sort & interpolate onto internal tp_c
                order = np.argsort(tp_in)
                tp_in, hs_in = tp_in[order], hs_in[order]
                hs_interp = np.interp(tp_c, tp_in, hs_in, left=hs_in[0], right=hs_in[-1])
                hs_interp = np.clip(np.round(hs_interp, 1), 0.0, 15.0)
                st.session_state[limits_key] = hs_interp.tolist()
                st.success(f"Imported {tp_in.size} CSV rows and mapped to {len(tp_c)} Tp bins.")
                st.rerun()

    # ------------- Editable table (old workflow) -------------
    df_limits = pd.DataFrame({
        "Tp (s)": tp_c,
        "Hs_limit (m)": st.session_state[limits_key]
    })
    df_limits = st.data_editor(
        df_limits, num_rows="fixed", hide_index=True, use_container_width=True,
        column_config={
            "Tp (s)": st.column_config.NumberColumn("Tp (s)", disabled=True, format="%.1f"),
            "Hs_limit (m)": st.column_config.NumberColumn(
                "Hs_limit (m)", min_value=0.0, max_value=15.0, step=0.1, format="%.1f",
                help="Edit with spinner/mouse wheel; step = 0.1 m"
            ),
        },
        key="per_tp_editor_old"
    )
    st.session_state[limits_key] = df_limits["Hs_limit (m)"].round(1).tolist()
    hs_limit_curve = np.array(st.session_state[limits_key], dtype=float)

    # ------------- CSV export (current curve) -------------
    templ_buf = StringIO()
    pd.DataFrame({"Tp (s)": tp_c, "Hs_limit (m)": [Hcrit]*len(tp_c)}).to_csv(templ_buf, index=False)
    st.download_button("Download CSV template (flat Hcrit)", templ_buf.getvalue(),
                       file_name="hs_limits_template.csv", mime="text/csv")

    ex_vals = np.clip(0.8 + 0.1*(tp_c - float(tp_c.min()))/2.0, 0.6, 3.5)  # simple ramp example
    ex_buf = StringIO()
    pd.DataFrame({"Tp (s)": tp_c, "Hs_limit (m)": np.round(ex_vals, 1)}).to_csv(ex_buf, index=False)
    st.download_button("Download CSV example (ramped)", ex_buf.getvalue(),
                       file_name="hs_limits_example.csv", mime="text/csv")

    cur_buf = StringIO()
    pd.DataFrame({"Tp (s)": tp_c, "Hs_limit (m)": st.session_state[limits_key]}).to_csv(cur_buf, index=False)
    st.download_button("Download current limits", cur_buf.getvalue(),
                       file_name="hs_limits_current.csv", mime="text/csv")

else:
    hs_limit_curve = None


# -----------------------------
# (NEW) Per‑Tp Hs limit table & 2D masks
# -----------------------------
def get_per_tp_limits_ui(tp_centers, default_val: float):
    """Editable table of Hs limits per Tp. Persists in session_state."""
    key = "hs_per_tp_limits"
    # Initialize or reset if length changed (different dataset)
    if (key not in st.session_state) or (len(st.session_state[key]) != len(tp_centers)):
        st.session_state[key] = [default_val] * len(tp_centers)

    st.subheader("Hs limit per Tp")
    df = pd.DataFrame({"Tp (s)": tp_centers, "Hs_limit (m)": st.session_state[key]})
    edited = st.data_editor(df, num_rows="fixed", use_container_width=True)
    st.session_state[key] = edited["Hs_limit (m)"].tolist()

    return np.array(st.session_state[key], dtype=float)

# Build per‑Tp limits (if mode selected)
if threshold_mode == "Hs limit per Tp (table)":
    hs_limit_curve = get_per_tp_limits_ui(tp_c, Hcrit)  # array len = tp_bins
else:
    hs_limit_curve = None  # signals single‑Hcrit mode

# Pre-broadcast helpers for 2D logical masks
Hs_1D = xr.DataArray(hs_c, dims=["hs_bin"])
Tp_limit_1D = None if hs_limit_curve is None else xr.DataArray(hs_limit_curve, dims=["tp_bin"])

# -----------------------------
# Compute statistics
# -----------------------------
hs_w = xr.DataArray(hs_c, dims=["hs_bin"])
tp_w = xr.DataArray(tp_c, dims=["tp_bin"])

mean_hs = (prob*hs_w).sum(dim=("hs_bin","tp_bin"))
mean_tp = (prob*tp_w).sum(dim=("hs_bin","tp_bin"))

hs_pdf = prob.sum(dim="tp_bin")
hs_cdf = hs_pdf.cumsum(dim="hs_bin")
hs_p50 = percentile_from_cdf(hs_cdf, hs_c, 0.50)
hs_p90 = percentile_from_cdf(hs_cdf, hs_c, 0.90)
hs_p95 = percentile_from_cdf(hs_cdf, hs_c, 0.95)

# --- Exceedance / Operability with per‑Tp support ---
if hs_limit_curve is None:
    # Single Hcrit (legacy)
    mask_exceed_1d = xr.DataArray((hs_c > Hcrit).astype(float), dims=["hs_bin"])
    p_exceed = (hs_pdf * mask_exceed_1d).sum(dim="hs_bin") * 100.0
    p_oper   = 100.0 - p_exceed
else:
    # Per-Tp: build 2D masks on (hs_bin, tp_bin)
    Hs2D = Hs_1D.broadcast_like(prob)                    # hs x tp x lat x lon
    HsLim2D = Tp_limit_1D.broadcast_like(prob)           # hs x tp x lat x lon

    mask_exceed_2d = (Hs2D >  HsLim2D).astype(float)
    mask_oper_2d   = (Hs2D <= HsLim2D).astype(float)

    p_exceed = (prob * mask_exceed_2d).sum(dim=("hs_bin","tp_bin")) * 100.0
    p_oper   = (prob * mask_oper_2d  ).sum(dim=("hs_bin","tp_bin")) * 100.0

# -----------------------------
# Select final field
# -----------------------------
if stat == "Mean Hs (m)":
    field = mean_hs
    label = "Mean Hs (m)" + title_suffix

elif stat == "Mean Tp (s)":
    field = mean_tp
    label = "Mean Tp (s)" + title_suffix

elif stat == "Hs P50 (m)":
    field = hs_p50
    label = "Hs P50 (m)" + title_suffix

elif stat == "Hs P90 (m)":
    field = hs_p90
    label = "Hs P90 (m)" + title_suffix

elif stat == "Hs P95 (m)":
    field = hs_p95
    label = "Hs P95 (m)" + title_suffix

elif stat.startswith("P(Hs"):
    field = p_exceed
    if hs_limit_curve is None:
        label = f"P(Hs > {Hcrit:.1f} m) (%)" + title_suffix
    else:
        label = "P(Hs > Hs_limit(Tp)) (%)" + title_suffix

else:
    field = p_oper
    if hs_limit_curve is None:
        label = f"Operability (% time Hs ≤ {Hcrit:.1f} m)" + title_suffix
    else:
        label = "Operability (% time Hs ≤ Hs_limit(Tp))" + title_suffix

# -----------------------------
# Prepare 2D field (sorted lon/lat)
# -----------------------------
field2d = field.transpose("lat3_bin","lon3_bin").values
field2d, latp, lonp, flip_lat, lon_sort_idx, lon_inv = to_sorted_lon_lat(
    field2d, lat_c_unsorted, lon_edges
)

# -----------------------------
# Projection factory for zoom
# -----------------------------
def get_zoom_projection(name: str):
    if name.startswith("PlateCarree"):
        return ccrs.PlateCarree()
    if name == "Mercator":
        return ccrs.Mercator(central_longitude=10, min_latitude=40, max_latitude=82)
    if name == "Lambert Conformal":
        return ccrs.LambertConformal(
            central_longitude=10, central_latitude=60, standard_parallels=(50, 65)
        )
    return ccrs.PlateCarree()

# -----------------------------
# Color scaling & levels
# -----------------------------
def region_slice(arr2d, lons, lats, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    j = (lons >= lon_min) & (lons <= lon_max)
    i = (lats >= lat_min) & (lats <= lat_max)
    if not np.any(i) or not np.any(j):
        return arr2d  # fallback
    return arr2d[np.ix_(i, j)]

def safe_minmax(a):
    vmin = np.nanmin(a); vmax = np.nanmax(a)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0, 1.0
    return float(vmin), float(vmax)

def prep_levels(arr, label, prefer_ticks_from=None, zoom=False):
    """
    - prefer_ticks_from: array used for deriving ticks (zoomed subset)
    - zoom: True only for zoomed view
    Returns: (filled_levels, contour_levels, colorbar_ticks)
    """
    base = prefer_ticks_from if prefer_ticks_from is not None else arr
    vmin, vmax = safe_minmax(base)

    # Percent metrics: fixed 0–100
    if "P(Hs" in label or "Operability" in label:
        return pct_shading(), pct_ticks(), pct_ticks()

    # Tp metrics (Mean Tp)
    if label.startswith("Mean Tp"):
        if zoom:
            contours = np.arange(math.floor(vmin/0.5)*0.5, math.ceil(vmax/0.5)*0.5 + 1e-9, 0.5)
            filled = tp_shading(base)
            ticks = contours
            return filled, contours, ticks
        else:
            ticks = tp_ticks(1.0, vmin, vmax)
            return tp_shading(base), ticks, ticks

    # Hs-based metrics (Mean Hs, Hs PXX)
    if is_hs_quantity(label):
        if zoom:
            # Filled colors every 0.1 m
            filled = np.arange(math.floor(vmin/0.1)*0.1, math.ceil(vmax/0.1)*0.1 + 1e-9, 0.1)
            # Contour lines every 0.2 m
            contours = np.arange(math.floor(vmin/0.2)*0.2, math.ceil(vmax/0.2)*0.2 + 1e-9, 0.2)
            ticks = contours
            return filled, contours, ticks
        else:
            ticks = hs_ticks(0.5, vmin, vmax)
            return hs_shading(base), ticks, ticks

    # Fallback
    lev = auto_levels(base, levels_generic)
    return lev, lev, None

is_percent_metric = ("P(Hs" in label) or ("Operability" in label)

# Robust caps
if is_percent_metric:
    hi_global = 100.0
else:
    hi_global = np.nanpercentile(field2d, clip_pct_robust)

# When zoomed: adapt color range to zoomed region (except % metrics)
if zoom_ns and not is_percent_metric:
    region = region_slice(field2d, lonp, latp, ZOOM_EXTENT)
    hi_zoom = np.nanpercentile(region, clip_pct_robust)
    hi_use = hi_zoom
    ticks_base = np.clip(region, None, hi_zoom)
else:
    hi_use = hi_global
    ticks_base = np.clip(field2d, None, hi_global)

arr_plot = np.clip(field2d, None, hi_use)

filled_levels, contour_levels, cbar_ticks = prep_levels(
    arr_plot, label, prefer_ticks_from=ticks_base, zoom=zoom_ns
)

cmap_use = base_cmap + "_r" if "Operability" in label else base_cmap

# -----------------------------
# POI drawer
# -----------------------------
def draw_pois(ax, pois):
    """Draw numbered markers and 'Nr Name' labels with a subtle white halo."""
    # markers
    lons = [p["lon"] for p in pois]
    lats = [p["lat"] for p in pois]
    ax.scatter(lons, lats, s=28, c="black", marker="o",
               transform=ccrs.PlateCarree(), zorder=20)

    # per-point offsets (deg) to reduce overlaps
    offsets = {
        1:(0.12,0.10), 2:(0.12,0.10), 3:(0.14,0.10), 4:(0.14,0.12), 5:(0.14,0.12),
        6:(0.12,0.12), 7:(0.12,0.12), 8:(0.12,0.12), 9:(0.12,0.12), 10:(0.14,0.12)
    }
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
    for p in pois:
        dx, dy = offsets.get(p["nr"], (0.12, 0.12))
        ax.text(p["lon"] + dx, p["lat"] + dy, f'{p["nr"]} {p["name"]}',
                transform=ccrs.PlateCarree(), fontsize=7, color="black",
                zorder=21, path_effects=halo)

# -----------------------------
# Plot function
# -----------------------------
def plot_map(lon_c, lat_c, arr2d, title, filled, contours, cmap, ticks,
             use_zoom: bool, zoom_proj):
    # Axes projection
    ax_proj = zoom_proj if use_zoom else ccrs.PlateCarree()

    fig = plt.figure(figsize=(15, 6), dpi=(200 if use_zoom else 150))
    ax = plt.axes(projection=ax_proj)

    # filled colors
    cf = ax.contourf(
        lon_c, lat_c, arr2d,
        levels=filled,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),  # data is lon/lat
        zorder=1
    )

    # contour lines
    try:
        cs = ax.contour(
            lon_c, lat_c, arr2d,
            levels=contours,
            colors="black",
            linewidths=0.45 if use_zoom else 0.4,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        # force renderer to prepare paths (helps clabel density)
        ax.figure.canvas.draw()

        # denser labels when zoomed
        ax.clabel(
            cs,
            fontsize=6,
            inline=True,
            inline_spacing=(1 if use_zoom else 6),
            fmt="%g",
            manual=False,
            rightside_up=True
        )
    except Exception:
        pass

    # feature detail
    feature_scale = "10m" if use_zoom else "110m"
    ax.add_feature(cfeature.LAND.with_scale(feature_scale),
                   facecolor="lightgray", edgecolor="none", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale(feature_scale),
                   linewidth=0.7 if use_zoom else 0.4, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale(feature_scale),
                   linewidth=0.3 if use_zoom else 0.2, zorder=12)

    # extent
    if use_zoom:
        ax.set_extent(ZOOM_EXTENT, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    # POIs only on zoomed map
    if use_zoom:
        draw_pois(ax, POIS)

    # colorbar
    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)

    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

    # compact legend (zoom only)
    if use_zoom:
        legend_items = ", ".join([f'{p["nr"]}: {p["name"]}' for p in POIS])
        st.caption(f"**Points of interest (Nr → Name):** {legend_items}")

# -----------------------------
# Render
# -----------------------------
plot_map(
    lonp, latp, arr_plot,
    label,
    filled_levels,
    contour_levels,
    cmap_use,
    cbar_ticks,
    use_zoom=zoom_ns,
    zoom_proj=get_zoom_projection(zoom_proj_name)
)

# -----------------------------
# Debug
# -----------------------------
if show_debug:
    st.write(
        "Totals BEFORE normalization:",
        float(_tot_before.min()),
        float(_tot_before.max())
    )
    src = REGIONAL_DATA_PATH if zoom_ns else GLOBAL_DATA_PATH
    st.write(
        "Color cap (hi_use):", float(hi_use),
        "| Zoomed:", bool(zoom_ns),
        "| Zoom projection:", zoom_proj_name,
        "| Source:", os.path.basename(src),
        "| Threshold mode:", threshold_mode
    )