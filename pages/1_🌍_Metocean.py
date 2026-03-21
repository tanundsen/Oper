# 01_Metocean.py — Metocean Explorer (regional zooms + robust renderers + deep debug)
# -----------------------------------------------------------------------------------
# What's included
# • Region presets: North Sea, Mediterranean
# • MED 0.5° dataset support
# • Auto-fit extent to data (with padding)
# • Robust color scaling (no-NaN fallbacks)
# • THREE renderers: Contourf / Pcolormesh / Imshow (force raster)
# • Grid points overlay at high zorder (never hidden)
# • Deep debug readouts to diagnose lon/lat/extent/renderer issues
# -----------------------------------------------------------------------------------

import math
import os
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objects as go
from matplotlib import patheffects
from matplotlib import colors as mcolors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Data sources
# -----------------------------
GLOBAL_DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_monthclim.nc")  # 3°
REGIONAL_DATA_PATHS = {
    "North Sea": os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc"),
    "Mediterranean": os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_MED_monthclim.nc"),
}

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
    # wrap to [-180, 180] if needed
    if np.nanmax(lon_edges) > 180:
        lon_c = ((lon_c + 180) % 360) - 180
    return lon_c

def to_sorted_lon_lat(field2d, lat_c, lon_edges):
    # Lat ascending
    if lat_c[0] > lat_c[-1]:
        field2d = field2d[::-1, :]
        lat_c = lat_c[::-1]
    # Lon sorted (and possibly wrapped)
    lon_unsorted = unwrap_lon_centers_from_edges(lon_edges)
    lon_sort_idx = np.argsort(lon_unsorted)
    lon_sorted = lon_unsorted[lon_sort_idx]
    field2d_sorted = field2d[:, lon_sort_idx]
    return field2d_sorted, lat_c, lon_sorted

def is_hs_quantity(label: str) -> bool:
    return ("%" not in label) and ("hs" in label.lower())

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

def finite_stats(a: np.ndarray):
    if a is None:
        return 0, None, None
    finite = np.isfinite(a)
    cnt = int(np.count_nonzero(finite))
    if cnt == 0:
        return 0, None, None
    return cnt, float(np.nanmin(a)), float(np.nanmax(a))

def region_slice(arr2d, lons, lats, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    j = (lons >= lon_min) & (lons <= lon_max)
    i = (lats >= lat_min) & (lats <= lat_max)
    if (not np.any(i)) or (not np.any(j)):
        return arr2d
    sub = arr2d[np.ix_(i, j)]
    cnt, _, _ = finite_stats(sub)
    return sub if cnt > 0 else arr2d

def clamp_extent(ext, lon_lo=-180.0, lon_hi=180.0, lat_lo=-90.0, lat_hi=90.0):
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = ext
    return [
        max(lon_lo, min(lon_hi, LON_MIN)),
        max(lon_lo, min(lon_hi, LON_MAX)),
        max(lat_lo, min(lat_hi, LAT_MIN)),
        max(lat_lo, min(lat_hi, LAT_MAX)),
    ]

def auto_fit_extent(lon_c, lat_c, arr2d, base_extent=None, pad_deg=1.0):
    work = None
    if base_extent is not None:
        work = region_slice(arr2d, lon_c, lat_c, base_extent)
        cnt, _, _ = finite_stats(work)
        if cnt == 0:
            work = None
    if work is None:
        work = arr2d
        cnt, _, _ = finite_stats(work)
        if cnt == 0:
            return base_extent
    finite_mask = np.isfinite(work)
    if not np.any(finite_mask):
        return base_extent
    if base_extent is not None:
        lon_min_b, lon_max_b, lat_min_b, lat_max_b = base_extent
        jmask = (lon_c >= lon_min_b) & (lon_c <= lon_max_b)
        imask = (lat_c >= lat_min_b) & (lat_c <= lat_max_b)
    else:
        jmask = np.ones_like(lon_c, dtype=bool)
        imask = np.ones_like(lat_c, dtype=bool)
    arr_masked = arr2d[np.ix_(imask, jmask)]
    rows = np.any(np.isfinite(arr_masked), axis=1)
    cols = np.any(np.isfinite(arr_masked), axis=0)
    if not np.any(rows) or not np.any(cols):
        return base_extent
    i_idx = np.where(rows)[0]
    j_idx = np.where(cols)[0]
    lat_min = float(lat_c[imask][i_idx[0]])
    lat_max = float(lat_c[imask][i_idx[-1]])
    lon_min = float(lon_c[jmask][j_idx[0]])
    lon_max = float(lon_c[jmask][j_idx[-1]])
    ext = [lon_min - pad_deg, lon_max + pad_deg, lat_min - pad_deg, lat_max + pad_deg]
    return clamp_extent(ext)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Data")
    st.caption(
        "Global: metocean_monthclim.nc (3°)\n"
        "Regionals (0.5°): NS = metocean_scatter_050deg_NS_monthclim.nc, "
        "MED = metocean_scatter_050deg_MED_monthclim.nc"
    )
    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month","Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    chosen_label = st.selectbox("Month", months, index=4)
    label_to_idx = dict(zip(months, month_vals))

    st.subheader("Statistic")
    Hcrit = st.number_input("Hs threshold (m)", 0.1, 15.0, 2.5, step=0.1)
    threshold_mode = st.radio(
        "Threshold mode",
        ["Single Hcrit", "Hs limit per Tp (CSV + graph)"],
        index=0,
        help="Use one constant Hcrit or import a CSV with one Hs limit per Tp bin."
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
    zoom_enabled = st.checkbox("Enable zoom", value=False)
    zoom_region = st.selectbox("Zoom region", ["North Sea", "Mediterranean"], index=0)
    auto_fit = st.checkbox("Auto-fit extent to data", value=True)
    pad_deg = st.number_input("Padding (deg) for auto-fit", 0.0, 10.0, 1.0, step=0.5)
    show_grid_points = st.checkbox("Show metocean grid points (both views)", value=True)
    zoom_proj_name = st.selectbox(
        "Zoom projection",
        ["PlateCarree (default)", "Mercator", "Lambert Conformal"],
        index=0
    )
    # Renderer selector
    filled_renderer = st.radio(
        "Filled renderer",
        ["Contourf (smooth)", "Pcolormesh (robust)", "Imshow (force raster)"],
        horizontal=True,
        index=2  # default to force-raster for robustness
    )

    st.subheader("Debug")
    show_debug = st.checkbox("Show debug", False)

# -----------------------------
# Fixed settings
# -----------------------------
ZOOM_EXTENTS = {
    "North Sea": [-13, 35, 52, 76],
    "Mediterranean": [-10, 40, 30, 46],
}
base_cmap = "turbo"
levels_generic = 50
clip_pct_robust = 99.6

# -----------------------------
# Points of Interest (North Sea only)
# -----------------------------
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
]

# -----------------------------
# Dataset selection
# -----------------------------
def pick_dataset_path(use_zoom: bool, region_name: str) -> str:
    if use_zoom and (region_name in REGIONAL_DATA_PATHS) and os.path.exists(REGIONAL_DATA_PATHS[region_name]):
        return REGIONAL_DATA_PATHS[region_name]
    return GLOBAL_DATA_PATH

DATA_PATH = pick_dataset_path(zoom_enabled, zoom_region)
try:
    ds = load_metocean(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found: {DATA_PATH}\n→ Place the file at this path or update REGIONAL_DATA_PATHS.")
    st.stop()

for k in ["prob","hs_edges","tp_edges","lat3_edges","lon3_edges"]:
    if k not in ds:
        st.error(f"Dataset missing required variable: {k}")
        st.stop()

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

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
prob = normalize_pdf(prob)  # hs_bin, tp_bin, lat3_bin, lon3_bin

# -----------------------------
# Per‑Tp Hs limit: CSV + Graph
# -----------------------------
def init_per_tp_limits(default_val: float, tp_centers: np.ndarray):
    key = "hs_per_tp_limits"
    if (key not in st.session_state) or (len(st.session_state[key]) != len(tp_centers)):
        st.session_state[key] = [default_val] * len(tp_centers)
    return key

if threshold_mode == "Hs limit per Tp (CSV + graph)":
    limits_key = init_per_tp_limits(Hcrit, tp_c)
    st.subheader("Hs limit per Tp — curve")
    up = st.file_uploader("Import CSV (columns: 'Tp (s)', 'Hs_limit (m)')", type=["csv"], key="hs_csv_upload")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except UnicodeDecodeError:
            df_in = pd.read_csv(up, encoding="latin-1")
        def find_col(candidates, cols):
            low = [c.lower().strip() for c in cols]
            for cand in candidates:
                if cand.lower().strip() in low:
                    return cols[low.index(cand.lower().strip())]
            return None
        tp_col = find_col(["tp (s)", "tp", "tp_s"], df_in.columns)
        hs_col = find_col(["hs_limit (m)", "hs limit (m)", "hs_limit", "hs (m)", "hs"], df_in.columns)
        if (tp_col is None) or (hs_col is None):
            st.error("CSV must contain columns 'Tp (s)' and 'Hs_limit (m)'.")
        else:
            tp_in = pd.to_numeric(df_in[tp_col], errors="coerce").astype(float).values
            hs_in = pd.to_numeric(df_in[hs_col], errors="coerce").astype(float).values
            mask = np.isfinite(tp_in) & np.isfinite(hs_in)
            tp_in, hs_in = tp_in[mask], hs_in[mask]
            if tp_in.size >= 2:
                order = np.argsort(tp_in)
                tp_in, hs_in = tp_in[order], hs_in[order]
                hs_interp = np.interp(tp_c, tp_in, hs_in, left=hs_in[0], right=hs_in[-1])
                hs_interp = np.clip(np.round(hs_interp, 1), 0.0, 15.0)
                st.session_state[limits_key] = hs_interp.tolist()
                st.success(f"Imported {tp_in.size} rows → mapped to {len(tp_c)} Tp bins.")
    hs_limit_curve = np.array(st.session_state[limits_key], dtype=float) if "hs_per_tp_limits" in st.session_state else None
    if hs_limit_curve is not None:
        hs_limit_curve = np.clip(np.round(hs_limit_curve, 1), 0.0, 15.0)
        st.session_state[limits_key] = hs_limit_curve.tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tp_c, y=hs_limit_curve, mode="lines+markers",
                                 line=dict(color="#1f77b4", width=2), marker=dict(size=6, color="#1f77b4")))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="Tp (s)", yaxis_title="Hs limit (m)",
                          template="plotly_white",
                          xaxis=dict(tickmode="linear", dtick=1),
                          yaxis=dict(range=[0, max(3.0, float(np.nanmax(hs_limit_curve)) + 0.5)], dtick=0.5),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    hs_limit_curve = None

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

if hs_limit_curve is None:
    mask_exceed_1d = xr.DataArray((hs_c > Hcrit).astype(float), dims=["hs_bin"])
    p_exceed = (hs_pdf * mask_exceed_1d).sum(dim="hs_bin") * 100.0
    p_oper = 100.0 - p_exceed
else:
    Hs_1D = xr.DataArray(hs_c, dims=["hs_bin"])
    Tp_limit_1D = xr.DataArray(hs_limit_curve, dims=["tp_bin"])
    Hs2D = Hs_1D.broadcast_like(prob)
    HsLim2D = Tp_limit_1D.broadcast_like(prob)
    mask_exceed_2d = (Hs2D > HsLim2D).astype(float)
    mask_oper_2d  = (Hs2D <= HsLim2D).astype(float)
    p_exceed = (prob * mask_exceed_2d).sum(dim=("hs_bin","tp_bin")) * 100.0
    p_oper  = (prob * mask_oper_2d ).sum(dim=("hs_bin","tp_bin")) * 100.0

# -----------------------------
# Select final field
# -----------------------------
if stat == "Mean Hs (m)":
    field = mean_hs; label = "Mean Hs (m)" + title_suffix
elif stat == "Mean Tp (s)":
    field = mean_tp; label = "Mean Tp (s)" + title_suffix
elif stat == "Hs P50 (m)":
    field = hs_p50; label = "Hs P50 (m)" + title_suffix
elif stat == "Hs P90 (m)":
    field = hs_p90; label = "Hs P90 (m)" + title_suffix
elif stat == "Hs P95 (m)":
    field = hs_p95; label = "Hs P95 (m)" + title_suffix
elif stat.startswith("P(Hs"):
    field = p_exceed; label = (f"P(Hs > {Hcrit:.1f} m) (%)" if hs_limit_curve is None
             else "P(Hs > Hs_limit(Tp)) (%)") + title_suffix
else:
    field = p_oper; label = (f"Operability (% time Hs ≤ {Hcrit:.1f} m)" if hs_limit_curve is None
             else "Operability (% time Hs ≤ Hs_limit(Tp))") + title_suffix

# -----------------------------
# Prepare 2D field & coords (sorted)
# -----------------------------
field2d = field.transpose("lat3_bin","lon3_bin").values
field2d, latp, lonp = to_sorted_lon_lat(field2d, lat_c_unsorted, lon_edges)

# -----------------------------
# Projections
# -----------------------------
def get_zoom_projection(name: str):
    if name.startswith("PlateCarree"): return ccrs.PlateCarree()
    if name == "Mercator": return ccrs.Mercator(central_longitude=10, min_latitude=10, max_latitude=82)
    if name == "Lambert Conformal":
        return ccrs.LambertConformal(central_longitude=10, central_latitude=50, standard_parallels=(30, 60))
    return ccrs.PlateCarree()

# -----------------------------
# Levels & scaling (robust)
# -----------------------------
def safe_minmax(a):
    vmin = np.nanmin(a); vmax = np.nanmax(a)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0, 1.0
    return float(vmin), float(vmax)

def prep_levels(arr, label, prefer_ticks_from=None, zoom=False):
    base = prefer_ticks_from if prefer_ticks_from is not None else arr
    vmin, vmax = safe_minmax(base)
    if "P(Hs" in label or "Operability" in label:
        return pct_shading(), pct_ticks(), pct_ticks()
    if label.startswith("Mean Tp"):
        if zoom:
            contours = np.arange(math.floor(vmin/0.5)*0.5, math.ceil(vmax/0.5)*0.5 + 1e-9, 0.5)
            filled = tp_shading(base); ticks = contours; return filled, contours, ticks
        else:
            ticks = tp_ticks(1.0, vmin, vmax); return tp_shading(base), ticks, ticks
    if is_hs_quantity(label):
        if zoom:
            vmin0 = min(0.0, vmin)
            filled = np.arange(math.floor(vmin0/0.1)*0.1, math.ceil(vmax/0.1)*0.1 + 1e-9, 0.1)
            contours = np.arange(math.floor(vmin0/0.2)*0.2, math.ceil(vmax/0.2)*0.2 + 1e-9, 0.2)
            ticks = contours; return filled, contours, ticks
        else:
            ticks = hs_ticks(0.5, vmin, vmax); return hs_shading(base), ticks, ticks
    lev = auto_levels(base, levels_generic); return lev, lev, None

is_percent_metric = ("P(Hs" in label) or ("Operability" in label)
hi_global = 100.0 if is_percent_metric else np.nanpercentile(field2d, clip_pct_robust)

preset_extent = ZOOM_EXTENTS.get(zoom_region, None) if zoom_enabled else None
final_extent = None
if zoom_enabled:
    if auto_fit:
        final_extent = auto_fit_extent(lonp, latp, field2d, base_extent=preset_extent, pad_deg=pad_deg)
        if final_extent is None and preset_extent is not None:
            final_extent = preset_extent
    else:
        final_extent = preset_extent

if zoom_enabled and not is_percent_metric:
    use_extent = final_extent if final_extent is not None else preset_extent
    region_array = region_slice(field2d, lonp, latp, use_extent) if use_extent is not None else field2d
    cnt_fin, _, _ = finite_stats(region_array)
    if cnt_fin > 0:
        hi_zoom = np.nanpercentile(region_array, clip_pct_robust)
        if not np.isfinite(hi_zoom): hi_zoom = np.nanmax(region_array)
        hi_use = hi_zoom; ticks_base = np.clip(region_array, None, hi_zoom)
    else:
        hi_use = hi_global; ticks_base = np.clip(field2d, None, hi_global)
else:
    hi_use = hi_global; ticks_base = np.clip(field2d, None, hi_global)

arr_plot = np.clip(field2d, None, hi_use)
filled_levels, contour_levels, cbar_ticks = prep_levels(arr_plot, label, prefer_ticks_from=ticks_base, zoom=zoom_enabled)
cmap_use = base_cmap + "_r" if "Operability" in label else base_cmap

# -----------------------------
# Plot
# -----------------------------
def plot_map(lon_c, lat_c, arr2d, title, filled, contours, cmap, ticks,
             use_zoom: bool, zoom_proj, region_name: str, extent=None, renderer="Imshow (force raster)"):
    ax_proj = zoom_proj if use_zoom else ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 6), dpi=(200 if use_zoom else 150))
    ax = plt.axes(projection=ax_proj)

    # Prepare colormap with visible "under"
    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    try: cmap_obj.set_under(cmap_obj(0.0))
    except Exception: pass

    A = np.ma.masked_invalid(arr2d)

    mappable = None
    if renderer.startswith("Imshow"):
        # Force raster in one go with explicit extent
        extent_img = [float(lon_c.min()), float(lon_c.max()), float(lat_c.min()), float(lat_c.max())]
        im = ax.imshow(
            A, origin="lower",
            extent=extent_img,  # [x0, x1, y0, y1]
            cmap=cmap_obj,
            vmin=filled[0] if len(filled) else None,
            vmax=filled[-1] if len(filled) else None,
            transform=ccrs.PlateCarree(),
            zorder=1,
            interpolation="nearest"
        )
        mappable = im
    elif renderer.startswith("Pcolor"):
        norm = mcolors.BoundaryNorm(filled, ncolors=cmap_obj.N, clip=False)
        Lon2D, Lat2D = np.meshgrid(lon_c, lat_c)
        pm = ax.pcolormesh(Lon2D, Lat2D, A, cmap=cmap_obj, norm=norm,
                           transform=ccrs.PlateCarree(), zorder=1, shading="auto")
        mappable = pm
    else:
        cf = ax.contourf(lon_c, lat_c, A, levels=filled, cmap=cmap_obj,
                         extend="both", transform=ccrs.PlateCarree(), zorder=1)
        mappable = cf
        # Optional contours
        try:
            cs = ax.contour(lon_c, lat_c, A, levels=contours, colors="black",
                            linewidths=0.45 if use_zoom else 0.4, transform=ccrs.PlateCarree(), zorder=2)
            ax.figure.canvas.draw()
            ax.clabel(cs, fontsize=6, inline=True, inline_spacing=(1 if use_zoom else 6),
                      fmt="%g", manual=False, rightside_up=True)
        except Exception:
            pass

    # Features
    feature_scale = "10m" if use_zoom else "110m"
    ax.add_feature(cfeature.LAND.with_scale(feature_scale),
                   facecolor="lightgray", edgecolor="none", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale(feature_scale),
                   linewidth=0.7 if use_zoom else 0.4, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale(feature_scale),
                   linewidth=0.3 if use_zoom else 0.2, zorder=12)

    # Extent
    if use_zoom and extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    elif use_zoom and (region_name in ZOOM_EXTENTS):
        ax.set_extent(ZOOM_EXTENTS[region_name], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    # Grid points — never hidden
    if show_grid_points:
        Lon2D, Lat2D = np.meshgrid(lon_c, lat_c)
        ax.plot(Lon2D.ravel(), Lat2D.ravel(), linestyle="None",
                marker="o", markersize=2.2, markeredgewidth=0.0,
                color="#ff00aa", alpha=0.95, transform=ccrs.PlateCarree(), zorder=99)

    # POIs only on North Sea
    if use_zoom and region_name == "North Sea":
        lons = [p["lon"] for p in POIS]; lats = [p["lat"] for p in POIS]
        ax.scatter(lons, lats, s=28, c="black", marker="o", transform=ccrs.PlateCarree(), zorder=98)
        halo = [patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
        offsets = {1:(0.12,0.10), 2:(0.12,0.10), 3:(0.14,0.10), 4:(0.14,0.12), 5:(0.14,0.12),
                   6:(0.12,0.12), 7:(0.12,0.12), 8:(0.12,0.12), 9:(0.12,0.12), 10:(0.14,0.12)}
        for p in POIS:
            dx, dy = offsets.get(p["nr"], (0.12, 0.12))
            ax.text(p["lon"] + dx, p["lat"] + dy, f'{p["nr"]} {p["name"]}',
                    transform=ccrs.PlateCarree(), fontsize=7, color="black", zorder=99, path_effects=halo)

    # Colorbar
    cb = plt.colorbar(mappable, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title); cb.ax.tick_params(labelsize=8)
    ax.set_title(title)

    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

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
    use_zoom=zoom_enabled,
    zoom_proj=get_zoom_projection(zoom_proj_name),
    region_name=zoom_region,
    extent=final_extent,
    renderer=filled_renderer
)

# -----------------------------
# Table (reference only)
# -----------------------------
if threshold_mode == "Hs limit per Tp (CSV + graph)":
    with st.expander("Show Hs limit per Tp (table)", expanded=False):
        df_view = pd.DataFrame({
            "Tp (s)": tp_c,
            "Hs_limit (m)": st.session_state["hs_per_tp_limits"]
        })
    st.dataframe(df_view.style.format({"Tp (s)": "{:.1f}", "Hs_limit (m)": "{:.1f}"}),
                 use_container_width=True)

# -----------------------------
# Debug
# -----------------------------
def inside_extent_counts(lon_c, lat_c, extent):
    if extent is None:
        return lon_c.size, lat_c.size
    lon_min, lon_max, lat_min, lat_max = extent
    j = (lon_c >= lon_min) & (lon_c <= lon_max)
    i = (lat_c >= lat_min) & (lat_c <= lat_max)
    return int(np.count_nonzero(j)), int(np.count_nonzero(i))

if show_debug:
    src = os.path.basename(DATA_PATH)
    use_ext = final_extent if (zoom_enabled and final_extent is not None) else (ZOOM_EXTENTS.get(zoom_region, None) if zoom_enabled else None)
    cnt_all, vmin_dbg, vmax_dbg = finite_stats(field2d)
    j_cnt, i_cnt = inside_extent_counts(lonp, latp, use_ext)
    st.write(
        f"[DEBUG] Source: {src}  |  Zoom: {zoom_region if zoom_enabled else 'Global'}  |  "
        f"Renderer: {filled_renderer}  |  Auto-fit: {bool(auto_fit)}  |  "
        f"Extent used: {use_ext}  |  arr2d shape: {field2d.shape}  |  "
        f"lon range [{float(lonp.min()):.2f}, {float(lonp.max()):.2f}] (monotonic: {np.all(np.diff(lonp)>=0)})  |  "
        f"lat range [{float(latp.min()):.2f}, {float(latp.max()):.2f}] (monotonic: {np.all(np.diff(latp)>=0)})  |  "
        f"finite(all): {cnt_all} (min={vmin_dbg}, max={vmax_dbg})  |  "
        f"grid cols inside extent: {j_cnt}  |  grid rows inside extent: {i_cnt}"
    )