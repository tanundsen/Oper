# 01_Metocean.py — Metocean Explorer (3° grid)
# ----------------------------------------------------------
# Features:
# • Global map: PlateCarree (fixed)
# • Zoomed map: PlateCarree by default, selector for Mercator/Lambert
# • Colorbar/ticks adapt to zoomed region for Hs/Tp metrics
# • Percent metrics stay 0–100 %
# • 110m features globally (speed), 10m when zoomed (detail)
# • North Sea Points of Interest (1–10) visible only when zoomed
# ----------------------------------------------------------

import math
import numpy as np
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from matplotlib import patheffects

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GLOBAL 3° dataset (unchanged)
GLOBAL_DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_monthclim.nc")

# REGIONAL 0.5° dataset (new)
REGIONAL_DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc")


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

def is_hs_quantity(label):
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
    c_lo = cdf.isel(hs_bin=idx_lo)
    c_hi = cdf.isel(hs_bin=idx_hi)
    cen = xr.DataArray(centers, dims=["hs_bin"])
    h_lo = cen.isel(hs_bin=idx_lo)
    h_hi = cen.isel(hs_bin=idx_hi)
    denom = xr.where((c_hi - c_lo) > 0, c_hi - c_lo, 1)
    w = (q - c_lo)/denom
    return h_lo + w*(h_hi - h_lo)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Data")
    st.caption("Using dataset: metocean_monthclim.nc")

    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month","Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    chosen_label = st.selectbox("Month", months, index=4)
    label_to_idx = dict(zip(months, month_vals))

    st.subheader("Statistic")
    Hcrit = st.number_input("Hs threshold (m)", 0.1, 15.0, 2.5, step=0.1)
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
ZOOM_EXTENT = [-13, 35, 52, 76]   # up to 72°N; west ~13°W; east 35°E
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
# Load dataset
# -----------------------------

# Hybrid data loading:
# - Global → use 3° file
# - Zoomed → use 0.5° file
if zoom_ns:
    ds = load_metocean(REGIONAL_DATA_PATH)
else:
    ds = load_metocean(GLOBAL_DATA_PATH)

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
prob = normalize_pdf(prob)

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

mask = xr.DataArray((hs_c > Hcrit).astype(float), dims=["hs_bin"])
p_exceed = (hs_pdf * mask).sum(dim="hs_bin")
p_below = 1 - p_exceed

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
    field = 100 * p_exceed
    label = f"P(Hs > {Hcrit:.1f} m) (%)" + title_suffix
else:
    field = 100 * p_below
    label = f"Operability (% time Hs ≤ {Hcrit:.1f} m)" + title_suffix

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

def prep_levels(arr, label, prefer_ticks_from=None, zoom=False):
    base = prefer_ticks_from if prefer_ticks_from is not None else arr

    # Percent metrics unchanged
    if "P(Hs" in label or "Operability" in label:
        return pct_shading(), pct_ticks(), pct_ticks()

    # Tp metrics
    if label.startswith("Mean Tp"):
        if zoom:
            # Zoomed contour spacing 0.5 s
            levels = np.arange(np.nanmin(base), np.nanmax(base) + 0.5, 0.5)
            ticks = levels
            return levels, ticks, ticks
        else:
            ticks = tp_ticks(1.0, np.nanmin(base), np.nanmax(base))
            return tp_shading(base), ticks, ticks

    # Hs metrics
    if is_hs_quantity(label):
        if zoom:
            # Zoomed contour spacing 0.2 m
            levels = np.arange(np.nanmin(base), np.nanmax(base) + 0.2, 0.1)
            ticks = levels
            return levels, ticks, ticks
        else:
            ticks = hs_ticks(0.5, np.nanmin(base), np.nanmax(base))
            return hs_shading(base), ticks, ticks

    # fallback
    lev = auto_levels(base, levels_generic)
    return lev, lev, None


is_percent_metric = ("P(Hs" in label) or ("Operability" in label)

# Robust caps
if is_percent_metric:
    hi_global = 100.0
else:
    hi_global = np.nanpercentile(field2d, clip_pct_robust)

# When zoomed: adapt scale to zoomed region (except % metrics)
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
    """
    Draws numbered markers and labels 'Nr Name' for POIs on the zoomed map.
    - ax: matplotlib/cartopy axes
    - pois: list of dicts with keys: name, nr, lat, lon
    """
    # Base marker style
    lons = [p["lon"] for p in pois]
    lats = [p["lat"] for p in pois]
    ax.scatter(
        lons, lats, s=28, c="black", marker="o",
        transform=ccrs.PlateCarree(), zorder=20
    )

    # Slight per-point offsets (deg) to reduce overlaps in dense clusters
    # Tune here if any labels collide on your screen
    offsets = {
        # nr: (dx, dy)
         1: ( 0.12,  0.10),  # Ekofisk
         2: ( 0.12,  0.10),  # Ula
         3: ( 0.14,  0.10),  # Sleipner
         4: ( 0.14,  0.12),  # Alvheim
         5: ( 0.14,  0.12),  # Oseberg
         6: ( 0.12,  0.12),  # Knarr
         7: ( 0.12,  0.12),  # Ormen Lange
         8: ( 0.12,  0.12),  # Skarv
         9: ( 0.12,  0.12),  # Aasta Hansteen
        10: ( 0.14,  0.12),  # Johan Castberg
    }

    # Text style: small font with white halo for readability
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
    for p in pois:
        dx, dy = offsets.get(p["nr"], (0.12, 0.12))
        ax.text(
            p["lon"] + dx, p["lat"] + dy,
            f'{p["nr"]} {p["name"]}',
            transform=ccrs.PlateCarree(),
            fontsize=7, color="black", zorder=21,
            path_effects=halo
        )
# -----------------------------
# Plot function
# -----------------------------
def plot_map(lon_c, lat_c, arr2d, title, filled, contours, cmap, ticks,
             use_zoom: bool, zoom_proj):
    # Axes projection: PlateCarree (global) or selected projection (zoom)
    ax_proj = (zoom_proj if use_zoom else ccrs.PlateCarree())

    fig = plt.figure(figsize=(15, 6), dpi=150)
    ax = plt.axes(projection=ax_proj)

    cf = ax.contourf(
        lon_c, lat_c, arr2d,
        levels=filled,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),  # data is lon/lat
        zorder=1
    )
    try:
        cs = ax.contour(
            lon_c, lat_c, arr2d,
            levels=contours,
            colors="black",
            linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=2
        )
        
        ax.clabel(
            cs,
            fontsize=6,
            inline=True,
            inline_spacing=1,   # << closer contour labels
            fmt="%g"
        )

    except Exception:
        pass

    # Feature detail: 10m when zoomed, 110m when global
    feature_scale = "10m" if use_zoom else "110m"
    ax.add_feature(cfeature.LAND.with_scale(feature_scale), facecolor="lightgray", edgecolor="none", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale(feature_scale), linewidth=0.7 if use_zoom else 0.4, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale(feature_scale), linewidth=0.3 if use_zoom else 0.2, zorder=12)

    # Extent
    if use_zoom:
        ax.set_extent(ZOOM_EXTENT, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    # Points of interest: only on zoomed map
    if use_zoom:
        draw_pois(ax, POIS)

    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

    # Compact legend (Nr → Name) below the figure when zoomed
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
    st.write(
        "Color cap (hi_use):", float(hi_use),
        "| Zoomed:", bool(zoom_ns),
        "| Zoom projection:", zoom_proj_name
    )