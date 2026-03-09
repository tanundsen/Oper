# 01_Metocean.py — Metocean Explorer (NaN-safe, Cloud-ready)
# ----------------------------------------------------------
# Key differences vs your previous version:
# - NaN-safe annual aggregation + normalization
# - Unified dataset path (relative to this file)
# - Optional Cartopy Natural Earth data_dir override (if vendored)
# - Guards for "no finite data" cases to avoid contour crashes
# - PlateCarree + turbo, 0.5 m Hs ticks, same land masking

from __future__ import annotations
import os
import math
from pathlib import Path

import numpy as np
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy

# -----------------------------------------------------------
# Configuration / paths
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "metocean_monthclim.nc"  # unified across pages

# If you've vendored Natural Earth data, point Cartopy to it:
# e.g., repo_root/data/natural_earth/...
NE_ENV = os.environ.get("NATURAL_EARTH_DIR", "")
if NE_ENV:
    cartopy.config["data_dir"] = NE_ENV  # optional hardening for Streamlit Cloud

# -----------------------------------------------------------
# Streamlit page setup
# -----------------------------------------------------------
st.set_page_config(page_title="🌍 Global wave statistics", layout="wide")
st.header("🌍 Global wave statistics")

# -----------------------------------------------------------
# Helpers & caching
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_metocean(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)

def bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])

def unwrap_lon_centers_from_edges(lon_edges: np.ndarray) -> np.ndarray:
    lon_c = bin_centers(lon_edges)
    if np.nanmax(lon_edges) > 180:  # wrap to [-180, 180)
        lon_c = ((lon_c + 180) % 360) - 180
    return lon_c

def to_sorted_lon_lat(field2d: np.ndarray,
                      lat_centers: np.ndarray,
                      lon_edges: np.ndarray):
    """Flip latitude if descending; wrap/sort longitudes to [-180, 180)."""
    flip_lat = False
    lat_c = lat_centers.copy()
    arr = field2d

    if lat_c[0] > lat_c[-1]:       # descending
        arr = arr[::-1, :]
        lat_c = lat_c[::-1]
        flip_lat = True

    lon_uns = unwrap_lon_centers_from_edges(lon_edges)
    idx = np.argsort(lon_uns)
    lon_sorted = lon_uns[idx]
    arr_sorted = arr[:, idx]
    inv = np.argsort(idx)          # for optional back-mapping
    return arr_sorted, lat_c, lon_sorted, flip_lat, idx, inv

def is_hs_quantity(label: str) -> bool:
    if "%" in label:
        return False
    return "hs" in label.lower()

def hs_ticks(step=0.5, vmin=0.0, vmax=10.0):
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + step / 2, step)

def hs_shading(field: np.ndarray, n=60):
    vmin = np.nanmin(field)
    vmax = np.nanmax(field)
    if (not np.isfinite(vmin)) or vmin == vmax:
        return np.linspace(0, 1, n)
    return np.linspace(vmin, vmax, n)

def tp_ticks(step=1.0, vmin=None, vmax=None):
    if vmin is None: vmin = 0.0
    if vmax is None: vmax = 20.0
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + step * 0.5, step)

def tp_shading(field: np.ndarray, n=80):
    vmin = np.nanmin(field)
    vmax = np.nanmax(field)
    if (not np.isfinite(vmin)) or vmin == vmax:
        return np.linspace(0, 20, n)
    return np.linspace(vmin, vmax, n)

def pct_ticks():
    return np.arange(0, 101, 10)

def pct_shading(n=61):
    return np.linspace(0, 100, n)

def auto_levels(arr: np.ndarray, n=50):
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if (not np.isfinite(vmin)) or vmin == vmax:
        return np.linspace(0, 1, n)
    return np.linspace(vmin, vmax, n)

def normalize_pdf(prob: xr.DataArray) -> xr.DataArray:
    """
    NaN-safe normalization to a proper PDF over (hs_bin, tp_bin).
    """
    prob = prob.fillna(0)  # neutralize NaNs before reductions
    tot = prob.sum(dim=("hs_bin", "tp_bin"), skipna=True)
    denom = xr.where(tot > 0, tot, 1)
    return prob / denom

def percentile_from_cdf(cdf: xr.DataArray,
                        centers: np.ndarray,
                        q: float) -> xr.DataArray:
    """
    Linear interpolation of percentile along hs_bin dimension.
    cdf: increasing along hs_bin (0..1)
    centers: hs centers (same length as hs_bin)
    q: target quantile in [0, 1]
    """
    idx_hi = (cdf >= q).argmax(dim="hs_bin")
    idx_lo = xr.where(idx_hi > 0, idx_hi - 1, 0)

    c_lo = cdf.isel(hs_bin=idx_lo)
    c_hi = cdf.isel(hs_bin=idx_hi)

    cen = xr.DataArray(centers, dims=["hs_bin"])
    h_lo = cen.isel(hs_bin=idx_lo)
    h_hi = cen.isel(hs_bin=idx_hi)

    denom = xr.where((c_hi - c_lo) > 0, c_hi - c_lo, 1)
    w = (q - c_lo) / denom
    return h_lo + w * (h_hi - h_lo)

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
with st.sidebar:
    st.subheader("Data")
    st.caption(f"Using dataset: {DATA_PATH.name}")

    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month", "Annual"], horizontal=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1, 13)
    chosen_label = st.selectbox("Month", months, index=4)
    label_to_idx = dict(zip(months, month_vals))

    st.subheader("Metric")
    Hcrit = st.number_input("Hs threshold (m)", 0.1, 15.0, 2.5, step=0.1)
    stat = st.selectbox(
        "Statistic:",
        [
            "Mean Hs (m)",
            "Mean Tp (s)",
            "Hs P50 (m)",
            "Hs P90 (m)",
            "Hs P95 (m)",
            "P(Hs > Hcrit) (%)",
            "Operability (% time Hs ≤ Hcrit)",
        ],
    )

    st.subheader("Debug")
    show_debug = st.checkbox("Show debug", False)

# Fixed visual settings
projection = "PlateCarree"
base_cmap = "turbo"
levels_generic = 50
clip_pct_default = 99.6  # non-% maps only

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
try:
    ds = load_metocean(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset at '{DATA_PATH}': {e}")
    st.stop()

for k in ["prob", "hs_edges", "tp_edges", "lat3_edges", "lon3_edges"]:
    if k not in ds:
        st.error(f"Dataset missing variable: {k}")
        st.stop()

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

# Unit fix for Hs edges (cm -> m)
units = str(ds["hs_edges"].attrs.get("units", "")).lower()
if ("cm" in units) or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c = bin_centers(hs_edges)
tp_c = bin_centers(tp_edges)
lat_c_unsorted = bin_centers(lat_edges)

# -----------------------------------------------------------
# Select probability slice (NaN-safe)
# -----------------------------------------------------------
if agg == "By month":
    prob = ds["prob"].sel(month=label_to_idx[chosen_label]).fillna(0).astype("float32")
    title_suffix = f" — {chosen_label}"
else:
    # SUM safely across months, then normalize
    prob = ds["prob"].fillna(0).sum(dim="month", skipna=True).astype("float32")
    title_suffix = " — Annual"

prob = normalize_pdf(prob)

# -----------------------------------------------------------
# Compute statistics
# -----------------------------------------------------------
hs_w = xr.DataArray(hs_c, dims=["hs_bin"])
tp_w = xr.DataArray(tp_c, dims=["tp_bin"])

# NaN-safe reductions
mean_hs = (prob * hs_w).sum(dim=("hs_bin", "tp_bin"), skipna=True)
mean_tp = (prob * tp_w).sum(dim=("hs_bin", "tp_bin"), skipna=True)

hs_pdf = prob.sum(dim="tp_bin", skipna=True)
hs_cdf = hs_pdf.cumsum(dim="hs_bin", skipna=True)

hs_p50 = percentile_from_cdf(hs_cdf, hs_c, 0.50)
hs_p90 = percentile_from_cdf(hs_cdf, hs_c, 0.90)
hs_p95 = percentile_from_cdf(hs_cdf, hs_c, 0.95)

mask_exceed = xr.DataArray((hs_c > Hcrit).astype(float), dims=["hs_bin"])
p_exceed = (hs_pdf * mask_exceed).sum(dim="hs_bin", skipna=True)  # fraction
p_below = 1.0 - p_exceed

# -----------------------------------------------------------
# Select field for plotting
# -----------------------------------------------------------
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
elif stat == "P(Hs > Hcrit) (%)":
    field = 100.0 * p_exceed
    label = f"P(Hs > {Hcrit:.1f} m) (%)" + title_suffix
else:  # Operability
    field = 100.0 * p_below
    label = f"Operability (% time Hs ≤ {Hcrit:.1f} m)" + title_suffix

# -----------------------------------------------------------
# Prepare 2D field for plotting
# -----------------------------------------------------------
field2d = field.transpose("lat3_bin", "lon3_bin").values
field2d, lat_plot, lon_plot, flip_lat, lon_sort_idx, lon_inv = to_sorted_lon_lat(
    field2d, lat_c_unsorted, lon_edges
)

# Clip for non-% maps (keeps bright outliers under control)
if ("P(Hs" in label) or ("Operability" in label)):
    clip_use = 100.0
else:
    clip_use = clip_pct_default

finite_mask = np.isfinite(field2d)
if not finite_mask.any():
    st.warning("No finite data available for this selection.")
    st.stop()

if clip_use < 100.0:
    hi = np.nanpercentile(field2d, clip_use)
    field2d = np.clip(field2d, None, hi)

# -----------------------------------------------------------
# Level/tick selection
# -----------------------------------------------------------
def prep_levels(arr: np.ndarray, label: str):
    if ("P(Hs" in label) or ("Operability" in label)):
        ticks = pct_ticks()
        return pct_shading(), ticks, ticks, base_cmap + "_r"
    elif label.startswith("Mean Tp"):
        ticks = tp_ticks(1.0, np.nanmin(arr), np.nanmax(arr))
        return tp_shading(arr), ticks, ticks, base_cmap
    elif is_hs_quantity(label):
        ticks = hs_ticks(0.5, float(np.nanmin(arr)), float(np.nanmax(arr)))
        return hs_shading(arr), ticks, ticks, base_cmap
    else:
        lev = auto_levels(arr, levels_generic)
        return lev, lev, None, base_cmap

filled_levels, contour_levels, cbar_ticks, cmap_use = prep_levels(field2d, label)

# -----------------------------------------------------------
# Plotting
# -----------------------------------------------------------
def plot_global_map(lon_c, lat_c, arr2d, title, filled, contours, cmap, ticks):
    fig = plt.figure(figsize=(15, 6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())

    cf = ax.contourf(
        lon_c, lat_c, arr2d,
        levels=filled,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    # Contour lines + labels
    try:
        cs = ax.contour(
            lon_c, lat_c, arr2d,
            levels=contours,
            colors="black",
            linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=2
        )
        ax.clabel(cs, fontsize=6, inline=True, fmt="%g")
    except Exception:
        pass

    # Land / coast / borders — same as earlier (avoids white gaps)
    ax.add_feature(cfeature.LAND.with_scale("110m"),
                   facecolor="lightgray", edgecolor="none", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   linewidth=0.8, zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"),
                   linewidth=0.3, zorder=12)

    ax.set_global()

    cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01, ticks=ticks)
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

# Render
plot_global_map(
    lon_plot, lat_plot, field2d,
    label, filled_levels, contour_levels, cmap_use, cbar_ticks
)

# -----------------------------------------------------------
# Debug (optional)
# -----------------------------------------------------------
if show_debug:
    # Show min/max of raw and clipped arrays for sanity checks
    st.write("Array stats (after prep):",
             f"min={float(np.nanmin(field2d)):.4g}, max={float(np.nanmax(field2d)):.4g}")