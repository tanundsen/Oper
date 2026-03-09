# 01_Metocean.py — Metocean Explorer (3° grid)
# ----------------------------------------------------------
# Updated version:
#  • Added "North Sea zoom" toggle in sidebar (Option A)
#  • When enabled: map extent = lon -10→10, lat 48→65
#  • Uses high‑detail coastline ("10m" scale)
#  • Visual-only zoom (data remains global)
#  • Everything else unchanged: PlateCarree, turbo colormap,
#    contours, ticks, clipping, edges, PDF normalization.
# ----------------------------------------------------------

import math
import numpy as np
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_monthclim.nc")

# -----------------------------------------------------------
# Page setup
# -----------------------------------------------------------
st.set_page_config(layout="wide")
st.header("🌍 Global wave statistics")

# -----------------------------------------------------------
# Helpers & caching
# -----------------------------------------------------------
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
    vmin = np.nanmin(field)
    vmax = np.nanmax(field)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0, 1, n)
    return np.linspace(vmin, vmax, n)

def tp_ticks(step=1.0, vmin=None, vmax=None):
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 20
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + step*0.5, step)

def tp_shading(field, n=80):
    vmin = np.nanmin(field)
    vmax = np.nanmax(field)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0, 20, n)
    return np.linspace(vmin, vmax, n)

def pct_ticks():
    return np.arange(0, 101, 10)

def pct_shading(n=61):
    return np.linspace(0, 100, n)

def auto_levels(arr, n=50):
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmin == vmax or not np.isfinite(vmin):
        return np.linspace(0,1,n)
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

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
with st.sidebar:
    st.subheader("Data")
    st.caption("Using dataset: metocean_monthclim.nc")

    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month","Annual"], horizontal=True, index = 1)
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vals = np.arange(1,13)
    chosen_label = st.selectbox("Month", months, index=11)
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

    # NEW: North Sea Zoom Toggle
    st.subheader("View")
    zoom_ns = st.checkbox("North Sea zoom", value=False)

    st.subheader("Debug")
    show_debug = st.checkbox("Show debug", False)

# -----------------------------------------------------------
# Fixed settings
# -----------------------------------------------------------
projection = "PlateCarree"
base_cmap = "turbo"
levels_generic = 50
clip_pct = 99.6

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
ds = load_metocean(DATA_PATH)

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

# -----------------------------------------------------------
# Select probability field
# -----------------------------------------------------------
if agg == "By month":
    prob = ds["prob"].sel(month=label_to_idx[chosen_label])
    title_suffix = f" — {chosen_label}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"

_tot_before = prob.sum(dim=("hs_bin","tp_bin"))
prob = normalize_pdf(prob)

# -----------------------------------------------------------
# Compute statistics
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Select final field
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
elif stat.startswith("P(Hs"):
    field = 100*p_exceed
    label = f"P(Hs > {Hcrit:.1f} m) (%)" + title_suffix
else:
    field = 100*p_below
    label = f"Operability (% time Hs ≤ {Hcrit:.1f} m)" + title_suffix

# -----------------------------------------------------------
# Prepare 2D field
# -----------------------------------------------------------
field2d = field.transpose("lat3_bin","lon3_bin").values
field2d, latp, lonp, flip_lat, lon_sort_idx, lon_inv = to_sorted_lon_lat(
    field2d, lat_c_unsorted, lon_edges
)

# Clipping
if ("P(Hs" in label) or ("Operability" in label):
    clip_use = 100
else:
    clip_use = clip_pct

hi = np.nanpercentile(field2d, clip_use)
field2d = np.clip(field2d, None, hi)

# Levels
def prep_levels(arr, label):
    if "P(Hs" in label or "Operability" in label:
        return pct_shading(), pct_ticks(), pct_ticks()
    elif label.startswith("Mean Tp"):
        ticks = tp_ticks(1.0, np.nanmin(arr), np.nanmax(arr))
        return tp_shading(arr), ticks, ticks
    elif is_hs_quantity(label):
        ticks = hs_ticks(0.5, np.nanmin(arr), np.nanmax(arr))
        return hs_shading(arr), ticks, ticks
    else:
        lev = auto_levels(arr, levels_generic)
        return lev, lev, None

filled_levels, contour_levels, cbar_ticks = prep_levels(field2d, label)

cmap_use = base_cmap + "_r" if "Operability" in label else base_cmap

# -----------------------------------------------------------
# Plot function (updated with 10m coastline + zoom toggle)
# -----------------------------------------------------------
def plot_global_map(lon_c, lat_c, arr2d, title, filled, contours, cmap, ticks):
    fig = plt.figure(figsize=(15,6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())

    cf = ax.contourf(
        lon_c, lat_c, arr2d,
        levels=filled,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
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
        ax.clabel(cs, fontsize=6, inline=True, fmt="%g")
    except:
        pass

    # High‑detail coastline
    ax.add_feature(
        cfeature.LAND.with_scale("10m"),
        facecolor="lightgray",
        edgecolor="none",
        zorder=10
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        linewidth=0.8,
        zorder=11
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        linewidth=0.3,
        zorder=12
    )

    # Zoom logic
    if zoom_ns:
        ax.set_extent([-11, 35, 50, 74], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    cb = plt.colorbar(
        cf, ax=ax,
        shrink=0.75, aspect=30,
        pad=0.01,
        ticks=ticks
    )
    cb.set_label(title)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(title)
    plt.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.06)
    st.pyplot(fig, use_container_width=True)

# -----------------------------------------------------------
# Render final map
# -----------------------------------------------------------
plot_global_map(
    lonp, latp, field2d,
    label,
    filled_levels,
    contour_levels,
    cmap_use,
    cbar_ticks
)

# -----------------------------------------------------------
# Debug
# -----------------------------------------------------------
if show_debug:
    st.write("Totals BEFORE normalization:",
             float(_tot_before.min()),
             float(_tot_before.max()))
    st.write("Mean Hs (global):",
             float(np.nanmin(mean_hs)),
             float(np.nanmax(mean_hs)))