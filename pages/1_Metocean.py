# 1_Metocean.py — Cartopy + GSHHS (cloud‑safe)
# --------------------------------------------------------------------
# • Global view uses GSHHS 'l' (low) → fast, like 110m
# • North Sea zoom uses GSHHS 'h' (high) → detailed, like 10m
# • No Natural Earth downloads
# • Annual mode safe (no memory spikes)
# • Works on Streamlit Cloud
# --------------------------------------------------------------------

import os
import math
import numpy as np
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_monthclim.nc")

ZOOM_EXTENT = [-13, 35, 48, 72]  # lon_min, lon_max, lat_min, lat_max

# -----------------------------------------------------------
# STREAMLIT PAGE
# -----------------------------------------------------------
st.set_page_config(layout="wide")
st.header("🌍 Global wave statistics (Cartopy + GSHHS)")

# -----------------------------------------------------------
# CACHE DATA
# -----------------------------------------------------------
@st.cache_resource
def load_nc(path):
    return xr.open_dataset(path)

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

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

def hs_ticks(step, vmin, vmax):
    lo = math.floor(vmin/step) * step
    hi = math.ceil(vmax/step) * step
    return np.arange(lo, hi+step/2, step)

def auto_levels(arr, n=60):
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if not np.isfinite(vmin) or vmin == vmax:
        return np.linspace(0,1,n)
    return np.linspace(vmin, vmax, n)

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
with st.sidebar:
    agg = st.radio("Aggregation:", ["By month", "Annual"], horizontal=True)

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    midx = st.selectbox("Month", range(1,13), format_func=lambda x: months[x-1])

    Hcrit = st.number_input("Hs threshold (m)", 0.1, 20.0, 2.5)
    stat = st.selectbox(
        "Statistic:",
        [
            "Mean Hs (m)",
            "Mean Tp (s)",
            "Hs P50 (m)",
            "Hs P90 (m)",
            "Hs P95 (m)",
            "P(Hs > Hcrit) (%)",
            "Operability (% time Hs ≤ Hcrit)"
        ]
    )

    st.subheader("View area")
    zoom_ns = st.checkbox("North Sea zoom (48–72N, -13–35E)", value=False)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
ds = load_nc(DATA_PATH)

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values

# Convert cm → m if needed
units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c = bin_centers(hs_edges)
tp_c = bin_centers(tp_edges)

lat_c = bin_centers(ds["lat3_edges"].values)
lon_c = bin_centers(ds["lon3_edges"].values)

# -----------------------------------------------------------
# SELECT PROB
# -----------------------------------------------------------
if agg == "By month":
    prob = ds["prob"].sel(month=midx)
    title_suffix = f" — {months[midx-1]}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"

prob = normalize_pdf(prob)

# -----------------------------------------------------------
# STATISTICS
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

p_exceed = (hs_pdf * (hs_c > Hcrit)).sum(dim="hs_bin")
p_oper = 1 - p_exceed

# -----------------------------------------------------------
# SELECT FIELD
# -----------------------------------------------------------
if stat == "Mean Hs (m)":
    field = mean_hs
elif stat == "Mean Tp (s)":
    field = mean_tp
elif stat == "Hs P50 (m)":
    field = hs_p50
elif stat == "Hs P90 (m)":
    field = hs_p90
elif stat == "Hs P95 (m)":
    field = hs_p95
elif stat.startswith("P(Hs"):
    field = 100*p_exceed
else:
    field = 100*p_oper

label = stat + title_suffix

# 2D array
arr = field.transpose("lat3_bin","lon3_bin").values

# Clip extremes
if "Hs" in stat and "%" not in stat:
    arr = np.clip(arr, None, np.nanpercentile(arr, 99.6))

# -----------------------------------------------------------
# PLOT (CARTOPY + GSHHS)
# -----------------------------------------------------------
fig = plt.figure(figsize=(15,6), dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())

# Filled contours
levels = auto_levels(arr, 60)
cf = ax.contourf(
    lon_c, lat_c, arr,
    levels=levels,
    cmap="turbo_r" if "Operability" in stat else "turbo",
    transform=ccrs.PlateCarree(),
    extend="both"
)

# Contours
try:
    cs = ax.contour(
        lon_c, lat_c, arr,
        levels=levels[::6],
        colors="black",
        linewidths=0.35,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fontsize=6, inline=True)
except:
    pass

# Coastlines
if zoom_ns:
    coast_scale = "h"   # High detail (10m‑like)
else:
    coast_scale = "l"   # Low detail (110m‑like)

ax.add_feature(
    cfeature.GSHHSFeature(scale=coast_scale, facecolor="lightgray"),
    zorder=10
)
ax.add_feature(
    cfeature.GSHHSFeature(scale=coast_scale, edgecolor="black", facecolor="none"),
    zorder=11
)
ax.add_feature(
    cfeature.BORDERS.with_scale("110m"),
    linewidth=0.4, zorder=12
)

# Zoom or global
if zoom_ns:
    ax.set_extent(ZOOM_EXTENT, crs=ccrs.PlateCarree())
else:
    ax.set_global()

# Colorbar
cb = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=30, pad=0.01)
cb.set_label(label)

ax.set_title(label)
st.pyplot(fig, width="stretch")