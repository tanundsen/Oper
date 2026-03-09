# 01_Metocean.py — Metocean Explorer (Plotly Version)
# -------------------------------------------------------------------
# • Fully replaces Cartopy with Plotly (no downloads, cloud-safe)
# • Global mode + North Sea zoom toggle
# • Same statistics: Mean Hs, Mean Tp, P(Hs>Hcrit), percentiles, etc.
# • Uses Plotly's built‑in coastlines (high detail, works everywhere)
# • Extended zoom region: lon -13→35, lat 48→72
# -------------------------------------------------------------------

import math
import numpy as np
import xarray as xr
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ---------------------------------------
# Paths
# ---------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_monthclim.nc")

# ---------------------------------------
# Streamlit Page
# ---------------------------------------
st.set_page_config(layout="wide")
st.header("🌍 Global wave statistics (Plotly version)")

# ---------------------------------------
# Cache dataset
# ---------------------------------------
@st.cache_resource
def load_metocean(path: str) -> xr.Dataset:
    return xr.open_dataset(path)

# ---------------------------------------
# Helpers
# ---------------------------------------
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

# ---------------------------------------
# Sidebar controls
# ---------------------------------------
with st.sidebar:
    st.subheader("Aggregation")
    agg = st.radio("Use:", ["By month","Annual"], horizontal=True)

    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_index = st.selectbox("Month", list(range(1,13)), 
                               format_func=lambda i: months[i-1])

    st.subheader("Statistic")
    Hcrit = st.number_input("Hs threshold (m)", 0.1, 15.0, 2.5)
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

    st.subheader("View")
    zoom_ns = st.checkbox("North Sea zoom (48–72N, -13–35E)", value=False)

    st.subheader("Debug")
    show_debug = st.checkbox("Show debug info", False)

# ---------------------------------------
# Load dataset
# ---------------------------------------
ds = load_metocean(DATA_PATH)

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values

# Unit correction
units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges/100.0

hs_c = bin_centers(hs_edges)
tp_c = bin_centers(tp_edges)

lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values
lat_c = bin_centers(lat_edges)
lon_c = bin_centers(lon_edges)

# ---------------------------------------
# Select probability field
# ---------------------------------------
if agg == "By month":
    prob = ds["prob"].sel(month=month_index)
    title_suffix = f" — {months[month_index-1]}"
else:
    prob = ds["prob"].sum(dim="month")
    title_suffix = " — Annual"

prob = normalize_pdf(prob)

# ---------------------------------------
# Compute statistics
# ---------------------------------------
hs_w = xr.DataArray(hs_c, dims=["hs_bin"])
tp_w = xr.DataArray(tp_c, dims=["tp_bin"])

mean_hs = (prob * hs_w).sum(dim=("hs_bin","tp_bin"))
mean_tp = (prob * tp_w).sum(dim=("hs_bin","tp_bin"))

hs_pdf = prob.sum(dim="tp_bin")
hs_cdf = hs_pdf.cumsum(dim="hs_bin")

hs_p50 = percentile_from_cdf(hs_cdf, hs_c, 0.50)
hs_p90 = percentile_from_cdf(hs_cdf, hs_c, 0.90)
hs_p95 = percentile_from_cdf(hs_cdf, hs_c, 0.95)

p_exceed = (hs_pdf * (hs_c > Hcrit)).sum(dim="hs_bin")
p_below = 1 - p_exceed

# ---------------------------------------
# Select the correct field
# ---------------------------------------
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
    field = 100 * p_exceed
else:
    field = 100 * p_below

label = f"{stat}{title_suffix}"

# ---------------------------------------
# Construct DataFrame for Plotly
# ---------------------------------------
arr2d = field.transpose("lat3_bin","lon3_bin").values

df = pd.DataFrame(arr2d, index=lat_c, columns=lon_c)
df = df.reset_index().melt(id_vars="index")
df.columns = ["lat", "lon", "value"]

# ---------------------------------------
# Make the Plotly figure
# ---------------------------------------
fig = px.density_mapbox(
    df,
    lat="lat",
    lon="lon",
    z="value",
    radius=1,  # no smoothing, ERA5 is coarse
    center={"lat": 60, "lon": 0},
    zoom=1,
    color_continuous_scale="Turbo",
    height=650,
    mapbox_style="carto-positron"
)

# Apply zoom extent
if zoom_ns:
    fig.update_layout(
        mapbox=dict(
            center={"lat": 60, "lon": 10},
            zoom=3.4  # tuned for your region
        )
    )

fig.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar_title=label
)

st.plotly_chart(fig, use_container_width=True)

# Debug
if show_debug:
    st.write("Field stats:", float(np.nanmin(arr2d)), float(np.nanmax(arr2d)))
