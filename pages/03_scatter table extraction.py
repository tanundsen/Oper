# 03_Extract_HsTp_Table.py
# -------------------------------------------------------------
# Extract interpolated Hs–Tp at clicked location
# Includes:
#   • PIN placement
#   • longitude wrapping (dataset uses −180..180)
#   • latitude axis correction (dataset lat is descending!)
#   • percent formatting
#   • heatmap + CSV
#   • st.rerun
# -------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------
@st.cache_resource
def load_metocean(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


# -------------------------------------------------------------
# Convert Folium lon (0..360) to dataset lon (-180..180)
# -------------------------------------------------------------
def wrap_lon(lon):
    if lon > 180:
        return lon - 360
    return lon


# -------------------------------------------------------------
# Page setup
# -------------------------------------------------------------
st.set_page_config(page_title="📊 Extract Hs–Tp Table", layout="wide")
st.title("📊 Extract Hs–Tp Table From Map Location")


# -------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------
ds = load_metocean("metocean_monthclim.nc")

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

# Fix Hs units
units = str(ds["hs_edges"].attrs.get("units","")).lower()
if "cm" in units or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

# Compute centers
hs_c = 0.5 * (hs_edges[:-1] + hs_edges[1:])
tp_c = 0.5 * (tp_edges[:-1] + tp_edges[1:])
lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])

# -------------------------------------------------------------
# IMPORTANT FIX: dataset latitude is descending -> must flip
# -------------------------------------------------------------
lat_descending = lat_c[0] > lat_c[-1]
if lat_descending:
    lat_c = lat_c[::-1]


# -------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    mode = st.radio("Aggregation:", ["By month", "Annual"])
    months = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]

    if mode == "By month":
        month = st.selectbox("Month", months)
        month_idx = months.index(month) + 1

    output_type = st.selectbox(
        "Output type",
        ["Normalized PDF (percent)", "Raw probability", "CDF over Hs (percent per Tp)"]
    )


# -------------------------------------------------------------
# Map + pin logic
# -------------------------------------------------------------
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None

st.subheader("📍 Select location")

m = folium.Map(location=[20,0], zoom_start=2)

# Add pin if exists
if st.session_state.selected_point:
    plat, plon = st.session_state.selected_point
    folium.Marker(
        [plat, plon],
        tooltip=f"{plat:.2f}, {plon:.2f}",
        icon=folium.Icon(color="red")
    ).add_to(m)

# Capture click
result = st_folium(m, height=500, width=1000, returned_objects=["last_clicked"])

if result and result.get("last_clicked"):
    lat = result["last_clicked"]["lat"]
    lon = wrap_lon(result["last_clicked"]["lng"])
    st.session_state.selected_point = (lat, lon)
    st.rerun()

# Stop if no click yet
if st.session_state.selected_point is None:
    st.warning("Click on the map to choose a location.")
    st.stop()

lat_sel, lon_sel = st.session_state.selected_point
st.success(f"Selected location: **Lat {lat_sel:.2f}°, Lon {lon_sel:.2f}°**")


# -------------------------------------------------------------
# Nearest grid cell lookup
# -------------------------------------------------------------
def nearest_grid(lat, lon):
    i_lat = np.abs(lat_c - lat).argmin()
    i_lon = np.abs(lon_c - lon).argmin()
    return i_lat, i_lon

ilat, ilon = nearest_grid(lat_sel, lon_sel)


# -------------------------------------------------------------
# Extract probabilities
# -------------------------------------------------------------
if mode == "By month":
    prob = ds["prob"].sel(month=month_idx).isel(lat3_bin=ilat, lon3_bin=ilon)
else:
    prob = ds["prob"].sum("month").isel(lat3_bin=ilat, lon3_bin=ilon)


prob_np = prob.values

# FIX #1: remove NaNs (monthly ERA5 often has NaN in all empty bins)
prob_np = np.nan_to_num(prob_np, nan=0.0)

# FIX #2: if lat orientation flipped earlier, maintain consistency:
if lat_descending:
    prob_np = prob_np[::-1, :]


# -------------------------------------------------------------
# If dataset latitude was descending -> slice must also flip
# -------------------------------------------------------------
if lat_descending:
    prob_np = prob_np[::-1, :]


# -------------------------------------------------------------
# Output transform
# -------------------------------------------------------------
if output_type == "Normalized PDF (percent)":
    s = prob_np.sum()
    table = (prob_np / s * 100.0) if s > 0 else np.zeros_like(prob_np)

elif output_type == "Raw probability":
    table = prob_np

else:  # CDF
    table = np.cumsum(prob_np, axis=0)
    maxv = table[-1, :]
    for j in range(table.shape[1]):
        if maxv[j] > 0:
            table[:, j] = table[:, j] / maxv[j] * 100.0


# -------------------------------------------------------------
# Build DataFrame
# -------------------------------------------------------------
df = pd.DataFrame(table, index=hs_c.round(3), columns=tp_c.round(3))
df.index.name = "Hs (m)"
df.columns.name = "Tp (s)"

if "percent" in output_type:
    df_display = df.map(lambda x: f"{x:.2f} %")
else:
    df_display = df


# -------------------------------------------------------------
# Show table
# -------------------------------------------------------------
st.subheader("📈 Hs–Tp Table")
st.caption(f"Grid cell index: lat3_bin={ilat}, lon3_bin={ilon}")
st.dataframe(df_display, use_container_width=True)


# -------------------------------------------------------------
# Heatmap
# -------------------------------------------------------------
st.subheader("🔵 Heatmap")

fig, ax = plt.subplots(figsize=(8, 5))
pc = ax.pcolormesh(tp_c, hs_c, table, shading="auto", cmap="turbo")
fig.colorbar(pc, ax=ax)
ax.set_xlabel("Tp (s)")
ax.set_ylabel("Hs (m)")
st.pyplot(fig)


# -------------------------------------------------------------
# CSV download
# -------------------------------------------------------------
st.download_button(
    "⬇ Download CSV",
    df.to_csv().encode("utf-8"),
    file_name="Hs_Tp_table.csv",
    mime="text/csv"
)