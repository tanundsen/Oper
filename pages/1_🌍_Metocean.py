
# FULL RESTRUCTURED METOCEAN PAGE WITH MEDITERRANEAN + NORTH SEA ZOOM
# ---------------------------------------------------------------------------
# This file is a clean, reorganized version of your metocean Streamlit page.
# It includes:
#   - Mediterranean zoom button
#   - North Sea zoom button
#   - Automatic dataset switching
#   - Clean projection logic
#   - Shared plotting pipeline
#   - No duplicated code paths
#   - Works with:
#       metocean_scatter_050deg_GLOBAL_monthclim.nc
#       metocean_scatter_050deg_NS_monthclim.nc
#       metocean_scatter_050deg_MED_monthclim.nc
# ---------------------------------------------------------------------------

import streamlit as st
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GLOBAL_DATA_PATH = os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_GLOBAL_monthclim.nc")
NS_DATA_PATH     = os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_NS_monthclim.nc")
MED_DATA_PATH    = os.path.join(BASE_DIR, "..", "metocean_scatter_050deg_MED_monthclim.nc")

# Zoom extents
EXTENTS = {
    "Global": [-180, 180, -90, 90],
    "North Sea": [-13, 35, 52, 76],
    "Mediterranean": [-10, 40, 30, 46],
}

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def load_ds(path):
    return xr.open_dataset(path)


def get_projection(name: str):
    if name == "PlateCarree":
        return ccrs.PlateCarree()
    elif name == "Mercator":
        return ccrs.Mercator()
    elif name == "Orthographic":
        return ccrs.Orthographic(10, 40)
    return ccrs.PlateCarree()


def compute_grid(ds):
    lon_edges = ds["lon3_edges"].values
    lat_edges = ds["lat3_edges"].values
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)
    return lon2d, lat2d


# ----------------------------------------------------------------------------
# SIDEBAR UI
# ----------------------------------------------------------------------------
st.sidebar.header("Map options")

zoom_mode = st.sidebar.radio(
    "Zoom region",
    ["Global", "North Sea", "Mediterranean"],
    index=0
)

proj_choice = st.sidebar.selectbox(
    "Projection",
    ["PlateCarree", "Mercator", "Orthographic"],
    index=0
)

month_choice = st.sidebar.selectbox(
    "Month",
    [1,2,3,4,5,6,7,8,9,10,11,12],
    format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
)

# ----------------------------------------------------------------------------
# DATASET SELECTION
# ----------------------------------------------------------------------------
if zoom_mode == "North Sea":
    ds = load_ds(NS_DATA_PATH)
elif zoom_mode == "Mediterranean":
    ds = load_ds(MED_DATA_PATH)
else:
    ds = load_ds(GLOBAL_DATA_PATH)

lon2d, lat2d = compute_grid(ds)

# Field example: total probability for selected month
arr2d = ds["prob"].sel(month=month_choice).sum(dim=("hs_bin", "tp_bin"))

# ----------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(9, 6))
proj = get_projection(proj_choice)
ax = plt.axes(projection=proj)

ax.add_feature(cfeat.LAND, facecolor="#f2f2f2")
ax.add_feature(cfeat.COASTLINE, linewidth=0.6)
ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6, linestyle="--")

pcm = ax.pcolormesh(
    lon2d, lat2d, arr2d,
    cmap="jet_r",
    shading="auto",
    transform=ccrs.PlateCarree()
)
plt.colorbar(pcm, ax=ax, label="Probability")

# Zoom
ax.set_extent(EXTENTS[zoom_mode], crs=ccrs.PlateCarree())

ax.set_title(f"Metocean Monthly Climatology – {zoom_mode}")
st.pyplot(fig)
