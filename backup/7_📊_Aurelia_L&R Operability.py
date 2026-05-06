# map.py
# -----------------------------------------------------------
# Monthly Location Planner & Operability Comparison
# NOW BASED ON Hs–Tp LIMIT CURVES FROM Limitations.csv
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import st_folium
import folium


# ===========================================================
# PAGE SETUP
# ===========================================================
st.set_page_config(page_title="📍 Operability Planner (Hs–Tp limits)", layout="wide")
st.title("📍 Monthly Location Planner & Operability (Hs–Tp restrictions)")


# ===========================================================
# LOAD METOCEAN NETCDF (cached)
# ===========================================================
@st.cache_resource(show_spinner=False)
def load_metocean(path: str) -> xr.Dataset:
    return xr.open_dataset(path)

nc_path = "metocean_monthclim.nc"
if not os.path.exists(nc_path):
    st.error(f"Could not find NetCDF file: {nc_path}")
    st.stop()

ds = load_metocean(nc_path)

# Basic checks
required = ["prob", "hs_edges", "tp_edges", "lat3_edges", "lon3_edges"]
missing = [k for k in required if k not in ds]
if missing:
    st.error(f"Dataset missing variable(s): {missing}")
    st.stop()

hs_edges = ds["hs_edges"].values
tp_edges = ds["tp_edges"].values
lat_edges = ds["lat3_edges"].values
lon_edges = ds["lon3_edges"].values

# Hs unit fix (cm -> m or suspiciously large)
units = str(ds["hs_edges"].attrs.get("units", "")).lower()
if ("cm" in units) or (np.nanmax(hs_edges) > 50 and "m" not in units):
    hs_edges = hs_edges / 100.0

hs_c = 0.5 * (hs_edges[:-1] + hs_edges[1:])
tp_c = 0.5 * (tp_edges[:-1] + tp_edges[1:])
lat_c_all = 0.5 * (lat_edges[:-1] + lat_edges[1:])
lon_c_all = 0.5 * (lon_edges[:-1] + lon_edges[1:])

# Handle lon convention
# If dataset lon centers span >180, assume 0..360; convert clicked lon to same convention.
lon_is_0360 = (np.nanmax(lon_c_all) > 180)


# ===========================================================
# LOAD LIMITATIONS CSV
# ===========================================================
st.sidebar.header("Hs–Tp Limitations")

default_lim_path = "Limitations.csv"
lim_upload = st.sidebar.file_uploader("Upload Limitations CSV (optional)", type=["csv"])

if lim_upload is not None:
    try:
        lim_df = pd.read_csv(lim_upload)
    except UnicodeDecodeError:
        lim_df = pd.read_csv(lim_upload, encoding="latin-1")
else:
    if not os.path.exists(default_lim_path):
        st.error(f"Could not find {default_lim_path}. Upload a Limitations CSV in the sidebar.")
        st.stop()
    lim_df = pd.read_csv(default_lim_path)

# Normalize column names lightly
lim_df.columns = [c.strip() for c in lim_df.columns]

if "Tp (s)" not in lim_df.columns:
    st.error("Limitations CSV must contain a column named 'Tp (s)'.")
    st.stop()

tp_lim = lim_df["Tp (s)"].astype(float).to_numpy()

# Candidate limit columns: everything except Tp (s)
limit_cols = [c for c in lim_df.columns if c != "Tp (s)"]
if not limit_cols:
    st.error("Limitations CSV has no limit columns besides 'Tp (s)'.")
    st.stop()

chosen_cols = st.sidebar.multiselect(
    "Choose restriction case(s) to evaluate",
    options=limit_cols,
    default=limit_cols[: min(3, len(limit_cols))]
)

within_bin = st.sidebar.checkbox("Within-bin interpolation in Hs (recommended)", value=True)

st.sidebar.caption(
    "Operability is computed as P(Hs ≤ Hs_limit(Tp)) using the joint Hs–Tp probability at each grid point."
)

if not chosen_cols:
    st.info("Select at least one restriction case in the sidebar.")
    st.stop()


# ===========================================================
# LIMIT CURVE PREVIEW
# ===========================================================
st.subheader("📉 Limitation curves (Hs_limit vs Tp)")
fig_lim, ax_lim = plt.subplots(figsize=(9, 3.5))
for col in chosen_cols:
    hs_lim = lim_df[col].astype(float).to_numpy()
    ax_lim.plot(tp_lim, hs_lim, marker="o", linewidth=1.8, label=col)
ax_lim.set_xlabel("Tp (s)")
ax_lim.set_ylabel("Hs limit (m)")
ax_lim.grid(True, alpha=0.3)
ax_lim.legend(ncol=min(3, len(chosen_cols)))
st.pyplot(fig_lim)


# ===========================================================
# MONTHLY LOCATION PLANNER (Folium)
# ===========================================================
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
short = [m[:3].lower() for m in months]

if "coords_by_month" not in st.session_state:
    st.session_state.coords_by_month = {}

st.sidebar.header("Assign Location per Month")

labels = []
label_to_month = {}
for m in months:
    s = m[:3].lower()
    has = s in st.session_state.coords_by_month
    label = f"{'✅' if has else '⬜'} {m}"
    labels.append(label)
    label_to_month[label] = m

remaining = [m for m in months if m[:3].lower() not in st.session_state.coords_by_month]
default_label = f"⬜ {remaining[0]}" if remaining else labels[0]

chosen_labels = st.sidebar.multiselect(
    "Choose month(s) to assign now:",
    options=labels,
    default=[default_label] if default_label in labels else []
)
selected_months = [label_to_month[x] for x in chosen_labels]

st.markdown("### 🌍 Click on the map to set the vessel’s location for the selected month(s):")
m = folium.Map(location=[20, 0], zoom_start=2)

# Existing markers
for mkey, coord in st.session_state.coords_by_month.items():
    folium.Marker(
        coord,
        tooltip=mkey.title(),
        popup=mkey.title(),
        icon=folium.DivIcon(html="📍")
    ).add_to(m)

folium_state = st_folium(m, height=600, width=1200, returned_objects=["last_clicked"])

if selected_months and folium_state and folium_state.get("last_clicked"):
    latlng = (folium_state["last_clicked"]["lat"], folium_state["last_clicked"]["lng"])
    for mon in selected_months:
        st.session_state.coords_by_month[mon[:3].lower()] = latlng
    st.success(f"Saved location for: {', '.join(selected_months)} → {latlng}")
    st.rerun()

if len(st.session_state.coords_by_month) < 12:
    st.warning("Assign a location for all 12 months to compute operability.")
    st.stop()

st.markdown("---")
st.subheader("📊 Monthly Operability per Restriction Case (Hs–Tp limits)")


# ===========================================================
# HELPERS
# ===========================================================
def normalize_pdf(prob2d: xr.DataArray) -> xr.DataArray:
    tot = prob2d.sum(dim=("hs_bin", "tp_bin"))
    return xr.where(tot > 0, prob2d / tot, 0)

def nearest_grid(lat_sel: float, lon_sel: float):
    # Match lon convention to dataset
    lon_use = (lon_sel + 360.0) % 360.0 if lon_is_0360 else lon_sel
    i_lat = int(np.abs(lat_c_all - lat_sel).argmin())
    i_lon = int(np.abs(lon_c_all - lon_use).argmin())
    return i_lat, i_lon, lon_use

def interp_limit_to_tp(tp_centers: np.ndarray, tp_lim_1d: np.ndarray, hs_lim_1d: np.ndarray):
    # np.interp clamps to end values outside range
    return np.interp(tp_centers, tp_lim_1d, hs_lim_1d)

def operability_ht(prob2d: xr.DataArray, hs_edges_arr: np.ndarray, tp_centers: np.ndarray,
                   tp_lim_1d: np.ndarray, hs_lim_1d: np.ndarray, within_bin_interp=True) -> float:
    """
    prob2d dims: (hs_bin, tp_bin), normalized or not.
    Computes 100 * P(Hs <= Hs_limit(Tp)) with optional within-bin interpolation in Hs.
    """
    prob2d = normalize_pdf(prob2d)

    hs_limit_tp = interp_limit_to_tp(tp_centers, tp_lim_1d, hs_lim_1d)
    lim_da = xr.DataArray(hs_limit_tp, dims=["tp_bin"])

    if not within_bin_interp:
        # Stepwise: whole-bin inclusion if bin center <= limit
        hs_cent = xr.DataArray(0.5 * (hs_edges_arr[:-1] + hs_edges_arr[1:]), dims=["hs_bin"])
        mask = (hs_cent <= lim_da)  # broadcasts to (hs_bin, tp_bin)
        return float((prob2d.where(mask, 0.0)).sum().values) * 100.0

    # Within-bin interpolation using fraction of each Hs bin that is <= limit
    hs_low = xr.DataArray(hs_edges_arr[:-1], dims=["hs_bin"])
    hs_high = xr.DataArray(hs_edges_arr[1:], dims=["hs_bin"])
    width = hs_high - hs_low

    frac = (lim_da - hs_low) / width  # broadcasts to (hs_bin, tp_bin)
    frac = frac.clip(0.0, 1.0)

    p_ok = (prob2d * frac).sum(dim=("hs_bin", "tp_bin"))
    return float(p_ok.values) * 100.0


# ===========================================================
# COMPUTE OPERABILITY FOR EACH MONTH & EACH LIMIT CASE
# ===========================================================
results = []

# Determine how to index month dimension robustly
prob_var = ds["prob"]
has_month = "month" in prob_var.dims
if not has_month:
    st.error("Dataset variable 'prob' has no 'month' dimension; expected monthly climatology.")
    st.stop()

# Identify lat/lon dims
lat_dim = "lat3_bin" if "lat3_bin" in prob_var.dims else None
lon_dim = "lon3_bin" if "lon3_bin" in prob_var.dims else None
if lat_dim is None or lon_dim is None:
    st.error("Dataset 'prob' must have dimensions 'lat3_bin' and 'lon3_bin'.")
    st.stop()

for idx, month in enumerate(months):
    key = month[:3].lower()
    lat_sel, lon_sel = st.session_state.coords_by_month[key]
    i_lat, i_lon, lon_use = nearest_grid(lat_sel, lon_sel)

    # Select the monthly Hs–Tp joint distribution at nearest grid point
    prob_ht = (
        prob_var
        .isel(month=idx, **{lat_dim: i_lat, lon_dim: i_lon})
    )

    # Ensure (hs_bin, tp_bin) ordering exists
    if not (("hs_bin" in prob_ht.dims) and ("tp_bin" in prob_ht.dims)):
        st.error("Expected prob to include 'hs_bin' and 'tp_bin' dimensions.")
        st.stop()

    prob_ht = prob_ht.transpose("hs_bin", "tp_bin")

    row = {"Month": month, "Lat": float(lat_sel), "Lon": float(lon_sel)}

    for col in chosen_cols:
        hs_lim_1d = lim_df[col].astype(float).to_numpy()
        row[col] = operability_ht(
            prob_ht, hs_edges, tp_c, tp_lim, hs_lim_1d, within_bin_interp=within_bin
        )

    results.append(row)

df_out = pd.DataFrame(results)
df_disp = df_out.drop(columns=["Lat", "Lon"])

st.dataframe(df_disp.set_index("Month").style.format("{:.1f}"), use_container_width=True)


# ===========================================================
# PLOT COMPARISON
# ===========================================================
st.markdown("### 📈 Operability Comparison (Hs–Tp limits)")

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(12)

for col in chosen_cols:
    ax.plot(x, df_out[col].values, marker="o", label=str(col))

ax.set_xticks(x)
ax.set_xticklabels(months, rotation=45, ha="right")
ax.set_ylabel("Operability (%)")
ax.set_title("Monthly Operability per Restriction Case")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)


# ===========================================================
# ANNUAL MEAN SUMMARY
# ===========================================================
st.markdown("### 🏁 Annual Mean Operability per Restriction Case")

annual = {col: float(df_out[col].mean()) for col in chosen_cols}
df_ann = pd.DataFrame({"Case": list(annual.keys()), "Annual Operability (%)": list(annual.values())})
st.dataframe(df_ann.set_index("Case").style.format("{:.1f}"), use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(list(annual.keys()), list(annual.values()), color="steelblue")
ax2.set_title("Annual Mean Operability")
ax2.set_ylabel("%")
ax2.set_ylim(0, 100)
ax2.grid(True, axis="y", alpha=0.25)
st.pyplot(fig2)